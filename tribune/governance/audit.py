"""Append-only, hash-chained, inspectable audit log.

Every step of a case run writes a record here. Records are chained by hash
(``record_hash = sha256(prev_hash + canonical(record))``) so any later tampering
with an earlier record is detectable via :meth:`AuditLog.verify_chain`. A person
or caseworker can read the full chain to see exactly why TRIBUNE said what it said.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..types import AuditRecord, SMState
from .judge import JudgeEvaluator, JudgeResult, get_default_judge

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # Email
    re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),  # Phone
    re.compile(r"\b(?:BEN|MEMBER|CLIENT|APPLICANT)-\d{4,}\b", re.IGNORECASE),  # Benefit ID
    re.compile(r"\b(?:sk|pk|api_key|secret_key|key|token)-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),  # API key
    re.compile(r"\bBearer\s+[A-Za-z0-9\._\-]+\b", re.IGNORECASE),  # Bearer token
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),  # JWT
]


def sanitize_audit_text(text: str) -> str:
    """Sanitize PII and credential patterns in strings."""
    if not isinstance(text, str):
        return text
    clean = text
    for pattern in _PII_PATTERNS:
        clean = pattern.sub("[REDACTED]", clean)
    return clean


def sanitize_audit_data(data: Any) -> Any:
    """Recursively sanitize PII and credentials in event dictionaries and telemetry data."""
    if isinstance(data, str):
        return sanitize_audit_text(data)
    elif isinstance(data, dict):
        return {k: sanitize_audit_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_audit_data(x) for x in data]
    return data


def _canonical(record: AuditRecord) -> str:
    body = {
        "case_id": record.case_id,
        "sequence": record.sequence,
        "state": record.state.value,
        "agent": record.agent,
        "action": record.action,
        "model_name": record.model_name,
        "model_version": record.model_version,
        "citation_ids": list(record.citation_ids),
        "evidence_ids": list(record.evidence_ids),
        "payload": record.payload,
        "prev_hash": record.prev_hash,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _hash(record: AuditRecord) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self) -> None:
        self._by_case: dict[str, list[AuditRecord]] = {}
        self._judge_results: dict[str, list[JudgeResult]] = {}

    def append(
        self,
        case_id: str,
        state: SMState,
        agent: str,
        action: str,
        *,
        model_name: str = "n/a",
        model_version: str = "n/a",
        citation_ids: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        payload: dict[str, str] | None = None,
    ) -> AuditRecord:
        chain = self._by_case.setdefault(case_id, [])
        prev_hash = chain[-1].record_hash if chain else ""
        clean_action = sanitize_audit_text(action)
        clean_payload = sanitize_audit_data(payload or {})
        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            case_id=case_id,
            sequence=len(chain),
            state=state,
            agent=agent,
            action=clean_action,
            model_name=model_name,
            model_version=model_version,
            citation_ids=citation_ids or [],
            evidence_ids=evidence_ids or [],
            payload=clean_payload,
            prev_hash=prev_hash,
        )
        record = record.model_copy(update={"record_hash": _hash(record)})
        chain.append(record)
        return record

    def evaluate_and_log_verifier(
        self,
        case_id: str,
        assessment: Any,
        verdict: Any,
        evidence: list[Any],
        jurisdiction: str,
        judge: JudgeEvaluator | None = None,
    ) -> tuple[AuditRecord, JudgeResult]:
        """Live execution hook: Evaluate verifier output using a specialized audit judge and log structured telemetry."""
        judge_eval = judge or get_default_judge()
        judge_res = judge_eval.evaluate(assessment, verdict, evidence, jurisdiction)
        self._judge_results.setdefault(case_id, []).append(judge_res)

        payload = {
            "program": str(getattr(assessment, "program", "unknown")),
            "judge_name": str(judge_res.judge_name),
            "judge_version": str(judge_res.judge_version),
            "passed": str(judge_res.passed),
            "perceived_error_score": f"{judge_res.perceived_error_score:.4f}",
            "uncited_claim_score": f"{judge_res.uncited_claim_score:.4f}",
            "citation_coverage_score": f"{judge_res.citation_coverage_score:.4f}",
            "rule_reference_coverage": f"{judge_res.rule_reference_coverage:.4f}",
            "judge_confidence": f"{judge_res.judge_confidence:.4f}",
            "judge_cost_usd": f"{judge_res.cost_estimate:.6f}",
            "reasons": "; ".join(judge_res.reasons)[:300],
        }
        cit_ids = [c.citation_id for c in getattr(assessment, "citations", [])]
        record = self.append(
            case_id=case_id,
            state=SMState.VERIFY,
            agent="verifier_judge",
            action=f"continuous audit evaluation ({judge_res.judge_name}: passed={judge_res.passed})",
            model_name=judge_res.judge_name,
            model_version=judge_res.judge_version,
            citation_ids=cit_ids,
            evidence_ids=getattr(assessment, "evidence_ids", []),
            payload=payload,
        )
        return record, judge_res

    def query(
        self,
        case_id: str | None = None,
        state: SMState | None = None,
        agent: str | None = None,
        search_text: str | None = None,
    ) -> list[AuditRecord]:
        """Query audit records with structured filters."""
        cases = [case_id] if case_id else list(self._by_case.keys())
        results: list[AuditRecord] = []
        for cid in cases:
            for rec in self._by_case.get(cid, []):
                if state is not None and rec.state != state:
                    continue
                if agent is not None and rec.agent != agent:
                    continue
                if search_text is not None and (search_text.lower() not in rec.action.lower()):
                    continue
                results.append(rec)
        return results

    def judge_results(self, case_id: str | None = None) -> list[JudgeResult]:
        """Retrieve all structured judge evaluation results."""
        if case_id:
            return list(self._judge_results.get(case_id, []))
        all_res: list[JudgeResult] = []
        for res_list in self._judge_results.values():
            all_res.extend(res_list)
        return all_res

    def calculate_live_metrics(self, case_id: str | None = None) -> dict[str, float]:
        """Calculate real-time aggregated metrics across all evaluated verifier traces."""
        judges = self.judge_results(case_id)
        if not judges:
            return {
                "perceived_error_rate": 0.0,
                "uncited_claim_rate": 0.0,
                "mean_citation_coverage": 1.0,
                "mean_rule_reference_coverage": 1.0,
                "mean_judge_confidence": 1.0,
                "total_judge_cost_usd": 0.0,
                "eval_count": 0.0,
            }

        n = len(judges)
        return {
            "perceived_error_rate": round(sum(j.perceived_error_score for j in judges) / n, 4),
            "uncited_claim_rate": round(sum(j.uncited_claim_score for j in judges) / n, 4),
            "mean_citation_coverage": round(sum(j.citation_coverage_score for j in judges) / n, 4),
            "mean_rule_reference_coverage": round(sum(j.rule_reference_coverage for j in judges) / n, 4),
            "mean_judge_confidence": round(sum(j.judge_confidence for j in judges) / n, 4),
            "total_judge_cost_usd": round(sum(j.cost_estimate for j in judges), 6),
            "eval_count": float(n),
        }

    def export_jsonl(self, path: str, case_id: str | None = None) -> int:
        """Export queryable audit log records to JSONL file."""
        records = self.query(case_id=case_id)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.model_dump(mode="json"), default=str) + "\n")
        return len(records)

    def load_records(self, case_id: str, records: list[AuditRecord]) -> None:
        """Restore audit records into the in-memory chain during crash recovery."""
        self._by_case[case_id] = list(records)

    def records(self, case_id: str) -> list[AuditRecord]:
        return list(self._by_case.get(case_id, []))

    def verify_chain(self, case_id: str) -> bool:
        prev = ""
        for rec in self._by_case.get(case_id, []):
            if rec.prev_hash != prev:
                return False
            expected = _hash(rec.model_copy(update={"record_hash": ""}))
            if expected != rec.record_hash:
                return False
            prev = rec.record_hash
        return True


class CheckpointManager:
    """Zero-data-loss atomic state-write operations and crash recovery manager.

    Writes execution checkpoints atomically to `.tribune/last_run.json` using a temporary file
    and atomic file replacement to protect against corruptions during unexpected process crashes.
    """

    DEFAULT_PATH = ".tribune/last_run.json"

    @classmethod
    def save_checkpoint(
        cls,
        case_id: str,
        jurisdiction: str,
        dag_dict: dict[str, Any],
        completed_task_ids: list[str],
        outcomes_data: list[dict[str, Any]],
        memory_snapshot: dict[str, Any],
        audit_records: list[AuditRecord],
        metadata: dict[str, Any] | None = None,
        path: str | None = None,
    ) -> str:
        """Atomically persist a milestone checkpoint."""
        import os

        final_path = path or cls.DEFAULT_PATH
        dirname = os.path.dirname(final_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        temp_path = f"{final_path}.tmp.{uuid.uuid4().hex}"
        payload = {
            "checkpoint_version": 1,
            "timestamp": _canonical_now(),
            "case_id": case_id,
            "jurisdiction": jurisdiction,
            "dag": dag_dict,
            "completed_task_ids": completed_task_ids,
            "outcomes": outcomes_data,
            "memory_snapshot": memory_snapshot,
            "audit_records": [r.model_dump(mode="json") for r in audit_records],
            "metadata": metadata or {},
        }

        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(temp_path, final_path)
        return final_path

    @classmethod
    def load_checkpoint(cls, path: str | None = None) -> dict[str, Any] | None:
        """Load and deserialize checkpoint from disk, returning None if absent or corrupt."""
        import os

        target_path = path or cls.DEFAULT_PATH
        if not os.path.exists(target_path):
            return None
        try:
            with open(target_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    @classmethod
    def clear_checkpoint(cls, path: str | None = None) -> bool:
        """Remove checkpoint after successful execution completion."""
        import os

        target_path = path or cls.DEFAULT_PATH
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                return True
            except OSError:
                pass
        return False


def _canonical_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()



# --------------------------------------------------------------------------- #
# Append-Only Disk-Backed Transactional State Journaling
# --------------------------------------------------------------------------- #


@dataclass
class JournalStateBoundaryRecord:
    """An append-only disk journal entry recorded at a discrete state boundary."""

    journal_id: str
    case_id: str
    sequence: int
    state: str  # e.g. SMState.PLAN.value
    timestamp: str
    environment_variables: dict[str, str]
    tool_outputs: dict[str, Any]
    dag_snapshot: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "journal_id": self.journal_id,
            "case_id": self.case_id,
            "sequence": self.sequence,
            "state": self.state,
            "timestamp": self.timestamp,
            "environment_variables": self.environment_variables,
            "tool_outputs": self.tool_outputs,
            "dag_snapshot": self.dag_snapshot,
            "metadata": self.metadata,
            "entry_hash": self.entry_hash,
        }


class DiskStateJournal:
    """Append-only, disk-backed transactional journal eliminating environment state amnesia.

    Serializes all environment variables, tool outputs, and dependency graphs at every
    discrete state boundary (PLAN -> GATHER -> ASSESS -> VERIFY -> PREPARE -> DONE).
    Supports exact replay and state re-hydration upon crash recovery.
    """

    DEFAULT_JOURNAL_PATH = ".tribune/state_journal.jsonl"

    def __init__(self, journal_path: str | None = None) -> None:
        self.journal_path = journal_path or self.DEFAULT_JOURNAL_PATH
        self._lock = threading.RLock()
        self._sequence_counter: dict[str, int] = {}

    def record_state_boundary(
        self,
        case_id: str,
        state: SMState | str,
        environment_variables: dict[str, str] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        dag_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JournalStateBoundaryRecord:
        """Atomically append environment variables, tool outputs, and DAG state to disk journal."""
        with self._lock:
            state_str = state.value if isinstance(state, SMState) else str(state)
            seq = self._sequence_counter.get(case_id, 0) + 1
            self._sequence_counter[case_id] = seq

            # Sanitize environment variables (remove sensitive tokens/keys)
            clean_env: dict[str, str] = {}
            if environment_variables:
                for k, v in environment_variables.items():
                    if any(secret_term in k.lower() for secret_term in ("key", "secret", "token", "password")):
                        clean_env[k] = "[REDACTED]"
                    else:
                        clean_env[k] = str(v)

            clean_tools = sanitize_audit_data(tool_outputs or {})
            clean_dag = dag_snapshot or {}

            journal_id = f"jnl_{case_id}_{seq}_{uuid.uuid4().hex[:8]}"
            timestamp = _canonical_now()

            # Hash entry for tamper detection
            content = f"{journal_id}:{case_id}:{seq}:{state_str}:{json.dumps(clean_tools, sort_keys=True, default=str)}"
            entry_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            record = JournalStateBoundaryRecord(
                journal_id=journal_id,
                case_id=case_id,
                sequence=seq,
                state=state_str,
                timestamp=timestamp,
                environment_variables=clean_env,
                tool_outputs=clean_tools,
                dag_snapshot=clean_dag,
                metadata=metadata or {},
                entry_hash=entry_hash,
            )

            # Atomic append to disk with fsync
            dirname = os.path.dirname(os.path.abspath(self.journal_path))
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            return record

    def recover_state(self, case_id: str) -> dict[str, Any] | None:
        """Replay disk journal to recover latest environment variables, tool outputs, and DAG state."""
        with self._lock:
            if not os.path.exists(self.journal_path):
                return None

            latest_record: dict[str, Any] | None = None
            all_records: list[dict[str, Any]] = []

            with open(self.journal_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("case_id") == case_id:
                            latest_record = rec
                            all_records.append(rec)
                    except Exception:
                        continue

            if not latest_record:
                return None

            # Merge all cumulative tool outputs across state boundaries to prevent amnesia
            merged_tool_outputs: dict[str, Any] = {}
            for r in all_records:
                merged_tool_outputs.update(r.get("tool_outputs", {}))

            return {
                "case_id": case_id,
                "latest_state": latest_record.get("state"),
                "sequence": latest_record.get("sequence"),
                "timestamp": latest_record.get("timestamp"),
                "environment_variables": latest_record.get("environment_variables", {}),
                "tool_outputs": merged_tool_outputs,
                "dag_snapshot": latest_record.get("dag_snapshot", {}),
                "total_state_boundaries_replayed": len(all_records),
            }

    def clear(self) -> None:
        """Clear the journal file."""
        with self._lock:
            if os.path.exists(self.journal_path):
                try:
                    os.remove(self.journal_path)
                except OSError:
                    pass
            self._sequence_counter.clear()


__all__ = [
    "AuditLog",
    "CheckpointManager",
    "JournalStateBoundaryRecord",
    "DiskStateJournal",
    "sanitize_audit_text",
    "sanitize_audit_data",
]

