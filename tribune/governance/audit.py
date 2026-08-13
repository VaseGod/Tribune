"""Append-only, hash-chained, inspectable audit log.

Every step of a case run writes a record here. Records are chained by hash
(``record_hash = sha256(prev_hash + canonical(record))``) so any later tampering
with an earlier record is detectable via :meth:`AuditLog.verify_chain`. A person
or caseworker can read the full chain to see exactly why TRIBUNE said what it said.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from ..types import AuditRecord, SMState

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
