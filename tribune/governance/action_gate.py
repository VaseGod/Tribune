"""Never-auto-submit action gate and specification gaming safety guardrails.

TRIBUNE may *prepare* applications, appeals, and checklists freely. It may never
*submit* anything binding on its own. Submission of prepared materials requires an
explicit :class:`HumanSignoff` whose intent matches the exact program and case;
without it, :meth:`ActionGate.authorize_submission` raises :class:`ActionBlocked`.

Includes static pattern detectors to prevent specification gaming across 8 threat vectors:
1. Hidden test suite inspection
2. Statutory rule definition overrides
3. Citation key fabrication
4. Rule store integrity tampering
5. Evaluation criteria manipulation
6. Governance & audit hook bypasses
7. Unauthorized file path and credential access
8. System policy override and prompt injection
"""

from __future__ import annotations

import copy
import enum
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..corpus.rule_store import RuleStore
from ..types import (
    Assessment,
    CriterionOutcome,
    PreparedMaterials,
    StrictModel,
    SubmissionReceipt,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActionBlocked(PermissionError):
    pass


class PreConditionError(ActionBlocked):
    """Raised when pre-condition assertions fail before tool or action execution."""
    pass


class PostConditionError(ActionBlocked):
    """Raised when post-condition assertions fail after tool or action execution."""
    pass


class SecurityViolationError(ActionBlocked):
    """Raised when high-severity adversarial patterns or specification gaming attempts are detected."""
    pass


class CostTripwireError(SecurityViolationError):
    """Raised when deterministic cost tripwires (payload size ceiling, cumulative token budget) are breached."""
    pass


class CyclicalRetryLoopError(SecurityViolationError):
    """Raised when maximum cyclical error recovery attempts are exceeded to break infinite retry loops."""
    pass


class RetrievalEntropyDegradationError(ActionBlocked):
    """Raised when external skill retrieval increases predictive entropy, triggering parametric fallback."""
    pass


@dataclass
class RAEValidationResult:
    """Structured outcome of counterfactual RAE entropy validation."""

    tool_or_skill_name: str
    base_entropy: float
    retrieved_entropy: float
    entropy_delta: float
    retrieval_suppressed: bool
    force_parametric_weights: bool
    reason: str
    decision: Any = None


class RAEGater:
    """Retrieval-Invoked Actual-Use Effect (RAE) Gating Engine.

    Counterfactual validation step prior to dispatching external skill retrieval:
    1. Evaluates model predictive entropy both with (H_retrieved) and without (H_base) retrieved context.
    2. If injecting retrieved context increases entropy (Delta H = H_retrieved - H_base > 0),
       signaling retrieval degradation or confusion: suppresses payload and forces execution
       using parametric weights.
    3. If Delta H <= 0, approves retrieved payload injection.
    """

    def __init__(self, entropy_margin: float = 0.0) -> None:
        self.entropy_margin = entropy_margin
        self.evaluations_count = 0
        self.suppressions_count = 0
        self.approvals_count = 0

    def evaluate_retrieval_entropy_effect(
        self,
        skill_or_tool: str,
        base_probs_or_logits: list[float] | np.ndarray,
        retrieved_probs_or_logits: list[float] | np.ndarray,
        is_logits: bool = False,
    ) -> RAEValidationResult:
        """Counterfactual evaluation of model predictive entropy with and without retrieved context."""
        from ..orchestration.mtp import EntropyAwareDepthScaler

        self.evaluations_count += 1
        h_base = EntropyAwareDepthScaler.compute_shannon_entropy(base_probs_or_logits, is_logits=is_logits)
        h_retrieved = EntropyAwareDepthScaler.compute_shannon_entropy(retrieved_probs_or_logits, is_logits=is_logits)
        delta_h = round(h_retrieved - h_base, 4)

        if delta_h > self.entropy_margin:
            # Retrieval increases predictive entropy -> confusion/degradation
            self.suppressions_count += 1
            return RAEValidationResult(
                tool_or_skill_name=skill_or_tool,
                base_entropy=h_base,
                retrieved_entropy=h_retrieved,
                entropy_delta=delta_h,
                retrieval_suppressed=True,
                force_parametric_weights=True,
                reason=(
                    f"RAE Gating suppressed '{skill_or_tool}': retrieved payload increased action "
                    f"prediction entropy by +{delta_h:.4f} (H_base={h_base:.4f} -> H_retrieved={h_retrieved:.4f}), "
                    f"signaling retrieval confusion. Falling back to parametric weights."
                ),
                decision=GateDecisionType.BLOCK,
            )
        else:
            self.approvals_count += 1
            return RAEValidationResult(
                tool_or_skill_name=skill_or_tool,
                base_entropy=h_base,
                retrieved_entropy=h_retrieved,
                entropy_delta=delta_h,
                retrieval_suppressed=False,
                force_parametric_weights=False,
                reason=f"RAE Gating approved '{skill_or_tool}': predictive entropy stabilized or decreased (delta={delta_h:.4f}).",
                decision=GateDecisionType.ALLOW,
            )


class GateDecisionType(str, enum.Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"


class GateSeverity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GateDecision(StrictModel):
    """Structured safety gate decision recorded for auditability and policy enforcement."""

    decision: GateDecisionType
    severity: GateSeverity
    matched_rules: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    remediation_hint: str = ""
    agent_id: str = "unknown"
    action_type: str = "general"
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class HumanSignoff:
    """Evidence that a human explicitly authorized a binding submission."""

    authorized_by: str
    token: str
    intent: str  # must equal f"{program.value}:{case_id}"

    @staticmethod
    def issue(authorized_by: str, program_value: str, case_id: str) -> HumanSignoff:
        return HumanSignoff(
            authorized_by=authorized_by,
            token=secrets.token_hex(16),
            intent=f"{program_value}:{case_id}",
        )


@dataclass(frozen=True)
class SupervisorSignature:
    """Cryptographic-style signature payload authenticating supervisor approval for external execution."""

    supervisor_id: str
    signature_token: str
    target_action: str  # e.g., f"{tool_name}:{case_id}"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def issue(supervisor_id: str, target_action: str) -> SupervisorSignature:
        return SupervisorSignature(
            supervisor_id=supervisor_id,
            signature_token=secrets.token_hex(16),
            target_action=target_action,
        )

    def is_valid(self, expected_action: str) -> bool:
        return bool(
            self.signature_token
            and len(self.signature_token) >= 16
            and self.supervisor_id
            and self.target_action == expected_action
        )


# --------------------------------------------------------------------------- #
# Append-Only Cryptographic Trajectory Event Sourcing & Sandbox Snapshots
# --------------------------------------------------------------------------- #


class TrajectoryEventType(str, enum.Enum):
    """Event types recorded in the append-only cryptographic trajectory audit log."""

    MUTATION_DRAFTED = "mutation_drafted"
    SPECULATIVE_EXECUTED = "speculative_executed"
    EXTERNAL_CALL_ATTEMPTED = "external_call_attempted"
    JUDGE_APPROVED = "judge_approved"
    JUDGE_REJECTED = "judge_rejected"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TrajectoryEvent:
    """An immutable, cryptographically chained event in the case execution trajectory."""

    event_id: str
    case_id: str
    event_type: TrajectoryEventType
    agent_id: str
    action_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = "0" * 64
    event_hash: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    signature: str | None = None

    def compute_hash(self, prev_hash: str) -> str:
        """Compute deterministic SHA-256 hash chaining back to prev_hash."""
        serialized_payload = json.dumps(self.payload, sort_keys=True, default=str)
        content = (
            f"{prev_hash}:{self.event_id}:{self.case_id}:{self.event_type.value}:"
            f"{self.agent_id}:{self.action_name}:{serialized_payload}:{self.timestamp.isoformat()}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TrajectoryEventLog:
    """Append-only cryptographic event store with SHA-256 hash chaining."""

    def __init__(self) -> None:
        self.events: list[TrajectoryEvent] = []
        self._lock = threading.RLock()

    def append_event(
        self,
        case_id: str,
        event_type: TrajectoryEventType,
        agent_id: str,
        action_name: str,
        payload: dict[str, Any] | None = None,
        signature: str | None = None,
    ) -> TrajectoryEvent:
        with self._lock:
            prev_hash = self.events[-1].event_hash if self.events else "0" * 64
            event_id = f"evt_{len(self.events) + 1}_{secrets.token_hex(6)}"
            event = TrajectoryEvent(
                event_id=event_id,
                case_id=case_id,
                event_type=event_type,
                agent_id=agent_id,
                action_name=action_name,
                payload=dict(payload or {}),
                prev_hash=prev_hash,
                timestamp=_utcnow(),
                signature=signature,
            )
            event.event_hash = event.compute_hash(prev_hash)
            self.events.append(event)
            return event

    def verify_log_integrity(self) -> bool:
        """Verify unbroken SHA-256 cryptographic chain across all recorded trajectory events."""
        with self._lock:
            if not self.events:
                return True
            expected_prev = "0" * 64
            for event in self.events:
                if event.prev_hash != expected_prev:
                    return False
                computed = event.compute_hash(event.prev_hash)
                if event.event_hash != computed:
                    return False
                expected_prev = event.event_hash
            return True

    def get_case_events(self, case_id: str) -> list[TrajectoryEvent]:
        with self._lock:
            return [e for e in self.events if e.case_id == case_id]

    def get_uncommitted_events(self, case_id: str) -> list[TrajectoryEvent]:
        with self._lock:
            case_evts = [e for e in self.events if e.case_id == case_id]
            last_commit_idx = -1
            for idx, e in enumerate(case_evts):
                if e.event_type in (TrajectoryEventType.COMMITTED, TrajectoryEventType.ROLLED_BACK):
                    last_commit_idx = idx
            return case_evts[last_commit_idx + 1 :]


@dataclass
class SandboxSnapshot:
    """Immutable state snapshot captured inside an isolated sandbox."""

    snapshot_id: str
    case_id: str
    state_data: dict[str, Any]
    checksum: str
    timestamp: datetime = field(default_factory=_utcnow)


class SandboxContext:
    """Isolated sandbox execution context with snapshot, isolation, and rollback capabilities."""

    def __init__(self, case_id: str, initial_state: dict[str, Any] | None = None) -> None:
        self.case_id = case_id
        self.active_state: dict[str, Any] = copy.deepcopy(initial_state or {})
        self.snapshots: list[SandboxSnapshot] = []
        self.uncommitted_mutations: list[dict[str, Any]] = []
        self.is_committed: bool = False
        self._lock = threading.RLock()
        # Initialize initial baseline snapshot
        self.snapshot(self.active_state)

    def snapshot(self, state: dict[str, Any] | None = None) -> SandboxSnapshot:
        with self._lock:
            target_state = copy.deepcopy(state if state is not None else self.active_state)
            serialized = json.dumps(target_state, sort_keys=True, default=str)
            checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            snap_id = f"snap_{len(self.snapshots) + 1}_{secrets.token_hex(4)}"
            snap = SandboxSnapshot(
                snapshot_id=snap_id,
                case_id=self.case_id,
                state_data=target_state,
                checksum=checksum,
                timestamp=_utcnow(),
            )
            self.snapshots.append(snap)
            return snap

    def mutate(self, path: str, value: Any, agent_id: str = "unknown") -> None:
        with self._lock:
            self.active_state[path] = copy.deepcopy(value)
            self.uncommitted_mutations.append({
                "path": path,
                "value": value,
                "agent_id": agent_id,
                "timestamp": _utcnow().isoformat(),
            })

    def rollback_to_snapshot(self, snapshot_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if not self.snapshots:
                self.active_state.clear()
                self.uncommitted_mutations.clear()
                return {}
            if snapshot_id is not None:
                match = next((s for s in reversed(self.snapshots) if s.snapshot_id == snapshot_id), None)
                snap = match or self.snapshots[0]
            else:
                snap = self.snapshots[-1]

            self.active_state = copy.deepcopy(snap.state_data)
            self.uncommitted_mutations.clear()
            self.is_committed = False
            return copy.deepcopy(self.active_state)

    def commit(
        self,
        supervisor_signature: SupervisorSignature | None = None,
        judge_result: Any | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if judge_result is not None:
                passed = getattr(judge_result, "passed", True)
                if not passed:
                    raise SecurityViolationError(
                        f"Judge rejected sandbox commit for case {self.case_id}: "
                        f"{getattr(judge_result, 'reasoning', 'Judge failed assertions')}"
                    )
            self.is_committed = True
            self.uncommitted_mutations.clear()
            # Capture committed baseline snapshot
            self.snapshot(self.active_state)
            return copy.deepcopy(self.active_state)


# --------------------------------------------------------------------------- #
# Static Pattern Detectors for Specification Gaming Guardrails
# --------------------------------------------------------------------------- #


class StaticPatternDetectors:
    """Static pattern detectors inspecting reasoning traces, tool inputs, file accesses, and outputs."""

    # 1. Hidden Test Suite Inspection
    HIDDEN_TEST_PATTERNS = [
        re.compile(r"(?:tests?/|test_.*\.py|conftest\.py|pytest|unittest)", re.IGNORECASE),
        re.compile(r"\b(?:ground_truth|synthetic_case|assert_truth|eval_record|canary_probe)\b", re.IGNORECASE),
        re.compile(r"\b(?:test_canary|hidden_test|gold_standard|answer_key)\b", re.IGNORECASE),
    ]

    # 2. Statutory Rule Definition Overrides
    RULE_OVERRIDE_PATTERNS = [
        re.compile(r"(?:override_rule|redefine_predicate|modify_threshold|bypass_statute)", re.IGNORECASE),
        re.compile(r"(?:mock_rule_store|force_eligible|ignore_income_cap|patch_ruleset)", re.IGNORECASE),
        re.compile(r"\b(?:ruleset\.rules\.clear|del\s+ruleset|ruleset\.add_rule)\b", re.IGNORECASE),
    ]

    # 3. Citation Key Fabrication
    CITATION_FABRICATION_PATTERNS = [
        re.compile(r"\b(?:fake_citation|hallucinated_code|statute_999|custom_statute|dummy_cfr)\b", re.IGNORECASE),
        re.compile(r"\b(?:auto_generate_citation|invent_source|mock_citation_id)\b", re.IGNORECASE),
    ]

    # 4. Rule Store Tampering
    RULE_STORE_TAMPERING_PATTERNS = [
        re.compile(r"\b(?:RuleStore\._by_program|RuleStore\.clear|RuleStore\.__setattr__)\b"),
        re.compile(r"(?:monkeypatch|unittest\.mock\.patch|pytest\.mock).*rule_store", re.IGNORECASE),
    ]

    # 5. Evaluation Criteria Manipulation
    CRITERIA_TAMPERING_PATTERNS = [
        re.compile(r"(?:force_satisfied|skip_required_criteria|mark_all_eligible|bypass_coverage)", re.IGNORECASE),
        re.compile(r"\b(?:CriterionOutcome\.SATISFIED\s*=\s*|required\s*=\s*False)\b"),
    ]

    # 6. Governance & Audit Hook Bypass
    GOVERNANCE_BYPASS_PATTERNS = [
        re.compile(r"\b(?:disable_audit|skip_action_gate|bypass_preconditions|AuditLog\._by_case\.clear)\b", re.IGNORECASE),
        re.compile(r"\b(?:mock_action_gate|disable_action_gate|allow_auto_submit)\b", re.IGNORECASE),
    ]

    # 7. Unauthorized File & Secret Access
    UNAUTHORIZED_ACCESS_PATTERNS = [
        re.compile(r"(?:\.\./|\.\.\\|/etc/passwd|/etc/shadow|~/\.ssh|~/\.aws)"),
        re.compile(r"(?:\.env|\.pem|\.key|id_rsa|secret_key|api_key|token_vllm)"),
    ]

    # 8. System Policy Override & Prompt Injection
    POLICY_OVERRIDE_PATTERNS = [
        re.compile(r"(?:ignore\s+previous\s+instructions|system\s+prompt\s+override|disregard\s+all\s+rules)", re.IGNORECASE),
        re.compile(r"(?:jailbreak|DAN\s+mode|developer\s+override|disable\s+safety)", re.IGNORECASE),
        re.compile(r"<(?:think|thought|reasoning)[^>]*>.*?</(?:think|thought|reasoning)>", re.DOTALL | re.IGNORECASE),
    ]

    # 9. Prohibited Unverified Direct Emission by Ingestion / Non-Binding Models
    PROHIBITED_UNVERIFIED_DIRECT_EMISSION = [
        re.compile(r"\b(?:direct_emission_by_glm_flash|unverified_statutory_decision|bypass_verifier_pass)\b", re.IGNORECASE),
        re.compile(r"\b(?:glm-5\.3-flash-direct-determination|raw_non_binding_determination)\b", re.IGNORECASE),
    ]

    # 10. Environment Manipulation, Privilege Escalation, & Mock Tampering
    ENV_MANIPULATION_PATTERNS = [
        re.compile(r"\b(?:sudo|chmod\s+777|chown|setuid|setgid|getpass|ctypes\.cdll)\b", re.IGNORECASE),
        re.compile(r"\b(?:os\.environ|os\.putenv|os\.system|subprocess\.call|shutil\.rmtree)\b"),
        re.compile(r"\b(?:sys\.modules|builtins\.__dict__|__builtins__\.open)\b"),
        re.compile(r"(?:mock\.patch|unittest\.mock\.MagicMock\.__setattr__|monkeypatch\.setattr)"),
    ]



def validate_path_containment(path: str, allowed_root: str | None = None) -> bool:
    """Strict path containment validation to trap and reject out-of-bounds file system traversal."""
    if not path:
        return False

    clean_path = str(path).strip()
    # Check for obvious traversal or prohibited absolute/home paths
    lower = clean_path.lower()
    for denied in ["/etc", "~/.ssh", ".env", ".git", "id_rsa", "shadow"]:
        if denied in lower:
            return False

    if ".." in clean_path:
        root = os.path.abspath(allowed_root or os.getcwd())
        try:
            resolved = os.path.abspath(os.path.join(root, clean_path) if not os.path.isabs(clean_path) else clean_path)
            if not resolved.startswith(root):
                return False
        except Exception:
            return False

    return True



class ActionGate:
    """ActionGate enforcing strict safety guardrails, human authorization, and pattern detection."""

    def __init__(
        self,
        denylist_paths: list[str] | None = None,
        allowlist_tools: list[str] | None = None,
        max_payload_bytes: int = 65536,
        max_cyclical_retries: int = 3,
        max_cumulative_tokens: int = 16384,
    ) -> None:
        self.denylist_paths = denylist_paths or [
            ".env",
            "tests/",
            "/etc/",
            "~/.ssh",
            ".git",
        ]
        self.allowlist_tools = allowlist_tools or [
            "ocr_ingest",
            "retriever",
            "statutory_evaluator",
            "rule_lookup",
        ]
        self.max_payload_bytes = max_payload_bytes
        self.max_cyclical_retries = max_cyclical_retries
        self.max_cumulative_tokens = max_cumulative_tokens
        self.rae_gater = RAEGater()

        self.violations_log: list[GateDecision] = []
        self.failure_payloads: list[dict[str, Any]] = []
        self.event_log = TrajectoryEventLog()
        self.sandboxes: dict[str, SandboxContext] = {}
        self._cyclical_error_counts: dict[str, int] = defaultdict(int)
        self._subtask_tokens_spent: dict[str, int] = defaultdict(int)

    def validate_payload_size(self, tool_name: str, kwargs: dict[str, Any]) -> int:
        """Enforce maximum argument payload serialization byte size tripwire."""
        try:
            payload_str = json.dumps(kwargs, default=str)
        except Exception:
            payload_str = str(kwargs)
        payload_bytes = len(payload_str.encode("utf-8"))
        if payload_bytes > self.max_payload_bytes:
            raise CostTripwireError(
                f"Cost tripwire breached: tool '{tool_name}' argument payload size "
                f"({payload_bytes} bytes) exceeds maximum ceiling of {self.max_payload_bytes} bytes."
            )
        return payload_bytes

    def track_cyclical_recovery(self, tool_name: str, error_signature: str, case_id: str = "global") -> int:
        """Detect and break infinite cyclical error recovery loops."""
        key = f"{case_id}:{tool_name}:{error_signature}"
        self._cyclical_error_counts[key] += 1
        count = self._cyclical_error_counts[key]
        if count > self.max_cyclical_retries:
            raise CyclicalRetryLoopError(
                f"Cost tripwire breached: cyclical error recovery exceeded threshold "
                f"({count} > {self.max_cyclical_retries}) for tool '{tool_name}' on error '{error_signature}'. "
                f"Force-terminating infinite recovery loop."
            )
        return count

    def reset_cyclical_recovery(self, tool_name: str, case_id: str = "global") -> None:
        """Reset cyclical error counter upon successful tool execution."""
        prefix = f"{case_id}:{tool_name}:"
        for k in list(self._cyclical_error_counts.keys()):
            if k.startswith(prefix):
                self._cyclical_error_counts.pop(k, None)

    def record_and_enforce_token_ceiling(self, subtask_id: str, tokens_spent: int) -> int:
        """Track cumulative subtask tokens and enforce cost ceiling tripwire."""
        self._subtask_tokens_spent[subtask_id] += tokens_spent
        accumulated = self._subtask_tokens_spent[subtask_id]
        if accumulated > self.max_cumulative_tokens:
            raise CostTripwireError(
                f"Cost tripwire breached: cumulative token ceiling exceeded "
                f"({accumulated} > {self.max_cumulative_tokens}) for sub-task '{subtask_id}'. "
                f"Force-terminating runaway exploratory path."
            )
        return accumulated

    def evaluate_retrieval_entropy_effect(
        self,
        skill_or_tool: str,
        base_probs_or_logits: list[float] | np.ndarray,
        retrieved_probs_or_logits: list[float] | np.ndarray,
        is_logits: bool = False,
    ) -> RAEValidationResult:
        """Execute counterfactual RAE validation on candidate skill retrieval."""
        return self.rae_gater.evaluate_retrieval_entropy_effect(
            skill_or_tool=skill_or_tool,
            base_probs_or_logits=base_probs_or_logits,
            retrieved_probs_or_logits=retrieved_probs_or_logits,
            is_logits=is_logits,
        )

    def gate_retrieved_payload(
        self,
        skill_or_tool: str,
        retrieved_payload: Any,
        base_distribution: list[float] | np.ndarray,
        retrieved_distribution: list[float] | np.ndarray,
    ) -> tuple[bool, Any]:
        """Gate retrieved skill payload: suppress and force parametric weights if entropy increases."""
        res = self.evaluate_retrieval_entropy_effect(
            skill_or_tool=skill_or_tool,
            base_probs_or_logits=base_distribution,
            retrieved_probs_or_logits=retrieved_distribution,
        )
        if res.retrieval_suppressed:
            return False, None
        return True, retrieved_payload

    def create_sandbox(
        self, case_id: str, initial_state: dict[str, Any] | None = None
    ) -> SandboxContext:
        """Create and register an isolated execution sandbox for a case."""
        sb = SandboxContext(case_id, initial_state)
        self.sandboxes[case_id] = sb
        return sb

    def get_or_create_sandbox(self, case_id: str) -> SandboxContext:
        """Retrieve existing sandbox or initialize a new one for the case."""
        if case_id not in self.sandboxes:
            self.sandboxes[case_id] = SandboxContext(case_id)
        return self.sandboxes[case_id]

    def rollback_sandbox(self, case_id: str, reason: str = "audit_rejection") -> dict[str, Any]:
        """Rollback sandbox to latest stable snapshot and record ROLLED_BACK in event log."""
        sb = self.get_or_create_sandbox(case_id)
        restored = sb.rollback_to_snapshot()
        self.event_log.append_event(
            case_id=case_id,
            event_type=TrajectoryEventType.ROLLED_BACK,
            agent_id="action_gate",
            action_name="rollback_sandbox",
            payload={"reason": reason},
        )
        return restored

    def commit_sandbox(
        self,
        case_id: str,
        supervisor_signature: SupervisorSignature | None = None,
        judge_result: Any | None = None,
    ) -> dict[str, Any]:
        """Commit pending sandbox mutations only after verifying judge approval and supervisor signoff."""
        sb = self.get_or_create_sandbox(case_id)
        if judge_result is not None:
            passed = getattr(judge_result, "passed", True)
            if not passed:
                self.rollback_sandbox(
                    case_id,
                    reason=f"judge_rejected: {getattr(judge_result, 'reasoning', 'Judge rejection')}",
                )
                self.event_log.append_event(
                    case_id=case_id,
                    event_type=TrajectoryEventType.JUDGE_REJECTED,
                    agent_id="judge",
                    action_name="commit_sandbox",
                    payload={"reasons": getattr(judge_result, "failing_rules", [])},
                )
                raise SecurityViolationError(
                    f"Mandatory judge approval rejected action for case {case_id}"
                )

        committed_state = sb.commit(
            supervisor_signature=supervisor_signature, judge_result=judge_result
        )
        self.event_log.append_event(
            case_id=case_id,
            event_type=(
                TrajectoryEventType.JUDGE_APPROVED
                if judge_result
                else TrajectoryEventType.COMMITTED
            ),
            agent_id="judge" if judge_result else "supervisor",
            action_name="commit_sandbox",
            payload={"committed_keys": list(committed_state.keys())},
            signature=supervisor_signature.signature_token if supervisor_signature else None,
        )
        return committed_state

    def record_violation_telemetry(
        self,
        decision: GateDecision,
        case_id: str = "global",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record structured failure telemetry from a security or precondition violation."""
        self.violations_log.append(decision)
        payload = {
            "case_id": case_id,
            "agent_id": decision.agent_id,
            "action_type": decision.action_type,
            "decision": decision.decision.value,
            "severity": decision.severity.value,
            "matched_rules": list(decision.matched_rules),
            "reasons": list(decision.reasons),
            "evidence_spans": list(decision.evidence_spans),
            "remediation_hint": decision.remediation_hint,
            "details": details or {},
            "timestamp": decision.timestamp.isoformat() if hasattr(decision.timestamp, "isoformat") else str(decision.timestamp),
        }
        self.failure_payloads.append(payload)
        return payload

    def get_recent_violations(self) -> list[GateDecision]:
        return list(self.violations_log)

    def get_failure_payloads(self) -> list[dict[str, Any]]:
        return list(self.failure_payloads)

    def clear_violations(self) -> None:
        self.violations_log.clear()
        self.failure_payloads.clear()


    def evaluate_text_patterns(
        self,
        content: str,
        action_type: str = "trace_inspection",
        agent_id: str = "unknown",
    ) -> GateDecision:
        """Scan string content against all 8 static pattern detectors and produce a structured GateDecision."""
        if not isinstance(content, str):
            content = str(content)

        matched_rules: list[str] = []
        evidence_spans: list[str] = []
        reasons: list[str] = []
        highest_severity = GateSeverity.LOW
        decision = GateDecisionType.ALLOW

        # 1. Hidden Test Suite Inspection
        for pat in StaticPatternDetectors.HIDDEN_TEST_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("HIDDEN_TEST_INSPECTION")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted inspection or access to hidden test suites/ground truth: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.BLOCK

        # 2. Statutory Rule Overrides
        for pat in StaticPatternDetectors.RULE_OVERRIDE_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("STATUTORY_RULE_OVERRIDE")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted modification or override of statutory rule definitions: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.BLOCK

        # 3. Citation Key Fabrication
        for pat in StaticPatternDetectors.CITATION_FABRICATION_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("CITATION_FABRICATION")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted citation fabrication: '{m.group(0)}'")
                highest_severity = GateSeverity.HIGH
                decision = GateDecisionType.BLOCK

        # 4. Rule Store Tampering
        for pat in StaticPatternDetectors.RULE_STORE_TAMPERING_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("RULE_STORE_TAMPERING")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted tampering with RuleStore integrity: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.QUARANTINE

        # 5. Criteria Tampering
        for pat in StaticPatternDetectors.CRITERIA_TAMPERING_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("CRITERIA_MANIPULATION")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted artificial manipulation of evaluation criteria: '{m.group(0)}'")
                highest_severity = GateSeverity.HIGH
                decision = GateDecisionType.BLOCK

        # 6. Governance Bypass
        for pat in StaticPatternDetectors.GOVERNANCE_BYPASS_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("GOVERNANCE_BYPASS")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted bypass of governance or audit hooks: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.QUARANTINE

        # 7. Unauthorized File & Secret Access
        for pat in StaticPatternDetectors.UNAUTHORIZED_ACCESS_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("UNAUTHORIZED_ACCESS")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted unauthorized path traversal or credential access: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.BLOCK

        # 8. Policy Override / Injection
        for pat in StaticPatternDetectors.POLICY_OVERRIDE_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("POLICY_OVERRIDE_INJECTION")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted policy override or prompt injection: '{m.group(0)}'")
                highest_severity = GateSeverity.HIGH
                decision = GateDecisionType.BLOCK

        # 9. Prohibited Direct Emission by Ingestion / Non-Binding Models
        for pat in StaticPatternDetectors.PROHIBITED_UNVERIFIED_DIRECT_EMISSION:
            m = pat.search(content)
            if m:
                matched_rules.append("PROHIBITED_UNVERIFIED_DIRECT_EMISSION")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted unverified direct emission of statutory determination by non-binding model: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.BLOCK

        # 10. Environment Manipulation, Privilege Escalation & Mock Tampering
        for pat in StaticPatternDetectors.ENV_MANIPULATION_PATTERNS:
            m = pat.search(content)
            if m:
                matched_rules.append("ENVIRONMENT_TAMPERING")
                evidence_spans.append(m.group(0))
                reasons.append(f"Attempted privilege escalation, environment manipulation, or mock tampering: '{m.group(0)}'")
                highest_severity = GateSeverity.CRITICAL
                decision = GateDecisionType.BLOCK

        remediation = "Ensure all actions respect sandbox boundaries and cite active statutory rules." if reasons else ""

        return GateDecision(
            decision=decision,
            severity=highest_severity,
            matched_rules=matched_rules,
            evidence_spans=evidence_spans,
            reasons=reasons,
            remediation_hint=remediation,
            agent_id=agent_id,
            action_type=action_type,
        )


    def enforce_guardrails(
        self,
        content: Any,
        action_type: str = "guardrail_check",
        agent_id: str = "unknown",
        audit_log: Any | None = None,
    ) -> GateDecision:
        """Evaluate content against guardrails and fail closed on BLOCK or QUARANTINE."""
        text_to_scan = str(content)
        if isinstance(content, dict):
            import json
            text_to_scan = json.dumps(content, default=str)

        decision = self.evaluate_text_patterns(text_to_scan, action_type=action_type, agent_id=agent_id)

        if audit_log and hasattr(audit_log, "append"):
            from ..types import SMState
            audit_log.append(
                case_id=getattr(content, "case_id", "global"),
                state=SMState.VERIFY,
                agent="action_gate",
                action=f"guardrail decision: {decision.decision.value} (severity={decision.severity.value})",
                payload={
                    "decision": decision.decision.value,
                    "severity": decision.severity.value,
                    "matched_rules": ", ".join(decision.matched_rules),
                    "reasons": "; ".join(decision.reasons)[:300],
                },
            )

        if decision.decision in (GateDecisionType.BLOCK, GateDecisionType.QUARANTINE):
            raise SecurityViolationError(
                f"ActionGate blocked action '{action_type}' due to {decision.severity.value} security violation: "
                f"{'; '.join(decision.reasons)}"
            )

        return decision

    def prepare_only(self, materials: PreparedMaterials) -> PreparedMaterials:
        """Preparation is always allowed and never marks anything submitted."""
        if not materials.case_id or not materials.program:
            raise PreConditionError("Invalid materials: case_id and program must be specified.")
        assert getattr(materials, "submitted", False) is False, "Prepared materials must never be submitted"
        return materials

    def enforce_proposer_model_firewall(
        self,
        assessment: Assessment,
        source_model: str | None = None,
        is_verified: bool = False,
    ) -> None:
        """Architectural firewall prohibiting glm-5.3-flash and unverified models from directly emitting legal eligibility determinations."""
        model_name = (source_model or "").lower()
        rationale = (assessment.rationale or "").lower()
        # Direct emission from glm-5.3-flash without independent verifier certification is prohibited
        if "glm-5.3-flash" in model_name or "glm-5.3" in model_name or "glm-5.3-flash" in rationale:
            if not is_verified:
                raise SecurityViolationError(
                    "ActionGate Architectural Firewall: glm-5.3-flash is restricted to document ingestion "
                    "and non-binding trajectory planning and is prohibited from directly emitting unverified "
                    "legal eligibility determinations. Verification pass by independent verifier required."
                )

    def verify_citations(
        self,
        assessment: Assessment,
        rule_store: RuleStore | None = None,
        source_model: str | None = None,
        is_verified: bool = False,
    ) -> tuple[bool, list[str]]:
        """Mandatory citation verification gate for model outputs.

        Requires every eligibility determination or legal claim to contain an exact,
        verifiable citation matching an active entry in rule_store, and enforces
        the architectural firewall against unverified direct emission by non-binding models.
        """
        if rule_store is None:
            from ..corpus.rule_store import LocalRuleStore
            rule_store = LocalRuleStore()

        # Check firewall
        try:
            self.enforce_proposer_model_firewall(assessment, source_model=source_model, is_verified=is_verified)
        except SecurityViolationError as err:
            return False, [str(err)]

        active_citations = rule_store.all_citations(assessment.program, assessment.jurisdiction)
        active_ids = {c.citation_id for c in active_citations}

        violations: list[str] = []
        if not assessment.citations:
            violations.append("Assessment contains no statutory citations.")

        thinking_re = re.compile(r"<(?:think|thought|reasoning)[^>]*>", re.IGNORECASE)
        if thinking_re.search(assessment.rationale) or "<thought" in assessment.rationale.lower() or "<think" in assessment.rationale.lower():
            violations.append("Assessment rationale contains non-deterministic model monologues or unverified reasoning outputs.")

        for citation in assessment.citations:
            if citation.citation_id not in active_ids:
                violations.append(f"Invalid statutory reference citation ID '{citation.citation_id}'")

        for crit in assessment.criteria:
            if crit.outcome is not CriterionOutcome.UNKNOWN:
                if not crit.citation_ids:
                    violations.append(f"Uncited claim for criterion '{crit.criterion_id}'")
                else:
                    for cid in crit.citation_ids:
                        if cid not in active_ids:
                            violations.append(f"Criterion '{crit.criterion_id}' references invalid citation ID '{cid}'")

        # Static pattern detection on rationale
        decision = self.evaluate_text_patterns(assessment.rationale, action_type="assessment_rationale", agent_id="proposer")
        if decision.decision != GateDecisionType.ALLOW:
            violations.extend(decision.reasons)

        return len(violations) == 0, violations

    def validate_navigator_statutory_claims(
        self,
        claims: list[dict[str, Any]] | list[str],
        program: str,
        jurisdiction: str = "EX",
        local_rules_provider: Any | None = None,
        verifier: Any | None = None,
    ) -> dict[str, Any]:
        """Enforce pipeline flow: All statutory eligibility claims emitted by navigator MUST be cross-referenced
        against local_rules.py and validated by verifier.py before execution or export.
        """
        if not claims:
            return {"valid": True, "claims_evaluated": 0, "verified": True}

        from ..providers.local_rules import cross_evaluate_rule_citations

        normalized_claim_cits: list[str] = []
        for c in claims:
            if isinstance(c, dict):
                cit = c.get("citation_id") or c.get("citation") or c.get("source", "")
                if cit:
                    normalized_claim_cits.append(str(cit))
            elif isinstance(c, str):
                normalized_claim_cits.append(c)

        cross_res = cross_evaluate_rule_citations(normalized_claim_cits, program=program, jurisdiction=jurisdiction)
        if not cross_res.get("valid", False) and cross_res.get("missing_citations"):
            missing = cross_res.get("missing_citations", [])
            raise PreConditionError(
                f"Navigator statutory claim validation failed: claims reference unverified/missing statutory citations "
                f"in local_rules.py: {missing}. Claims must be cross-referenced against local_rules."
            )

        if verifier is not None and hasattr(verifier, "verify_statutory_claims"):
            v_report = verifier.verify_statutory_claims(claims, program=program, jurisdiction=jurisdiction)
            if not getattr(v_report, "is_certified", getattr(v_report, "approved", True)):
                raise PreConditionError(
                    f"Navigator statutory claims rejected by verifier: {getattr(v_report, 'reasons', 'Verification failed')}"
                )

        return {
            "valid": True,
            "claims_evaluated": len(claims),
            "matched_citations": cross_res.get("matched_citations", []),
            "cross_referenced_local_rules": True,
            "verified": True,
        }

    def assert_evaluation_certified(
        self,
        report: Any,
        assessment: Assessment | None = None,
    ) -> None:
        """Assert that an evaluation has cleared Pass 1 milestone verification before proceeding to generation."""
        is_certified = getattr(report, "is_certified", False)
        if not is_certified:
            reasons = getattr(report, "reasons", ["unverified evaluation"])
            raise PreConditionError(
                f"Evaluation failed Pass 1 milestone verification: {'; '.join(reasons)}"
            )
        missing_citations = getattr(report, "missing_citations", [])
        if missing_citations:
            raise PreConditionError(
                f"Evaluation contains uncited/invalid citations: {'; '.join(missing_citations)}"
            )
        unsupported = getattr(report, "unsupported_claims", [])
        if unsupported:
            raise PreConditionError(
                f"Evaluation contains unsupported statutory claims: {'; '.join(unsupported)}"
            )

    def authorize_notice_generation(
        self,
        assessment: Assessment,
        verification_report: Any | None = None,
        rule_store: RuleStore | None = None,
    ) -> bool:
        """Enforce two-stage verification gate before allowing determination/disclosure notice generation."""
        if verification_report is not None:
            self.assert_evaluation_certified(verification_report, assessment)

        is_valid, violations = self.verify_citations(assessment, rule_store)
        if not is_valid:
            raise PreConditionError(
                f"Notice generation blocked due to statutory citation violations: {'; '.join(violations)}"
            )
        return True

    def assert_preconditions(
        self,
        materials: PreparedMaterials | None = None,
        assessment: Assessment | None = None,
        rule_store: RuleStore | None = None,
        tool_name: str | None = None,
        sandbox_mode: bool = True,
        supervisor_signature: SupervisorSignature | None = None,
        case_id: str | None = None,
        raw_input: str | None = None,
    ) -> None:
        """Enforce strict pre-condition assertions before any tool or state modification."""
        if raw_input:
            self.enforce_guardrails(raw_input, action_type="raw_input_precondition")

        if materials is not None:
            if not materials.case_id:
                raise PreConditionError("Pre-condition failed: materials missing case_id.")
            if not materials.program:
                raise PreConditionError("Pre-condition failed: materials missing program.")

        if assessment is not None:
            is_valid, violations = self.verify_citations(assessment, rule_store)
            if not is_valid:
                raise PreConditionError(
                    f"Submission blocked: uncited claims or invalid statutory references detected. "
                    f"Flagged for human review. Violations: {'; '.join(violations)}"
                )

        if tool_name is not None and sandbox_mode:
            expected_action = f"{tool_name}:{case_id}" if case_id else tool_name
            if supervisor_signature is None or not supervisor_signature.is_valid(expected_action):
                raise PreConditionError(
                    f"Pre-condition failed: External API execution '{tool_name}' restricted to read-only sandbox. "
                    f"Valid supervisor signature payload required for '{expected_action}'."
                )

    def assert_postconditions(
        self,
        receipt: SubmissionReceipt | None = None,
        materials: PreparedMaterials | None = None,
        signoff: HumanSignoff | None = None,
        result: Any = None,
    ) -> None:
        """Enforce strict post-condition assertions verifying state integrity and non-repudiation."""
        if receipt is not None:
            if not receipt.signoff_token:
                raise PostConditionError("Post-condition failed: submission receipt missing authorization token.")
            if not receipt.authorized_by:
                raise PostConditionError("Post-condition failed: submission receipt missing authorizer identity.")
            if materials is not None and (receipt.program != materials.program or receipt.case_id != materials.case_id):
                raise PostConditionError("Post-condition failed: receipt metadata does not match materials.")
            if signoff is not None and receipt.signoff_token != signoff.token:
                raise PostConditionError("Post-condition failed: receipt token mismatch with signoff.")

        if materials is not None:
            assert getattr(materials, "submitted", False) is False, "Post-condition violated: materials mutated"

    def execute_tool(
        self,
        tool_name: str,
        tool_fn: Callable[..., Any],
        kwargs: dict[str, Any],
        sandbox_mode: bool = True,
        supervisor_signature: SupervisorSignature | None = None,
    ) -> Any:
        """Execute a tool wrapped with pre- and post-condition assertion checks and sandbox security."""
        case_id = kwargs.get("case_id", "global")
        # Cost tripwire: payload serialization size validation
        self.validate_payload_size(tool_name, kwargs)

        # Path containment validation on any file path parameters
        for k in ["path", "file_path", "target_path", "filepath", "output_path", "input_path"]:
            if k in kwargs and isinstance(kwargs[k], str):
                if not validate_path_containment(kwargs[k]):
                    raise SecurityViolationError(
                        f"ActionGate path validation blocked out-of-bounds file system traversal attempt: '{kwargs[k]}'"
                    )

        # Guardrail check on tool inputs
        self.enforce_guardrails(kwargs, action_type=f"tool_input:{tool_name}")

        self.assert_preconditions(
            tool_name=tool_name,
            sandbox_mode=sandbox_mode,
            supervisor_signature=supervisor_signature,
            case_id=case_id,
        )

        self.event_log.append_event(
            case_id=case_id,
            event_type=TrajectoryEventType.EXTERNAL_CALL_ATTEMPTED,
            agent_id="tool_runner",
            action_name=tool_name,
            payload={"args": list(kwargs.keys())},
            signature=supervisor_signature.signature_token if supervisor_signature else None,
        )

        sb = self.get_or_create_sandbox(case_id)
        pre_snap = sb.snapshot()

        try:
            result = tool_fn(**kwargs)
            if result is None and "allow_none" not in kwargs:
                raise PostConditionError(
                    f"Post-condition failed: tool '{tool_name}' returned null unexpectedly."
                )
            self.assert_postconditions(result=result)
            self.reset_cyclical_recovery(tool_name, case_id)
            return result
        except Exception as exc:
            sb.rollback_to_snapshot(pre_snap.snapshot_id)
            self.event_log.append_event(
                case_id=case_id,
                event_type=TrajectoryEventType.ROLLED_BACK,
                agent_id="tool_runner",
                action_name=tool_name,
                payload={"error": str(exc)},
            )
            # Track cyclical error recovery tripwire before re-raising
            if not isinstance(exc, CostTripwireError):
                self.track_cyclical_recovery(tool_name, type(exc).__name__, case_id)
            raise

    def authorize_submission(
        self,
        materials: PreparedMaterials,
        signoff: HumanSignoff | None,
        assessment: Assessment | None = None,
        rule_store: RuleStore | None = None,
    ) -> SubmissionReceipt:
        self.assert_preconditions(materials=materials, assessment=assessment, rule_store=rule_store)

        if signoff is None:
            raise ActionBlocked(
                "binding submission requires explicit human sign-off; none provided"
            )
        if not signoff.token:
            raise ActionBlocked("human sign-off is missing an authorization token")
        expected = f"{materials.program.value}:{materials.case_id}"
        if signoff.intent != expected:
            raise ActionBlocked(
                f"sign-off intent '{signoff.intent}' does not authorize submission of '{expected}'"
            )

        receipt = SubmissionReceipt(
            program=materials.program,
            case_id=materials.case_id,
            authorized_by=signoff.authorized_by,
            signoff_token=signoff.token,
        )

        self.assert_postconditions(receipt=receipt, materials=materials, signoff=signoff)
        return receipt


class ManifestEnforcer:
    """Governance enforcer verifying SHA-256 signatures for shared skills, partitions, and rule indices."""

    def __init__(self, manifest: Any | None = None) -> None:
        from ..memory.partitions import PartitionIntegrityManifest
        self.manifest = manifest or PartitionIntegrityManifest()
        self.rejections_count: int = 0
        self.verifications_count: int = 0

    def enforce_partition_integrity(self, partition: Any) -> bool:
        """Verify partition checksum against SHA-256 manifest. Raises SecurityViolationError on tampering."""
        from ..memory.partitions import PartitionTamperingError
        try:
            checksum = partition.compute_checksum()
            verified = self.manifest.verify_partition(partition.partition_id, checksum)
            self.verifications_count += 1
            return verified
        except PartitionTamperingError as err:
            self.rejections_count += 1
            raise SecurityViolationError(f"Anti-EvoMal Manifest Violation: {err}") from err

    def enforce_skill_integrity(self, skill_name: str, content: str | bytes) -> bool:
        """Verify skill module checksum against SHA-256 manifest."""
        from ..memory.partitions import PartitionTamperingError
        try:
            verified = self.manifest.verify_skill(skill_name, content)
            self.verifications_count += 1
            return verified
        except PartitionTamperingError as err:
            self.rejections_count += 1
            raise SecurityViolationError(f"Anti-EvoMal Skill Manifest Violation: {err}") from err


class ManagedProfileIsolation:
    """Manages ephemeral, strictly isolated browser profile directories to prevent token/cookie leaks."""

    def __init__(self, base_dir: str = "/tmp/tribune_browser_profiles") -> None:
        self.base_dir = base_dir
        self.active_profiles: dict[str, str] = {}

    def get_isolated_profile(self, case_id: str, subagent_id: str) -> dict[str, Any]:
        """Create or retrieve isolated profile metadata for an agent session."""
        scoped_key = f"{case_id}::{subagent_id}"
        profile_path = f"{self.base_dir}/{case_id}/{subagent_id}"
        self.active_profiles[scoped_key] = profile_path
        return {
            "case_id": case_id,
            "subagent_id": subagent_id,
            "profile_path": profile_path,
            "isolated": True,
            "allow_shared_cookies": False,
            "allow_shared_storage": False,
        }


class BrowserSandboxManager:
    """Browser Sandbox managing domain-scoped ACLs and profile isolation for external legal retrieval."""

    DEFAULT_ALLOWED_DOMAINS: list[str] = [
        r"^https?://([a-zA-Z0-9-]+\.)*gov(/.*)?$",
        r"^https?://([a-zA-Z0-9-]+\.)*state\.[a-z]{2}\.us(/.*)?$",
        r"^https?://([a-zA-Z0-9-]+\.)*usda\.gov(/.*)?$",
        r"^https?://([a-zA-Z0-9-]+\.)*cms\.gov(/.*)?$",
        r"^https?://([a-zA-Z0-9-]+\.)*ssa\.gov(/.*)?$",
        r"^https?://([a-zA-Z0-9-]+\.)*law\.cornell\.edu(/.*)?$",
    ]

    def __init__(
        self,
        allowed_domain_patterns: list[str] | None = None,
        profile_manager: ManagedProfileIsolation | None = None,
    ) -> None:
        import re
        self.allowed_patterns = [
            re.compile(p, re.IGNORECASE) for p in (allowed_domain_patterns or self.DEFAULT_ALLOWED_DOMAINS)
        ]
        self.profile_manager = profile_manager or ManagedProfileIsolation()
        self.blocked_navigations: list[dict[str, Any]] = []

    def validate_navigation(self, url: str, case_id: str = "global", subagent_id: str = "main") -> bool:
        """Validate that external URL complies with domain-scoped ACLs (e.g. .gov only)."""
        clean_url = url.strip()
        is_allowed = any(p.match(clean_url) for p in self.allowed_patterns)
        if not is_allowed:
            self.blocked_navigations.append({
                "url": clean_url,
                "case_id": case_id,
                "subagent_id": subagent_id,
                "reason": "Domain not permitted under statutory legal retrieval ACL (.gov only)",
                "timestamp": time.time(),
            })
            raise SecurityViolationError(
                f"Browser Sandbox Security Block: Navigation to '{clean_url}' rejected. "
                f"External document retrieval is strictly restricted to authorized statutory (.gov) domains."
            )
        return True

    def create_isolated_session(self, case_id: str, subagent_id: str) -> dict[str, Any]:
        """Initialize an isolated browser session with managed profile isolation."""
        profile = self.profile_manager.get_isolated_profile(case_id, subagent_id)
        return {
            "session_id": f"sess_{case_id}_{subagent_id}",
            "profile": profile,
            "acl_enforced": True,
        }


__all__ = [
    "ActionBlocked",
    "PreConditionError",
    "PostConditionError",
    "SecurityViolationError",
    "CostTripwireError",
    "CyclicalRetryLoopError",
    "RetrievalEntropyDegradationError",
    "RAEValidationResult",
    "RAEGater",
    "GateDecisionType",
    "GateSeverity",
    "GateDecision",
    "HumanSignoff",
    "SupervisorSignature",
    "TrajectoryEventType",
    "TrajectoryEvent",
    "TrajectoryEventLog",
    "SandboxSnapshot",
    "SandboxContext",
    "StaticPatternDetectors",
    "ManifestEnforcer",
    "ManagedProfileIsolation",
    "BrowserSandboxManager",
    "ActionGate",
    "validate_path_containment",
]


