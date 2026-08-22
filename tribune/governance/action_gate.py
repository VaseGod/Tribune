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

import enum
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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


class ActionGate:
    """ActionGate enforcing strict safety guardrails, human authorization, and pattern detection."""

    def __init__(
        self,
        denylist_paths: list[str] | None = None,
        allowlist_tools: list[str] | None = None,
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

    def verify_citations(
        self, assessment: Assessment, rule_store: RuleStore | None = None
    ) -> tuple[bool, list[str]]:
        """Mandatory citation verification gate for local model outputs.

        Requires every eligibility determination or legal claim to contain an exact,
        verifiable citation matching an active entry in rule_store.
        """
        if rule_store is None:
            from ..corpus.rule_store import LocalRuleStore
            rule_store = LocalRuleStore()

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
        # Guardrail check on tool inputs
        self.enforce_guardrails(kwargs, action_type=f"tool_input:{tool_name}")

        self.assert_preconditions(
            tool_name=tool_name,
            sandbox_mode=sandbox_mode,
            supervisor_signature=supervisor_signature,
            case_id=case_id,
        )

        result = tool_fn(**kwargs)

        if result is None and "allow_none" not in kwargs:
            raise PostConditionError(f"Post-condition failed: tool '{tool_name}' returned null unexpectedly.")
        self.assert_postconditions(result=result)
        return result

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


__all__ = [
    "ActionBlocked",
    "PreConditionError",
    "PostConditionError",
    "SecurityViolationError",
    "GateDecisionType",
    "GateSeverity",
    "GateDecision",
    "HumanSignoff",
    "SupervisorSignature",
    "StaticPatternDetectors",
    "ActionGate",
]

