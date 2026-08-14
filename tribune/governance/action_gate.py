"""Never-auto-submit action gate.

TRIBUNE may *prepare* applications, appeals, and checklists freely. It may never
*submit* anything binding on its own. Submission of prepared materials requires an
explicit :class:`HumanSignoff` whose intent matches the exact program and case;
without it, :meth:`ActionGate.authorize_submission` raises :class:`ActionBlocked`.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..corpus.rule_store import RuleStore
from ..types import Assessment, CriterionOutcome, PreparedMaterials, SubmissionReceipt


class ActionBlocked(PermissionError):
    pass


class PreConditionError(ActionBlocked):
    """Raised when pre-condition assertions fail before tool or action execution."""
    pass


class PostConditionError(ActionBlocked):
    """Raised when post-condition assertions fail after tool or action execution."""
    pass


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


class ActionGate:
    def prepare_only(self, materials: PreparedMaterials) -> PreparedMaterials:
        """Preparation is always allowed and never marks anything submitted."""
        # Pre-condition check
        if not materials.case_id or not materials.program:
            raise PreConditionError("Invalid materials: case_id and program must be specified.")
        # Post-condition check: structurally verify submission state remains False
        assert getattr(materials, "submitted", False) is False, "Prepared materials must never be submitted"
        return materials

    def verify_citations(
        self, assessment: Assessment, rule_store: RuleStore | None = None
    ) -> tuple[bool, list[str]]:
        """Mandatory citation verification gate for local model outputs.

        Requires every eligibility determination or legal claim to contain an exact,
        verifiable citation matching an active entry in rule_store.
        Returns (is_valid, list_of_violations).
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

        return len(violations) == 0, violations

    def assert_preconditions(
        self,
        materials: PreparedMaterials | None = None,
        assessment: Assessment | None = None,
        rule_store: RuleStore | None = None,
        tool_name: str | None = None,
        sandbox_mode: bool = True,
        supervisor_signature: SupervisorSignature | None = None,
        case_id: str | None = None,
    ) -> None:
        """Enforce strict pre-condition assertions before any tool or state modification."""
        if materials is not None:
            if not materials.case_id:
                raise PreConditionError("Pre-condition failed: materials missing case_id.")
            if not materials.program:
                raise PreConditionError("Pre-condition failed: materials missing program.")

        if assessment is not None:
            is_valid, violations = self.verify_citations(assessment, rule_store)
            if not is_valid:
                raise PreConditionError(
                    f"Submission blocked: uncited claims or invalid statutory references detected. Flagged for human review. Violations: {'; '.join(violations)}"
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
            # Ensure materials remained unsubmitted structurally
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
        # Pre-condition check
        self.assert_preconditions(
            tool_name=tool_name,
            sandbox_mode=sandbox_mode,
            supervisor_signature=supervisor_signature,
            case_id=case_id,
        )

        # Execution
        result = tool_fn(**kwargs)

        # Post-condition check
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
        # Pre-condition assertions
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

        # Post-condition assertions
        self.assert_postconditions(receipt=receipt, materials=materials, signoff=signoff)
        return receipt
