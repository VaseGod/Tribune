"""Never-auto-submit action gate.

TRIBUNE may *prepare* applications, appeals, and checklists freely. It may never
*submit* anything binding on its own. Submission of prepared materials requires an
explicit :class:`HumanSignoff` whose intent matches the exact program and case;
without it, :meth:`ActionGate.authorize_submission` raises :class:`ActionBlocked`.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from ..corpus.rule_store import RuleStore
from ..types import Assessment, CriterionOutcome, PreparedMaterials, SubmissionReceipt


class ActionBlocked(PermissionError):
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


class ActionGate:
    def prepare_only(self, materials: PreparedMaterials) -> PreparedMaterials:
        """Preparation is always allowed and never marks anything submitted."""
        return materials  # submitted is structurally pinned False on the model

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

        import re
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

    def authorize_submission(
        self,
        materials: PreparedMaterials,
        signoff: HumanSignoff | None,
        assessment: Assessment | None = None,
        rule_store: RuleStore | None = None,
    ) -> SubmissionReceipt:
        if assessment is not None:
            is_valid, violations = self.verify_citations(assessment, rule_store)
            if not is_valid:
                raise ActionBlocked(
                    f"Submission blocked: uncited claims or invalid statutory references detected. Flagged for human review. Violations: {'; '.join(violations)}"
                )
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
        return SubmissionReceipt(
            program=materials.program,
            case_id=materials.case_id,
            authorized_by=signoff.authorized_by,
            signoff_token=signoff.token,
        )
