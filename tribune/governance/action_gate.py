"""Never-auto-submit action gate.

TRIBUNE may *prepare* applications, appeals, and checklists freely. It may never
*submit* anything binding on its own. Submission of prepared materials requires an
explicit :class:`HumanSignoff` whose intent matches the exact program and case;
without it, :meth:`ActionGate.authorize_submission` raises :class:`ActionBlocked`.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from ..types import PreparedMaterials, SubmissionReceipt


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

    def authorize_submission(
        self, materials: PreparedMaterials, signoff: HumanSignoff | None
    ) -> SubmissionReceipt:
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
