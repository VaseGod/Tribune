"""Preparer (proposer of materials).

Assembles application/appeal materials and a document checklist for an assessment
that cleared verification and was confident enough to assert. Everything it
produces is explicitly *unsubmitted*: the action-gate guarantees nothing binding
is filed without an explicit human sign-off.
"""

from __future__ import annotations

from ..corpus.programs.jurisdictions import get_profile
from ..governance.action_gate import ActionGate
from ..types import (
    Assessment,
    Evidence,
    EvidenceType,
    EvidenceView,
    PreparedMaterials,
    ProgramId,
)

_CHECKLISTS: dict[ProgramId, list[str]] = {
    ProgramId.SNAP: [
        "Photo ID for the applicant",
        "Proof of residency (lease, utility bill, or mail)",
        "Proof of gross income for the last 30 days (pay stubs, benefit letters)",
        "Social Security numbers for household members (or proof of application)",
        "Proof of immigration status, if a non-citizen",
        "Recent bank statements (only if your state applies an asset test)",
    ],
    ProgramId.UNEMPLOYMENT: [
        "Photo ID and Social Security number",
        "Most recent employer name, address, and dates of employment",
        "Reason for separation and any separation/termination letter",
        "Wage records or W-2s covering the base period",
        "Bank account details for benefit payment",
    ],
    ProgramId.MEDICAID: [
        "Photo ID for each applicant",
        "Proof of household income (pay stubs, tax return)",
        "Proof of state residency",
        "Proof of citizenship or eligible immigration status",
        "Pregnancy verification or disability documentation, if applicable",
    ],
    ProgramId.HOUSING: [
        "Photo ID and Social Security numbers for all household members",
        "Proof of annual household income from all sources",
        "Proof of citizenship or eligible immigration status",
        "Current address and contact information",
        "Documentation of any local preference (residency, veteran status, etc.)",
    ],
    ProgramId.APPEALS: [
        "A copy of the denial / adverse-action notice",
        "The date you received the notice",
        "A short written statement of why you believe the decision was wrong",
        "Any documents supporting your position (pay stubs, letters, records)",
        "Request for benefits to continue during the appeal, if still within the window",
    ],
}


class ProgrammaticPreparerTools:
    """Typed Python stubs executed directly in-code by preparer loops."""

    @staticmethod
    def assemble_document_checklist(program_name: str) -> list[str]:
        """Return the statutory document checklist for a program."""
        prog_key = None
        for p in ProgramId:
            if p.value == program_name:
                prog_key = p
                break
        return list(_CHECKLISTS.get(prog_key, [])) if prog_key else []

    @staticmethod
    def calculate_appeal_deadline(days_since_denial: int, window_days: int) -> dict:
        """Compute remaining days for fair hearing appeal window."""
        remaining = max(0, window_days - days_since_denial)
        return {
            "filing_window_days": window_days,
            "estimated_days_remaining": remaining,
            "is_expired": days_since_denial > window_days,
        }

    @classmethod
    def get_tool_signatures(cls) -> str:
        """Expose typed Python signatures for model prompt generation."""
        return (
            "class ProgrammaticPreparerTools:\n"
            "    @staticmethod\n"
            "    def assemble_document_checklist(program_name: str) -> list[str]: ...\n"
            "    @staticmethod\n"
            "    def calculate_appeal_deadline(days_since_denial: int, window_days: int) -> dict: ...\n"
        )


class Preparer:
    def __init__(self, action_gate: ActionGate) -> None:
        self.action_gate = action_gate
        self.tools = ProgrammaticPreparerTools()

    def generate_prompt(self, program: ProgramId, jurisdiction: str) -> str:
        """Generate prompt incorporating programmatic Python tool signatures."""
        return (
            f"You are the preparer for {program.value} in {jurisdiction}.\n"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticPreparerTools.get_tool_signatures()}\n"
            "Use these tools directly to build document checklists and appeal packets."
        )

    def prepare(
        self, assessment: Assessment, evidence: list[Evidence]
    ) -> PreparedMaterials:
        view = EvidenceView(evidence)
        fields: dict[str, str] = {}
        for ev in evidence:
            fields[ev.type.value] = str(ev.value)

        appeal_packet = None
        if assessment.program is ProgramId.APPEALS:
            profile = get_profile(assessment.jurisdiction)
            days = view.num(EvidenceType.DAYS_SINCE_DENIAL)
            grounds = view.text(EvidenceType.APPEAL_GROUNDS) or "(state your reason)"
            remaining = (
                int(profile.appeal_window_days - days) if days is not None else profile.appeal_window_days
            )
            appeal_packet = {
                "filing_window_days": str(profile.appeal_window_days),
                "estimated_days_remaining": str(max(0, remaining)),
                "stated_grounds": grounds,
                "instructions": (
                    "Submit a written fair-hearing/appeal request to the administering "
                    "agency before the deadline. You may ask for benefits to continue "
                    "during the appeal. Consider contacting legal aid."
                ),
            }

        materials = PreparedMaterials(
            program=assessment.program,
            jurisdiction=assessment.jurisdiction,
            case_id=assessment.case_id,
            application_fields=fields,
            document_checklist=list(_CHECKLISTS.get(assessment.program, [])),
            appeal_packet=appeal_packet,
        )
        # Preparation is always allowed; this never marks anything as submitted.
        return self.action_gate.prepare_only(materials)
