"""Synthetic labeler for housing assistance (Housing Choice Voucher).

Note: ground truth here is *eligibility*. Access is separately gated by waitlist
status, which the ambiguity detector flags so that unknown/closed waitlists are
treated as escalate-to-human cases — eligibility is not access.
"""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...types import ApplicantSituation, ProgramGroundTruth, ProgramId
from ...types import EvidenceType as ET
from .base import generic_ground_truth

PROGRAM = ProgramId.HOUSING
RELEVANT_EVIDENCE = [
    ET.ANNUAL_INCOME,
    ET.HOUSEHOLD_SIZE,
    ET.CITIZENSHIP_STATUS,
    ET.RESIDENT,
    ET.RESIDENCY_STATE,
    ET.WAITLIST_STATUS,
]


def ground_truth(situation: ApplicantSituation, profile: JurisdictionProfile) -> ProgramGroundTruth:
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale="HCV eligibility is income (vs. HUD AMI limits), citizenship/immigration "
        "status, and PHA jurisdiction. Even when eligible, access is gated by waitlists.",
    )
