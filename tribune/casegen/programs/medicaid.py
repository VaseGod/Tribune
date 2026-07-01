"""Synthetic labeler for Medicaid (the abstention stress-test program)."""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...types import ApplicantSituation, ProgramGroundTruth, ProgramId
from ...types import EvidenceType as ET
from .base import generic_ground_truth

PROGRAM = ProgramId.MEDICAID
RELEVANT_EVIDENCE = [
    ET.RESIDENT,
    ET.RESIDENCY_STATE,
    ET.CITIZENSHIP_STATUS,
    ET.MONTHLY_INCOME,
    ET.HOUSEHOLD_SIZE,
    ET.AGE,
    ET.PREGNANT,
    ET.HAS_DEPENDENT_CHILD,
    ET.DISABLED,
]


def ground_truth(situation: ApplicantSituation, profile: JurisdictionProfile) -> ProgramGroundTruth:
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale="Medicaid eligibility depends on the applicable MAGI/non-MAGI category and "
        "the state's expansion status; the coverage gap and immigration interactions are "
        "genuinely ambiguous and must be escalated rather than guessed.",
    )
