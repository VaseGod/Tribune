"""Synthetic labeler for SNAP. Adding a program means adding a module like this."""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...types import ApplicantSituation, ProgramGroundTruth, ProgramId
from ...types import EvidenceType as ET
from .base import generic_ground_truth

PROGRAM = ProgramId.SNAP
RELEVANT_EVIDENCE = [
    ET.RESIDENT,
    ET.RESIDENCY_STATE,
    ET.CITIZENSHIP_STATUS,
    ET.MONTHLY_INCOME,
    ET.HOUSEHOLD_SIZE,
    ET.LIQUID_ASSETS,
]


def ground_truth(situation: ApplicantSituation, profile: JurisdictionProfile) -> ProgramGroundTruth:
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale="SNAP turns on residency, citizenship/immigration status, and gross "
        "monthly income vs. 130% of the poverty guideline (assets often waived under BBCE).",
    )
