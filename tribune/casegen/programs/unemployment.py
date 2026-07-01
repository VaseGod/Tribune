"""Synthetic labeler for unemployment insurance."""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...types import ApplicantSituation, ProgramGroundTruth, ProgramId
from ...types import EvidenceType as ET
from .base import generic_ground_truth

PROGRAM = ProgramId.UNEMPLOYMENT
RELEVANT_EVIDENCE = [
    ET.BASE_PERIOD_EARNINGS,
    ET.WEEKS_WORKED,
    ET.SEPARATION_REASON,
    ET.ABLE_AND_AVAILABLE,
    ET.EMPLOYMENT_STATUS,
]


def ground_truth(situation: ApplicantSituation, profile: JurisdictionProfile) -> ProgramGroundTruth:
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale="Unemployment turns on monetary eligibility (base-period earnings/weeks), "
        "a non-disqualifying separation, and being able and available; a quit 'with good "
        "cause' is fact-intensive and ambiguous.",
    )
