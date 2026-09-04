"""Synthetic labeler for unemployment insurance."""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...corpus.wiki import get_statutory_wiki
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
    wiki = get_statutory_wiki()
    rationale = wiki.get_program_rationale(PROGRAM, profile.code)
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale=rationale,
    )

