"""Synthetic labeler for the appeals / fair-hearing workflow.

Ground truth here is whether an appeal is *available* (timely + grounds stated)."""

from __future__ import annotations

from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...types import ApplicantSituation, ProgramGroundTruth, ProgramId
from ...types import EvidenceType as ET
from .base import generic_ground_truth

PROGRAM = ProgramId.APPEALS
RELEVANT_EVIDENCE = [ET.DAYS_SINCE_DENIAL, ET.APPEAL_GROUNDS]


def ground_truth(situation: ApplicantSituation, profile: JurisdictionProfile) -> ProgramGroundTruth:
    return generic_ground_truth(
        PROGRAM,
        situation,
        profile,
        rationale="An appeal is available when the request is within the filing window and "
        "the person states a basis for contesting the action.",
    )
