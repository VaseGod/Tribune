"""Registry of synthetic ground-truth labelers, parallel to corpus/programs/."""

from __future__ import annotations

from ...types import ProgramId
from . import appeals, housing, medicaid, snap, unemployment

_LABELERS = {
    snap.PROGRAM: snap,
    unemployment.PROGRAM: unemployment,
    medicaid.PROGRAM: medicaid,
    housing.PROGRAM: housing,
    appeals.PROGRAM: appeals,
}


def ground_truth(program: ProgramId, situation, profile):
    return _LABELERS[program].ground_truth(situation, profile)


def relevant_evidence(program: ProgramId):
    return list(_LABELERS[program].RELEVANT_EVIDENCE)
