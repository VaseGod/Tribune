"""Registry of program rule sets.

Adding a new program or jurisdiction is the primary contribution path and requires
*no changes to core code*: drop a module exporting ``PROGRAM`` and ``RULESET`` and
register it here (and add a matching synthetic labeler under
``tribune/casegen/programs/``).
"""

from __future__ import annotations

from ...types import ProgramId
from . import appeals, housing, medicaid, snap, unemployment
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile, get_profile, known_jurisdictions

_REGISTRY: dict[ProgramId, RuleSet] = {
    snap.PROGRAM: snap.RULESET,
    unemployment.PROGRAM: unemployment.RULESET,
    medicaid.PROGRAM: medicaid.RULESET,
    housing.PROGRAM: housing.RULESET,
    appeals.PROGRAM: appeals.RULESET,
}


def get_ruleset(program: ProgramId) -> RuleSet:
    return _REGISTRY[program]


def all_programs() -> list[ProgramId]:
    return list(_REGISTRY.keys())


def benefit_programs() -> list[ProgramId]:
    """Programs that make an eligibility determination (excludes the appeals workflow)."""
    return [p for p in _REGISTRY if p is not ProgramId.APPEALS]


__all__ = [
    "Rule",
    "RuleSet",
    "JurisdictionProfile",
    "get_profile",
    "get_ruleset",
    "all_programs",
    "benefit_programs",
    "known_jurisdictions",
]
