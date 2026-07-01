"""Abstractions shared by every program rule set.

A :class:`Rule` couples one eligibility criterion to (a) the citation that governs
it and (b) a *pure predicate* that evaluates it against an :class:`EvidenceView`.
The same predicate is the single source of truth used by:

* the eligibility proposer (over the evidence TRIBUNE actually has),
* the verifier (re-derivation from the cited rules), and
* the synthetic ground-truth labeler (over the complete situation).

That is how the corpus and the synthetic generator stay consistent without
duplicating eligibility logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ...types import Citation, CriterionOutcome, EvidenceType, EvidenceView, ProgramId
from .jurisdictions import JurisdictionProfile

Predicate = Callable[[EvidenceView, JurisdictionProfile], CriterionOutcome]
# Returns a normalized "distance to the decision boundary" in [0, 1], or None when
# the criterion is not a numeric near-miss kind of test. Used only for calibration.
MarginFn = Callable[[EvidenceView, JurisdictionProfile], float | None]
# Returns structural ambiguity signals (e.g. the Medicaid coverage gap) detectable
# from the evidence alone — no ground truth. Used by both the live abstention path
# and the synthetic ground-truth labeler so they agree on what "ambiguous" means.
AmbiguityFn = Callable[[EvidenceView, JurisdictionProfile], list]


@dataclass(frozen=True)
class Rule:
    criterion_id: str
    description: str
    required: bool
    source: str
    title: str
    text: str
    predicate: Predicate
    evidence_types: tuple[EvidenceType, ...] = ()
    locator: str = ""
    effective_date: str | None = None
    margin_fn: MarginFn | None = None

    def citation(self, program: ProgramId, jurisdiction: str) -> Citation:
        return Citation(
            citation_id=f"{program.value}:{jurisdiction}:{self.criterion_id}",
            program=program,
            jurisdiction=jurisdiction,
            source=self.source,
            title=self.title,
            text=self.text,
            locator=self.locator,
            effective_date=self.effective_date,
        )

    def evidence_ids(self, view: EvidenceView) -> list[str]:
        return view.ids(*self.evidence_types)


@dataclass(frozen=True)
class RuleSet:
    program: ProgramId
    rules: list[Rule] = field(default_factory=list)
    ambiguity_fn: AmbiguityFn | None = None

    def ambiguity_signals(self, view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
        if self.ambiguity_fn is None:
            return []
        return list(self.ambiguity_fn(view, profile))

    @property
    def required_ids(self) -> list[str]:
        return [r.criterion_id for r in self.rules if r.required]

    def get(self, criterion_id: str) -> Rule | None:
        for r in self.rules:
            if r.criterion_id == criterion_id:
                return r
        return None
