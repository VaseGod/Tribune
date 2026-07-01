"""Shared machinery for synthetic ground-truth labeling.

Ground truth is computed by running the *corpus* rule predicates over the
*complete* situation (every fact known), so the synthetic labels never drift from
the live eligibility logic. The per-program modules in this package add the
program-specific list of relevant evidence and a human-readable rationale; the
heavy lifting (build full evidence, evaluate, detect ambiguity) lives here.
"""

from __future__ import annotations

from ...corpus import programs as program_registry
from ...corpus.programs.jurisdictions import JurisdictionProfile
from ...corpus.provenance import make_provenance
from ...types import (
    ApplicantSituation,
    CriterionOutcome,
    Evidence,
    EvidenceType,
    GroundTruthLabel,
    IngestMethod,
    ProgramGroundTruth,
    ProgramId,
)

_SYNTH_PROV = make_provenance("synthetic", IngestMethod.SYNTHETIC, "synthetic", anonymized=True)


def _ev(etype: EvidenceType, value) -> Evidence:
    return Evidence(
        evidence_id=f"synthetic:{etype.value}",
        type=etype,
        value=value,
        provenance=_SYNTH_PROV,
    )


def build_all_evidence(situation: ApplicantSituation) -> list[Evidence]:
    """Materialize evidence for *every* known fact (the ground-truth view)."""
    ev: list[Evidence] = [
        _ev(EvidenceType.HOUSEHOLD_SIZE, float(situation.household_size)),
        _ev(EvidenceType.MONTHLY_INCOME, float(situation.monthly_income)),
        _ev(EvidenceType.ANNUAL_INCOME, float(situation.annual_income)),
        _ev(EvidenceType.LIQUID_ASSETS, float(situation.liquid_assets)),
        _ev(EvidenceType.RESIDENCY_STATE, situation.jurisdiction),
        _ev(EvidenceType.RESIDENT, bool(situation.resident)),
        _ev(EvidenceType.CITIZENSHIP_STATUS, situation.citizenship_status),
        _ev(EvidenceType.AGE, float(situation.age)),
        _ev(EvidenceType.DISABLED, bool(situation.disabled)),
        _ev(EvidenceType.PREGNANT, bool(situation.pregnant)),
        _ev(EvidenceType.HAS_DEPENDENT_CHILD, bool(situation.has_dependent_child)),
        _ev(EvidenceType.EMPLOYMENT_STATUS, situation.employment_status),
        _ev(EvidenceType.BASE_PERIOD_EARNINGS, float(situation.base_period_earnings)),
        _ev(EvidenceType.WEEKS_WORKED, float(situation.weeks_worked)),
        _ev(EvidenceType.ABLE_AND_AVAILABLE, bool(situation.able_and_available)),
        _ev(EvidenceType.MONTHLY_RENT, float(situation.monthly_rent)),
        _ev(EvidenceType.WAITLIST_STATUS, situation.waitlist_status.value),
    ]
    if situation.separation_reason is not None:
        ev.append(_ev(EvidenceType.SEPARATION_REASON, situation.separation_reason))
    if situation.days_since_denial is not None:
        ev.append(_ev(EvidenceType.DAYS_SINCE_DENIAL, float(situation.days_since_denial)))
    if situation.appeal_grounds is not None:
        ev.append(_ev(EvidenceType.APPEAL_GROUNDS, situation.appeal_grounds))
    return ev


def generic_ground_truth(
    program: ProgramId, situation: ApplicantSituation, profile: JurisdictionProfile, rationale: str
) -> ProgramGroundTruth:
    from ...types import EvidenceView

    ruleset = program_registry.get_ruleset(program)
    view = EvidenceView(build_all_evidence(situation))

    not_satisfied: list[str] = []
    all_required: list[str] = []
    for rule in ruleset.rules:
        if not rule.required:
            continue
        all_required.append(rule.criterion_id)
        outcome = rule.predicate(view, profile)
        if outcome is not CriterionOutcome.SATISFIED:
            not_satisfied.append(rule.criterion_id)

    label = GroundTruthLabel.INELIGIBLE if not_satisfied else GroundTruthLabel.ELIGIBLE
    decisive = not_satisfied if not_satisfied else all_required
    ambiguous = len(ruleset.ambiguity_signals(view, profile)) > 0
    return ProgramGroundTruth(
        program=program,
        label=label,
        ambiguous=ambiguous,
        rationale=rationale,
        decisive_criteria=decisive,
    )
