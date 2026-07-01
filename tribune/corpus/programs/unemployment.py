"""Unemployment insurance.

Unemployment eligibility is state-law specific, so the citations here reference an
*example* state Unemployment Insurance Code (clearly fictional section numbers).
Monetary thresholds come from the jurisdiction profile. Separation reason is the
classic ambiguous axis: a quit "with good cause" is fact-intensive and is exactly
the kind of case TRIBUNE must escalate rather than guess.
"""

from __future__ import annotations

from ...types import CriterionOutcome as CO
from ...types import EvidenceType as ET
from ...types import EvidenceView, ProgramId
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile

PROGRAM = ProgramId.UNEMPLOYMENT


def _monetary(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    earnings = view.num(ET.BASE_PERIOD_EARNINGS)
    weeks = view.num(ET.WEEKS_WORKED)
    if earnings is None or weeks is None:
        return CO.UNKNOWN
    ok = earnings >= profile.ui_min_base_period_earnings and weeks >= profile.ui_min_weeks_worked
    return CO.SATISFIED if ok else CO.NOT_SATISFIED


def _monetary_margin(view: EvidenceView, profile: JurisdictionProfile):
    earnings = view.num(ET.BASE_PERIOD_EARNINGS)
    if earnings is None or profile.ui_min_base_period_earnings <= 0:
        return None
    return min(1.0, abs(earnings - profile.ui_min_base_period_earnings) / profile.ui_min_base_period_earnings)


def _separation(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    reason = view.text(ET.SEPARATION_REASON)
    if reason is None:
        return CO.UNKNOWN
    if reason == "laid_off":
        return CO.SATISFIED
    if reason == "quit_good_cause":
        return CO.SATISFIED  # ambiguous in practice; flagged by the synthetic labeler
    # quit_no_cause, fired_misconduct
    return CO.NOT_SATISFIED


def _able_available(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    aa = view.flag(ET.ABLE_AND_AVAILABLE)
    if aa is None:
        return CO.UNKNOWN
    return CO.SATISFIED if aa else CO.NOT_SATISFIED


def ambiguity(view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
    sigs: list[str] = []
    reason = view.text(ET.SEPARATION_REASON)
    if reason == "quit_good_cause":
        sigs.append("a voluntary quit 'with good cause' is fact-intensive and decided case by case")
    m = _monetary_margin(view, profile)
    if m is not None and m < 0.05:
        sigs.append("base-period earnings are right at the monetary eligibility minimum")
    return sigs


RULES: list[Rule] = [
    Rule(
        criterion_id="monetary_eligibility",
        description="Sufficient base-period earnings and weeks worked.",
        required=True,
        source="EX Unemployment Insurance Code §1276",
        title="Monetary eligibility",
        text="A claimant must have earned at least the statutory minimum in base-period "
        "wages and worked the minimum number of weeks to establish a valid claim.",
        evidence_types=(ET.BASE_PERIOD_EARNINGS, ET.WEEKS_WORKED),
        locator="example-state-code://unemployment/1276",
        predicate=_monetary,
        margin_fn=_monetary_margin,
    ),
    Rule(
        criterion_id="separation_reason",
        description="The reason for separation does not disqualify the claimant.",
        required=True,
        source="EX Unemployment Insurance Code §1256",
        title="Disqualifying separations",
        text="A claimant is disqualified if they left work voluntarily without good "
        "cause or were discharged for misconduct connected with the work. A layoff or a "
        "voluntary quit with good cause is not disqualifying.",
        evidence_types=(ET.SEPARATION_REASON,),
        locator="example-state-code://unemployment/1256",
        predicate=_separation,
    ),
    Rule(
        criterion_id="able_and_available",
        description="The claimant is able to work, available for work, and seeking work.",
        required=True,
        source="EX Unemployment Insurance Code §1253(c)",
        title="Able and available",
        text="To receive benefits for a week, a claimant must be able to work, available "
        "for work, and actively seeking suitable work during that week.",
        evidence_types=(ET.ABLE_AND_AVAILABLE,),
        locator="example-state-code://unemployment/1253",
        predicate=_able_available,
    ),
]

RULESET = RuleSet(program=PROGRAM, rules=RULES, ambiguity_fn=ambiguity)
