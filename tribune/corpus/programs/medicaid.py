"""Medicaid / health coverage.

Medicaid is the strongest stress test for TRIBUNE's abstention machinery: the
income pathway is heavily state-dependent (expansion vs. non-expansion, MAGI vs.
non-MAGI), and the coverage gap and immigration-status interactions are genuinely
ambiguous. The income-pathway predicate below encodes the category logic; the
synthetic labeler (``casegen/programs/medicaid.py``) flags the ambiguous edges so
that abstaining on them is the rewarded outcome.

Citations reference 42 CFR Part 435 and the Social Security Act eligibility
groups. Illustrative; validate against current state plans before any real use.
"""

from __future__ import annotations

from ...types import CriterionOutcome as CO
from ...types import EvidenceType as ET
from ...types import EvidenceView, ProgramId
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile

PROGRAM = ProgramId.MEDICAID


def _residency(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    r = view.flag(ET.RESIDENT)
    if r is None:
        return CO.UNKNOWN
    return CO.SATISFIED if r else CO.NOT_SATISFIED


def _citizenship(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    status = view.text(ET.CITIZENSHIP_STATUS)
    if status is None:
        return CO.UNKNOWN
    return CO.SATISFIED if status in ("citizen", "qualified_immigrant") else CO.NOT_SATISFIED


def _applicable_pct(view: EvidenceView, profile: JurisdictionProfile):
    """Return the FPL fraction threshold for the applicable eligibility category,
    or ``None`` when there is no MAGI pathway (e.g. a childless adult in a
    non-expansion state — the coverage gap)."""
    age = view.num(ET.AGE)
    pregnant = view.flag(ET.PREGNANT)
    has_child = view.flag(ET.HAS_DEPENDENT_CHILD)
    disabled = view.flag(ET.DISABLED)
    if age is None or pregnant is None or has_child is None or disabled is None:
        return "unknown"
    if pregnant:
        return profile.medicaid_pregnant_pct
    if age < 19:
        return profile.medicaid_child_pct
    if disabled or age >= 65:
        return 1.00  # aged/disabled non-MAGI pathway, approximated at 100% FPL
    if has_child:  # parent / caretaker relative
        return profile.medicaid_magi_adult_pct if profile.medicaid_expansion else profile.medicaid_parent_pct_nonexpansion
    # childless, non-disabled adult 19-64
    if profile.medicaid_expansion:
        return profile.medicaid_magi_adult_pct
    return None  # coverage gap: no pathway


def _income_pathway(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    income = view.num(ET.MONTHLY_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    pct = _applicable_pct(view, profile)
    if pct == "unknown" or income is None or hh is None:
        return CO.UNKNOWN
    if pct is None:
        # No MAGI pathway in this state for this category (coverage gap).
        return CO.NOT_SATISFIED
    limit = profile.fpl_monthly(int(hh)) * float(pct)
    return CO.SATISFIED if income <= limit else CO.NOT_SATISFIED


def _income_margin(view: EvidenceView, profile: JurisdictionProfile):
    income = view.num(ET.MONTHLY_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    pct = _applicable_pct(view, profile)
    if pct == "unknown" or pct is None or income is None or hh is None:
        return None
    limit = profile.fpl_monthly(int(hh)) * float(pct)
    if limit <= 0:
        return None
    return min(1.0, abs(income - limit) / limit)


def ambiguity(view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
    sigs: list[str] = []
    pct = _applicable_pct(view, profile)
    if pct is None:
        sigs.append(
            "a childless, non-disabled adult in a non-expansion state may fall in the "
            "Medicaid coverage gap (no eligibility pathway, but possibly marketplace help)"
        )
    status = view.text(ET.CITIZENSHIP_STATUS)
    if status == "qualified_immigrant":
        sigs.append("qualified-immigrant status can trigger the five-year bar and other interactions")
    m = _income_margin(view, profile)
    if m is not None and m < 0.07:
        sigs.append("household income is close to the applicable MAGI threshold")
    return sigs


RULES: list[Rule] = [
    Rule(
        criterion_id="residency",
        description="Applicant is a resident of the state.",
        required=True,
        source="42 CFR 435.403",
        title="State residence",
        text="A State must provide Medicaid to eligible residents of the State, "
        "including residents who are absent temporarily.",
        evidence_types=(ET.RESIDENT, ET.RESIDENCY_STATE),
        locator="https://www.ecfr.gov/current/title-42/section-435.403",
        predicate=_residency,
    ),
    Rule(
        criterion_id="citizenship",
        description="Applicant is a citizen or a qualified non-citizen meeting immigration rules.",
        required=True,
        source="42 CFR 435.406 / 8 USC 1612",
        title="Citizenship and immigration status",
        text="Medicaid is available to U.S. citizens and nationals and to qualified "
        "non-citizens; certain qualified immigrants are subject to a five-year waiting period.",
        evidence_types=(ET.CITIZENSHIP_STATUS,),
        locator="https://www.ecfr.gov/current/title-42/section-435.406",
        predicate=_citizenship,
    ),
    Rule(
        criterion_id="income_pathway",
        description="Household MAGI income is within the threshold for the applicable eligibility group.",
        required=True,
        source="42 CFR 435 Subpart B (incl. 435.119 expansion adults)",
        title="MAGI-based income eligibility groups",
        text="Eligibility is determined by modified adjusted gross income against the "
        "income standard for the applicable group: expansion adults to 133% FPL (with a "
        "5% disregard, effectively 138%), pregnant women and children at higher standards, "
        "and parents/caretakers per the State plan. In non-expansion States, childless "
        "non-disabled adults may have no eligibility pathway (the coverage gap).",
        evidence_types=(
            ET.MONTHLY_INCOME,
            ET.HOUSEHOLD_SIZE,
            ET.AGE,
            ET.PREGNANT,
            ET.HAS_DEPENDENT_CHILD,
            ET.DISABLED,
        ),
        locator="https://www.ecfr.gov/current/title-42/part-435/subpart-B",
        predicate=_income_pathway,
        margin_fn=_income_margin,
    ),
]

RULESET = RuleSet(program=PROGRAM, rules=RULES, ambiguity_fn=ambiguity)
