"""SNAP (Supplemental Nutrition Assistance Program) — food assistance.

Citations reference the federal SNAP regulations at 7 CFR Part 273. Income limits
are tied to the federal poverty guidelines via the jurisdiction profile. These are
illustrative and must be validated against current figures before any real use.
"""

from __future__ import annotations

from ...types import CriterionOutcome as CO
from ...types import EvidenceType as ET
from ...types import EvidenceView, ProgramId
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile

PROGRAM = ProgramId.SNAP


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


def _gross_income(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    income = view.num(ET.MONTHLY_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    if income is None or hh is None:
        return CO.UNKNOWN
    limit = profile.fpl_monthly(int(hh)) * profile.snap_gross_income_pct
    return CO.SATISFIED if income <= limit else CO.NOT_SATISFIED


def _gross_income_margin(view: EvidenceView, profile: JurisdictionProfile):
    income = view.num(ET.MONTHLY_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    if income is None or hh is None:
        return None
    limit = profile.fpl_monthly(int(hh)) * profile.snap_gross_income_pct
    if limit <= 0:
        return None
    return min(1.0, abs(income - limit) / limit)


def _assets(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    # Broad-Based Categorical Eligibility waives the asset test in many states.
    if profile.snap_bbce_waives_assets:
        return CO.SATISFIED
    assets = view.num(ET.LIQUID_ASSETS)
    if assets is None:
        return CO.UNKNOWN
    return CO.SATISFIED if assets <= profile.snap_asset_limit else CO.NOT_SATISFIED


def ambiguity(view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
    sigs: list[str] = []
    m = _gross_income_margin(view, profile)
    if m is not None and m < 0.05:
        sigs.append("gross income sits right at the SNAP 130% limit")
    return sigs


RULES: list[Rule] = [
    Rule(
        criterion_id="residency",
        description="Applicant resides in the jurisdiction administering the benefit.",
        required=True,
        source="7 CFR 273.3",
        title="Residency",
        text="A household shall live in the State in which it files an application "
        "for participation in the Supplemental Nutrition Assistance Program.",
        evidence_types=(ET.RESIDENT, ET.RESIDENCY_STATE),
        locator="https://www.ecfr.gov/current/title-7/section-273.3",
        predicate=_residency,
    ),
    Rule(
        criterion_id="citizenship",
        description="Applicant is a U.S. citizen or a qualified non-citizen.",
        required=True,
        source="7 CFR 273.4",
        title="Citizenship and alien status",
        text="To be eligible for SNAP, an individual must be a U.S. citizen or a "
        "qualified alien as defined in the regulations; ineligible aliens are excluded.",
        evidence_types=(ET.CITIZENSHIP_STATUS,),
        locator="https://www.ecfr.gov/current/title-7/section-273.4",
        predicate=_citizenship,
    ),
    Rule(
        criterion_id="gross_income",
        description="Household gross monthly income is at or below 130% of the poverty guideline.",
        required=True,
        source="7 CFR 273.9(a)(1)",
        title="Gross income eligibility standard",
        text="The gross income eligibility standard is 130 percent of the federal "
        "poverty income guidelines for the appropriate household size.",
        evidence_types=(ET.MONTHLY_INCOME, ET.HOUSEHOLD_SIZE),
        locator="https://www.ecfr.gov/current/title-7/section-273.9",
        predicate=_gross_income,
        margin_fn=_gross_income_margin,
    ),
    Rule(
        criterion_id="assets",
        description="Countable resources are within the resource limit (or waived under BBCE).",
        required=True,
        source="7 CFR 273.8 / 273.2(j)",
        title="Resource eligibility and broad-based categorical eligibility",
        text="Households must have countable resources at or below the resource limit, "
        "unless the State confers broad-based categorical eligibility which waives the asset test.",
        evidence_types=(ET.LIQUID_ASSETS,),
        locator="https://www.ecfr.gov/current/title-7/section-273.8",
        predicate=_assets,
    ),
]

RULESET = RuleSet(program=PROGRAM, rules=RULES, ambiguity_fn=ambiguity)
