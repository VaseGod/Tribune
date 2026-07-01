"""Housing assistance — Housing Choice Voucher (Section 8) / public housing.

Housing carries a safety property that is easy to get catastrophically wrong:
*eligibility is not access*. Income eligibility is Area-Median-Income based and
locality specific, but actual access is gated by waitlists that are frequently
closed or years long. These rules determine eligibility only; the waitlist status
is carried separately on the assessment and the system must never imply imminent
access. Where waitlist status is unknown, TRIBUNE abstains on the access question.

Citations reference 24 CFR Parts 5 and 982. Income limits derive from the
jurisdiction profile's AMI. Illustrative; validate before any real use.
"""

from __future__ import annotations

from ...types import CriterionOutcome as CO
from ...types import EvidenceType as ET
from ...types import EvidenceView, ProgramId
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile

PROGRAM = ProgramId.HOUSING


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


def _income_limit(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    annual = view.num(ET.ANNUAL_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    if annual is None or hh is None:
        return CO.UNKNOWN
    low_limit = profile.ami_limit_annual(int(hh), profile.housing_low_pct)  # 80% AMI
    return CO.SATISFIED if annual <= low_limit else CO.NOT_SATISFIED


def _income_margin(view: EvidenceView, profile: JurisdictionProfile):
    annual = view.num(ET.ANNUAL_INCOME)
    hh = view.num(ET.HOUSEHOLD_SIZE)
    if annual is None or hh is None:
        return None
    low_limit = profile.ami_limit_annual(int(hh), profile.housing_low_pct)
    if low_limit <= 0:
        return None
    return min(1.0, abs(annual - low_limit) / low_limit)


def ambiguity(view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
    sigs: list[str] = []
    ws = view.text(ET.WAITLIST_STATUS)
    if ws is None or ws == "unknown":
        sigs.append("waitlist status is unknown — being eligible is not the same as having access")
    elif ws == "closed":
        sigs.append("the waitlist is closed — eligible families cannot currently apply for a voucher")
    m = _income_margin(view, profile)
    if m is not None and m < 0.05:
        sigs.append("annual income is near the HUD income limit")
    return sigs


RULES: list[Rule] = [
    Rule(
        criterion_id="income_limit",
        description="Household annual income is within the applicable HUD income limit.",
        required=True,
        source="24 CFR 982.201(b)",
        title="Income eligibility for the Housing Choice Voucher program",
        text="A family is income-eligible if its annual income does not exceed the "
        "applicable HUD income limit for the area: very low income is 50% of area median "
        "income and low income is 80%; admissions are targeted to extremely low and very "
        "low income families.",
        evidence_types=(ET.ANNUAL_INCOME, ET.HOUSEHOLD_SIZE),
        locator="https://www.ecfr.gov/current/title-24/section-982.201",
        predicate=_income_limit,
        margin_fn=_income_margin,
    ),
    Rule(
        criterion_id="citizenship",
        description="Family members have eligible citizenship or immigration status.",
        required=True,
        source="24 CFR 5.508",
        title="Eligible immigration status",
        text="Assistance is restricted to U.S. citizens and non-citizens with eligible "
        "immigration status; prorated assistance may apply to mixed families.",
        evidence_types=(ET.CITIZENSHIP_STATUS,),
        locator="https://www.ecfr.gov/current/title-24/section-5.508",
        predicate=_citizenship,
    ),
    Rule(
        criterion_id="jurisdiction_residency",
        description="Applicant is within the public housing agency's jurisdiction.",
        required=True,
        source="24 CFR 982.201",
        title="Eligibility within the PHA jurisdiction",
        text="A public housing agency determines eligibility for families within its "
        "jurisdiction; residency or other local preferences may apply.",
        evidence_types=(ET.RESIDENT, ET.RESIDENCY_STATE),
        locator="https://www.ecfr.gov/current/title-24/section-982.201",
        predicate=_residency,
    ),
]

RULESET = RuleSet(program=PROGRAM, rules=RULES, ambiguity_fn=ambiguity)
