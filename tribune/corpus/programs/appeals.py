"""Appeals / fair-hearing workflow.

Unlike the benefit programs, the "eligibility" question here is whether the person
still has the *right to appeal* a denial (is the request timely, and are there
grounds). When that holds, the preparer assembles an appeal packet — but, as with
everything binding, nothing is filed without explicit human action.

Citations reference SNAP fair-hearing rules (7 CFR 273.15) and Medicaid fair-
hearing rules (42 CFR 431.221). The appeal window comes from the jurisdiction
profile (90 days by default).
"""

from __future__ import annotations

from ...types import CriterionOutcome as CO
from ...types import EvidenceType as ET
from ...types import EvidenceView, ProgramId
from .base import Rule, RuleSet
from .jurisdictions import JurisdictionProfile

PROGRAM = ProgramId.APPEALS


def _timely(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    days = view.num(ET.DAYS_SINCE_DENIAL)
    if days is None:
        return CO.UNKNOWN
    return CO.SATISFIED if days <= profile.appeal_window_days else CO.NOT_SATISFIED


def _timely_margin(view: EvidenceView, profile: JurisdictionProfile):
    days = view.num(ET.DAYS_SINCE_DENIAL)
    if days is None or profile.appeal_window_days <= 0:
        return None
    return min(1.0, abs(profile.appeal_window_days - days) / profile.appeal_window_days)


def _grounds(view: EvidenceView, profile: JurisdictionProfile) -> CO:
    grounds = view.text(ET.APPEAL_GROUNDS)
    if grounds is None or grounds.strip() == "":
        return CO.UNKNOWN
    return CO.SATISFIED


def ambiguity(view: EvidenceView, profile: JurisdictionProfile) -> list[str]:
    sigs: list[str] = []
    days = view.num(ET.DAYS_SINCE_DENIAL)
    if days is not None and 0 <= (profile.appeal_window_days - days) <= 7:
        sigs.append("the appeal deadline is within a week — a navigator should confirm timeliness urgently")
    return sigs


RULES: list[Rule] = [
    Rule(
        criterion_id="timely_request",
        description="The appeal/fair-hearing request is within the filing window.",
        required=True,
        source="7 CFR 273.15(g) / 42 CFR 431.221",
        title="Timely request for a fair hearing",
        text="A household or beneficiary must be allowed to request a fair hearing within "
        "the regulatory window (90 days from the date of the agency action being contested).",
        evidence_types=(ET.DAYS_SINCE_DENIAL, ET.DENIAL_DATE),
        locator="https://www.ecfr.gov/current/title-7/section-273.15",
        predicate=_timely,
        margin_fn=_timely_margin,
    ),
    Rule(
        criterion_id="grounds_stated",
        description="The person has stated a basis for contesting the agency action.",
        required=True,
        source="7 CFR 273.15(a)",
        title="Right to a fair hearing",
        text="A household has the right to a fair hearing to appeal a denial, reduction, "
        "or termination of benefits, or other adverse action it believes was wrong.",
        evidence_types=(ET.APPEAL_GROUNDS,),
        locator="https://www.ecfr.gov/current/title-7/section-273.15",
        predicate=_grounds,
    ),
]

RULESET = RuleSet(program=PROGRAM, rules=RULES, ambiguity_fn=ambiguity)
