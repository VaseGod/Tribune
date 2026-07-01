"""Privacy-preserving synthetic case generator.

Produces internally consistent applicant situations with labeled ground-truth
eligibility and a controlled, class-imbalanced distribution (mostly clear cases,
a minority of genuinely ambiguous ones). No real PII is used or produced. The
situations are the source of both the evaluation data and the demo walkthrough.

Each case carries:
* the complete ``situation`` (the truth),
* the ``documents`` TRIBUNE actually receives (a subset, ingested into evidence),
* per-program ``ground_truth`` (label + ambiguity), computed from the corpus rules.
"""

from __future__ import annotations

import random

from ..corpus.programs.jurisdictions import get_profile
from ..ingestion.structured import StructuredIngest
from ..types import (
    ApplicantSituation,
    ProgramId,
    RawDocument,
    SyntheticCase,
    WaitlistStatus,
)
from . import programs as labelers
from .programs.base import build_all_evidence

_DEFAULTS = dict(
    household_size=1,
    monthly_income=0.0,
    liquid_assets=500.0,
    resident=True,
    citizenship_status="citizen",
    age=35,
    disabled=False,
    pregnant=False,
    has_dependent_child=False,
    employment_status="employed",
    separation_reason=None,
    base_period_earnings=0.0,
    weeks_worked=0,
    able_and_available=True,
    monthly_rent=900.0,
    waitlist_status=WaitlistStatus.UNKNOWN,
    days_since_denial=None,
    appeal_grounds=None,
)

_BENEFIT_ORDER = [ProgramId.SNAP, ProgramId.UNEMPLOYMENT, ProgramId.MEDICAID, ProgramId.HOUSING]
_ALL_ORDER = _BENEFIT_ORDER + [ProgramId.APPEALS]


def _str_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.2f}"
    return str(value)


class SyntheticCaseGenerator:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    # -- public API --------------------------------------------------------- #

    def build_case(
        self,
        case_id: str,
        jurisdiction: str,
        overrides: dict,
        target_programs: list[ProgramId],
        omit: set | None = None,
    ) -> SyntheticCase:
        fields = dict(_DEFAULTS)
        fields.update(overrides)
        situation = ApplicantSituation(case_id=case_id, jurisdiction=jurisdiction, **fields)
        profile = get_profile(jurisdiction)

        documents = self._documents(case_id, situation, target_programs, omit or set())
        evidence = StructuredIngest().ingest_many(documents)
        ground_truth = {p: labelers.ground_truth(p, situation, profile) for p in target_programs}
        return SyntheticCase(
            case_id=case_id,
            jurisdiction=jurisdiction,
            situation=situation,
            evidence=evidence,
            documents=documents,
            ground_truth=ground_truth,
            target_programs=target_programs,
        )

    def generate_eval_set(
        self, n_per_program: int = 24, ambiguous_ratio: float = 0.25
    ) -> list[SyntheticCase]:
        rng = random.Random(self.seed)
        cases: list[SyntheticCase] = []
        n_amb = round(n_per_program * ambiguous_ratio)
        for program in _ALL_ORDER:
            for i in range(n_per_program):
                if i < n_amb:
                    desired = "ambiguous"
                elif i % 2 == 0:
                    desired = "eligible"
                else:
                    desired = "ineligible"
                overrides, jurisdiction = self._tailor(program, desired, rng)
                cases.append(
                    self.build_case(
                        case_id=f"eval-{program.value}-{i}",
                        jurisdiction=jurisdiction,
                        overrides=overrides,
                        target_programs=[program],
                    )
                )
        return cases

    def generate_demo_set(self) -> list[SyntheticCase]:
        """A handful of narrative cases covering prepare / replan / abstain / assert-ineligible."""
        cases = []

        # A) Low-income parent (EX): SNAP eligible -> prepare; Medicaid eligible ->
        #    replan (complex) then prepare; Housing eligible but waitlist unknown -> abstain.
        cases.append(
            self.build_case(
                "demo-parent-multi",
                "EX",
                dict(
                    household_size=3,
                    has_dependent_child=True,
                    age=34,
                    monthly_income=1500.0,
                    liquid_assets=400.0,
                    waitlist_status=WaitlistStatus.UNKNOWN,
                ),
                [ProgramId.SNAP, ProgramId.MEDICAID, ProgramId.HOUSING],
            )
        )

        # B) Unemployment, quit "with good cause" -> ambiguous -> abstain.
        cases.append(
            self.build_case(
                "demo-quit-good-cause",
                "EX",
                dict(
                    employment_status="unemployed",
                    separation_reason="quit_good_cause",
                    base_period_earnings=22000.0,
                    weeks_worked=40,
                    able_and_available=True,
                    monthly_income=0.0,
                ),
                [ProgramId.UNEMPLOYMENT],
            )
        )

        # C) Childless adult below poverty in a NON-expansion state -> Medicaid
        #    coverage gap -> ambiguous -> abstain.
        cases.append(
            self.build_case(
                "demo-coverage-gap",
                "NX",
                dict(age=40, has_dependent_child=False, disabled=False, monthly_income=900.0),
                [ProgramId.MEDICAID],
            )
        )

        # D) Appeal of a denial, 80 days in (within the 90-day window) -> prepare an appeal.
        cases.append(
            self.build_case(
                "demo-appeal",
                "EX",
                dict(
                    days_since_denial=80,
                    appeal_grounds="My SNAP was denied but my income is below the limit.",
                    monthly_income=1200.0,
                ),
                [ProgramId.APPEALS],
            )
        )

        # E) SNAP, income well above the limit -> assert likely-ineligible + recommend human review.
        cases.append(
            self.build_case(
                "demo-snap-ineligible",
                "EX",
                dict(household_size=1, monthly_income=6000.0, liquid_assets=200.0),
                [ProgramId.SNAP],
            )
        )
        return cases

    # -- situation tailoring ------------------------------------------------ #

    def _tailor(self, program: ProgramId, desired: str, rng: random.Random) -> tuple[dict, str]:
        jurisdiction = "EX"
        hh = rng.choice([1, 2, 3, 4])
        o: dict = dict(
            household_size=hh,
            age=rng.randint(25, 55),
            liquid_assets=float(rng.randint(0, 1500)),
            monthly_rent=float(rng.randint(600, 1800)),
            monthly_income=0.0,
        )
        profile = get_profile(jurisdiction)

        if program is ProgramId.SNAP:
            limit = profile.fpl_monthly(hh) * profile.snap_gross_income_pct
            if desired == "eligible":
                o["monthly_income"] = limit * rng.uniform(0.3, 0.85)
            elif desired == "ineligible":
                o["monthly_income"] = limit * rng.uniform(1.3, 2.0)
            else:
                o["monthly_income"] = limit * rng.uniform(0.96, 0.995)

        elif program is ProgramId.UNEMPLOYMENT:
            o["employment_status"] = "unemployed"
            o["weeks_worked"] = rng.randint(10, 45)
            o["monthly_income"] = float(rng.randint(0, 500))
            min_e = profile.ui_min_base_period_earnings
            if desired == "eligible":
                o["separation_reason"] = "laid_off"
                o["base_period_earnings"] = min_e * rng.uniform(1.5, 4.0)
            elif desired == "ineligible":
                o["separation_reason"] = rng.choice(["fired_misconduct", "quit_no_cause"])
                o["base_period_earnings"] = min_e * rng.uniform(1.5, 4.0)
            else:
                if rng.random() < 0.7:
                    o["separation_reason"] = "quit_good_cause"
                    o["base_period_earnings"] = min_e * rng.uniform(1.5, 4.0)
                else:
                    o["separation_reason"] = "laid_off"
                    o["base_period_earnings"] = min_e * rng.uniform(0.97, 1.03)

        elif program is ProgramId.MEDICAID:
            o["age"] = rng.randint(19, 60)
            o["has_dependent_child"] = False
            if desired == "eligible":
                lim = profile.fpl_monthly(hh) * profile.medicaid_magi_adult_pct
                o["monthly_income"] = lim * rng.uniform(0.4, 0.9)
            elif desired == "ineligible":
                lim = profile.fpl_monthly(hh) * profile.medicaid_magi_adult_pct
                o["monthly_income"] = lim * rng.uniform(1.6, 2.5)
            else:
                if rng.random() < 0.6:
                    jurisdiction = "NX"
                    profile = get_profile("NX")
                    o["monthly_income"] = profile.fpl_monthly(hh) * rng.uniform(0.3, 0.9)
                else:
                    lim = profile.fpl_monthly(hh) * profile.medicaid_magi_adult_pct
                    o["monthly_income"] = lim * rng.uniform(0.95, 1.05)

        elif program is ProgramId.HOUSING:
            vlow = profile.ami_limit_annual(hh, profile.housing_very_low_pct)
            low = profile.ami_limit_annual(hh, profile.housing_low_pct)
            if desired == "eligible":
                ann = vlow * rng.uniform(0.3, 0.85)
                o["waitlist_status"] = WaitlistStatus.OPEN
            elif desired == "ineligible":
                ann = low * rng.uniform(1.2, 1.8)
                o["waitlist_status"] = WaitlistStatus.OPEN
            else:
                if rng.random() < 0.6:
                    ann = vlow * rng.uniform(0.4, 0.9)
                    o["waitlist_status"] = rng.choice([WaitlistStatus.UNKNOWN, WaitlistStatus.CLOSED])
                else:
                    ann = low * rng.uniform(0.97, 0.995)
                    o["waitlist_status"] = WaitlistStatus.OPEN
            o["monthly_income"] = ann / 12.0

        elif program is ProgramId.APPEALS:
            w = profile.appeal_window_days
            o["appeal_grounds"] = "Benefits were denied but I believe I qualify."
            o["monthly_income"] = float(rng.randint(0, 1500))
            if desired == "eligible":
                o["days_since_denial"] = rng.randint(10, w - 10)
            elif desired == "ineligible":
                o["days_since_denial"] = rng.randint(w + 5, w + 60)
            else:
                o["days_since_denial"] = rng.randint(w - 6, w)

        return o, jurisdiction

    # -- document construction --------------------------------------------- #

    def _documents(
        self,
        case_id: str,
        situation: ApplicantSituation,
        target_programs: list[ProgramId],
        omit: set,
    ) -> list[RawDocument]:
        relevant = set()
        for p in target_programs:
            relevant.update(labelers.relevant_evidence(p))
        relevant -= set(omit)

        fields: dict[str, str] = {}
        for ev in build_all_evidence(situation):
            if ev.type in relevant:
                fields[ev.type.value] = _str_value(ev.value)

        text = "\n".join(f"{k}: {v}" for k, v in fields.items())
        return [
            RawDocument(
                doc_id=f"{case_id}:intake",
                doc_type="application_intake",
                text=text,
                fields=fields,
            )
        ]
