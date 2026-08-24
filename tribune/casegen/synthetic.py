"""Privacy-preserving synthetic case generator & interactive scenario environment.

Produces:
1. Internally consistent applicant situations with labeled ground-truth eligibility.
2. Partially observed runnable synthetic environments with hidden variables (latent facts),
   such as hidden secondary household income, ambiguous dependent custody, undocumented
   seasonal earnings, and unverified asset thresholds.
3. Scenario state machine allowing agents to interactively perform legal discovery actions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..corpus.programs.jurisdictions import get_profile
from ..ingestion.structured import StructuredIngest
from ..types import (
    ApplicantSituation,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    Provenance,
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


def _str_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.2f}"
    return str(value)


# --------------------------------------------------------------------------- #
# Latent Variables & Interactive Scenario State Machine
# --------------------------------------------------------------------------- #


@dataclass
class LatentFact:
    """A hidden variable in the partially observed synthetic environment."""

    variable_id: str
    name: str
    true_value: Any
    is_revealed: bool = False
    discovery_action_type: str = "request_w2_wages"
    statutory_impact: str = "Income exceeds gross threshold if unverified secondary wages exist"
    ambiguity_type: str = "hidden_secondary_income"  # "hidden_secondary_income" | "ambiguous_custody" | "undocumented_earnings" | "unverified_assets"
    revealed_evidence_type: EvidenceType = EvidenceType.MONTHLY_INCOME


@dataclass(frozen=True)
class DiscoveryAction:
    """A legal discovery or fact-verification action an agent can take."""

    action_id: str
    action_type: str  # e.g. "request_w2_wages", "verify_custody_order", "cross_check_tax_records", "asset_audit"
    description: str
    target_variable: str
    cost_usd: float = 0.005
    latency_ms: float = 85.0


class SyntheticEnvironment:
    """Interactive runnable scenario state machine with hidden latent variables.

    Simulates a real-world welfare intake where initial documentation is incomplete
    or contains latent ambiguities. Agents can perform step-by-step discovery actions
    to uncover latent facts before rendering a final eligibility determination.
    """

    def __init__(
        self,
        case_id: str,
        jurisdiction: str,
        initial_situation: ApplicantSituation,
        initial_evidence: list[Evidence],
        latent_facts: list[LatentFact] | None = None,
    ) -> None:
        self.case_id = case_id
        self.jurisdiction = jurisdiction
        self.situation = initial_situation
        self.evidence = list(initial_evidence)
        self.latent_facts: dict[str, LatentFact] = {f.variable_id: f for f in (latent_facts or [])}
        self.action_history: list[dict[str, Any]] = []
        self.total_discovery_cost = 0.0

    def get_visible_evidence(self) -> list[Evidence]:
        """Return currently observable evidence list."""
        return list(self.evidence)

    def get_available_actions(self) -> list[DiscoveryAction]:
        """Return discovery actions available to reveal unobserved latent facts."""
        actions: list[DiscoveryAction] = []
        for lf in self.latent_facts.values():
            if not lf.is_revealed:
                actions.append(
                    DiscoveryAction(
                        action_id=f"action:{lf.variable_id}",
                        action_type=lf.discovery_action_type,
                        description=f"Perform discovery for {lf.name} ({lf.ambiguity_type})",
                        target_variable=lf.variable_id,
                        cost_usd=0.005,
                        latency_ms=85.0,
                    )
                )
        return actions

    def has_unresolved_blocking_ambiguities(self) -> bool:
        """Check if any critical latent facts remain hidden."""
        return any(not lf.is_revealed for lf in self.latent_facts.values())

    def step(self, action: DiscoveryAction | str) -> tuple[Evidence | None, float, bool, dict[str, Any]]:
        """Execute a legal discovery action to uncover latent facts.

        Returns: (newly_uncovered_evidence, action_cost, all_resolved, metadata)
        """
        action_type = action.action_type if isinstance(action, DiscoveryAction) else str(action)
        target_var = action.target_variable if isinstance(action, DiscoveryAction) else None

        cost = action.cost_usd if isinstance(action, DiscoveryAction) else 0.005
        self.total_discovery_cost += cost

        uncovered_ev: Evidence | None = None
        matched_fact: LatentFact | None = None

        for lf in self.latent_facts.values():
            if not lf.is_revealed:
                if (target_var and lf.variable_id == target_var) or (lf.discovery_action_type == action_type):
                    matched_fact = lf
                    lf.is_revealed = True
                    # Create structured evidence
                    prov = Provenance(
                        source_doc_id=f"discovery:{action_type}:{lf.variable_id}",
                        ingest_method=IngestMethod.STRUCTURED,
                        anonymized=True,
                        content_hash=f"latent_hash_{lf.variable_id}",
                    )
                    uncovered_ev = Evidence(
                        evidence_id=f"ev_latent_{lf.variable_id}",
                        type=lf.revealed_evidence_type,
                        value=lf.true_value,
                        provenance=prov,
                    )
                    self.evidence.append(uncovered_ev)
                    break

        all_resolved = not self.has_unresolved_blocking_ambiguities()

        step_record = {
            "action_type": action_type,
            "target_variable": target_var,
            "matched_latent_fact": matched_fact.name if matched_fact else None,
            "cost_usd": cost,
            "all_resolved": all_resolved,
        }
        self.action_history.append(step_record)

        return uncovered_ev, cost, all_resolved, step_record


# --------------------------------------------------------------------------- #
# Synthetic Case Generator
# --------------------------------------------------------------------------- #


class SyntheticCaseGenerator:
    routing_intent: str = "synthetic_casegen"
    default_engine: str = "DeepSeek V4 Pro"

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

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

    def build_scenario_environment(
        self,
        case_id: str,
        jurisdiction: str,
        overrides: dict,
        target_programs: list[ProgramId],
        latent_ambiguities: list[str] | None = None,
    ) -> tuple[SyntheticCase, SyntheticEnvironment]:
        """Generate a complete runnable synthetic case paired with an interactive SyntheticEnvironment."""
        case = self.build_case(case_id, jurisdiction, overrides, target_programs)
        latent_facts: list[LatentFact] = []

        ambiguities = latent_ambiguities or ["hidden_secondary_income", "ambiguous_custody"]

        if "hidden_secondary_income" in ambiguities:
            latent_facts.append(
                LatentFact(
                    variable_id=f"{case_id}:sec_income",
                    name="Undocumented Gig Economy / Secondary Monthly Earnings",
                    true_value=850.0,
                    is_revealed=False,
                    discovery_action_type="request_w2_wages",
                    statutory_impact="Increases gross monthly income by $850, testing MAGI & SNAP limits",
                    ambiguity_type="hidden_secondary_income",
                    revealed_evidence_type=EvidenceType.MONTHLY_INCOME,
                )
            )

        if "ambiguous_custody" in ambiguities:
            latent_facts.append(
                LatentFact(
                    variable_id=f"{case_id}:custody",
                    name="Court-Ordered Dependent Shared Custody Verification",
                    true_value=True,
                    is_revealed=False,
                    discovery_action_type="verify_custody_order",
                    statutory_impact="Establishes qualifying dependent child status for family Medicaid",
                    ambiguity_type="ambiguous_custody",
                    revealed_evidence_type=EvidenceType.HAS_DEPENDENT_CHILD,
                )
            )

        if "unverified_assets" in ambiguities:
            latent_facts.append(
                LatentFact(
                    variable_id=f"{case_id}:assets",
                    name="Liquid Asset Bank Account Audit",
                    true_value=1200.0,
                    is_revealed=False,
                    discovery_action_type="asset_audit",
                    statutory_impact="Verifies liquid assets remain below SNAP $2,750 threshold",
                    ambiguity_type="unverified_assets",
                    revealed_evidence_type=EvidenceType.LIQUID_ASSETS,
                )
            )

        env = SyntheticEnvironment(
            case_id=case_id,
            jurisdiction=jurisdiction,
            initial_situation=case.situation,
            initial_evidence=case.evidence,
            latent_facts=latent_facts,
        )
        return case, env

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


__all__ = [
    "LatentFact",
    "DiscoveryAction",
    "SyntheticEnvironment",
    "SyntheticCaseGenerator",
]
