"""Privacy-preserving synthetic case generator & interactive scenario environment.

Implements Dual-Agent Scenario Mining:
1. Research Agent: Parses raw statutory rules, program guidelines, and citation indices
   from tribune/corpus/programs/ and RuleStore to identify eligibility boundaries and ambiguity vectors.
2. Scenario Agent: Synthesizes complex, multi-turn factual cases embedded with hidden states
   (e.g., undisclosed household income, contradictory asset statements, ambiguous temporal dependency dates,
   unverified dependent custody, undocumented seasonal earnings).
3. Dual-Agent Scenario Miner & Synthetic Environment: Interactive scenario state machine allowing agents
   to perform legal discovery actions and output structured schema representations compatible with appeals eval.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..corpus import programs as program_registry
from ..corpus.programs.jurisdictions import JurisdictionProfile, get_profile
from ..corpus.rule_store import LocalRuleStore, RuleStore
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
from ..clients.routing import TieredRoutingGateway

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
    ambiguity_type: str = "hidden_secondary_income"  # "hidden_secondary_income" | "contradictory_assets" | "ambiguous_temporal_dates" | "ambiguous_custody" | "undocumented_earnings" | "unverified_assets"
    revealed_evidence_type: EvidenceType = EvidenceType.MONTHLY_INCOME
    contradiction_notes: str = ""


@dataclass(frozen=True)
class DiscoveryAction:
    """A legal discovery or fact-verification action an agent can take."""

    action_id: str
    action_type: str  # e.g. "request_w2_wages", "verify_custody_order", "cross_check_tax_records", "asset_audit", "request_agency_denial_record"
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
# Research Agent: Statutory Analysis & Boundary Extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AmbiguityVector:
    """A latent vulnerability or edge-case vector derived from statutory analysis."""

    vector_id: str
    program: ProgramId
    ambiguity_type: str
    statutory_basis: str
    target_evidence_type: EvidenceType
    default_action_type: str
    impact_description: str


@dataclass(frozen=True)
class ProgramResearchProfile:
    """Structured statutory research digest compiled by the Research Agent."""

    program: ProgramId
    jurisdiction: str
    required_criteria: list[str]
    rule_summaries: list[dict[str, Any]]
    income_thresholds: dict[str, float]
    asset_thresholds: dict[str, float]
    temporal_windows: dict[str, Any]
    ambiguity_vectors: list[AmbiguityVector]


class ResearchAgent:
    """Parses raw statutory rules and program guidelines from tribune/corpus/programs/ and RuleStore.

    Identifies:
    1. Mandatory statutory criteria and decision boundaries (FPL limits, MAGI thresholds, AMI caps).
    2. Temporal requirements (statutory appeal deadlines, base period work histories).
    3. Structural ambiguity vectors suitable for synthesizing adversarial or multi-turn cases.
    """

    def __init__(self, rule_store: RuleStore | None = None) -> None:
        self.rule_store = rule_store or LocalRuleStore()

    def parse_statutory_rules(self, program: ProgramId, jurisdiction: str = "EX") -> ProgramResearchProfile:
        """Parse raw statutory rules and compile a structured statutory research profile."""
        profile = get_profile(jurisdiction)
        ruleset = program_registry.get_ruleset(program)
        citations = self.rule_store.all_citations(program, jurisdiction)
        cit_map = {c.citation_id.split(":")[-1]: c for c in citations}

        rule_summaries: list[dict[str, Any]] = []
        for r in ruleset.rules:
            c = cit_map.get(r.criterion_id)
            rule_summaries.append(
                {
                    "criterion_id": r.criterion_id,
                    "description": r.description,
                    "required": r.required,
                    "citation_id": c.citation_id if c else f"{program.value}:{jurisdiction}:{r.criterion_id}",
                    "statutory_text": r.text,
                    "source": r.source,
                    "evidence_types": [et.value for et in r.evidence_types],
                }
            )

        # Extract thresholds
        income_thresholds: dict[str, float] = {}
        asset_thresholds: dict[str, float] = {}
        temporal_windows: dict[str, Any] = {}
        ambiguity_vectors: list[AmbiguityVector] = []

        if program == ProgramId.SNAP:
            income_thresholds["snap_gross_pct"] = profile.snap_gross_income_pct
            asset_thresholds["liquid_asset_limit"] = profile.snap_asset_limit
            ambiguity_vectors.extend([
                AmbiguityVector(
                    vector_id="snap:hidden_secondary_income",
                    program=ProgramId.SNAP,
                    ambiguity_type="hidden_secondary_income",
                    statutory_basis="7 C.F.R. § 273.9(a) Gross Income Test",
                    target_evidence_type=EvidenceType.MONTHLY_INCOME,
                    default_action_type="request_w2_wages",
                    impact_description="Undisclosed wage or gig income breaches SNAP 130% FPL gross threshold.",
                ),
                AmbiguityVector(
                    vector_id="snap:contradictory_assets",
                    program=ProgramId.SNAP,
                    ambiguity_type="contradictory_assets",
                    statutory_basis="7 C.F.R. § 273.8 Asset Limitation",
                    target_evidence_type=EvidenceType.LIQUID_ASSETS,
                    default_action_type="asset_audit",
                    impact_description="Contradiction between intake self-attestation ($500) and bank account records ($3,200).",
                ),
            ])

        elif program == ProgramId.UNEMPLOYMENT:
            income_thresholds["min_base_period_earnings"] = profile.ui_min_base_period_earnings
            temporal_windows["min_weeks_worked"] = profile.ui_min_weeks_worked
            ambiguity_vectors.extend([
                AmbiguityVector(
                    vector_id="ui:ambiguous_separation",
                    program=ProgramId.UNEMPLOYMENT,
                    ambiguity_type="ambiguous_separation",
                    statutory_basis="State UI Code § 1256 Voluntary Quit / Good Cause",
                    target_evidence_type=EvidenceType.SEPARATION_REASON,
                    default_action_type="request_employer_separation_statement",
                    impact_description="Discrepancy between voluntary resignation vs good cause constructive dismissal.",
                ),
                AmbiguityVector(
                    vector_id="ui:undocumented_earnings",
                    program=ProgramId.UNEMPLOYMENT,
                    ambiguity_type="undocumented_earnings",
                    statutory_basis="State UI Code § 1281 Base Period Earnings",
                    target_evidence_type=EvidenceType.BASE_PERIOD_EARNINGS,
                    default_action_type="cross_check_tax_records",
                    impact_description="Off-the-books seasonal agricultural earnings affecting base-period minimums.",
                ),
            ])

        elif program == ProgramId.MEDICAID:
            income_thresholds["magi_adult_pct"] = profile.medicaid_magi_adult_pct
            ambiguity_vectors.extend([
                AmbiguityVector(
                    vector_id="medicaid:ambiguous_custody",
                    program=ProgramId.MEDICAID,
                    ambiguity_type="ambiguous_custody",
                    statutory_basis="42 C.F.R. § 435.119 MAGI Household Composition",
                    target_evidence_type=EvidenceType.HAS_DEPENDENT_CHILD,
                    default_action_type="verify_custody_order",
                    impact_description="Shared custody order ambiguities determining MAGI household size and dependent qualifying child status.",
                ),
                AmbiguityVector(
                    vector_id="medicaid:coverage_gap",
                    program=ProgramId.MEDICAID,
                    ambiguity_type="coverage_gap",
                    statutory_basis="ACA § 2001 / Non-Expansion Medicaid Gap",
                    target_evidence_type=EvidenceType.MONTHLY_INCOME,
                    default_action_type="verify_jurisdiction_expansion_status",
                    impact_description="Childless adult below 100% FPL in non-expansion state triggering coverage gap abstention.",
                ),
            ])

        elif program == ProgramId.HOUSING:
            income_thresholds["very_low_ami_pct"] = profile.housing_very_low_pct
            income_thresholds["low_ami_pct"] = profile.housing_low_pct
            ambiguity_vectors.extend([
                AmbiguityVector(
                    vector_id="housing:waitlist_unknown",
                    program=ProgramId.HOUSING,
                    ambiguity_type="waitlist_unknown",
                    statutory_basis="24 C.F.R. § 982.204 Waiting List Administration",
                    target_evidence_type=EvidenceType.WAITLIST_STATUS,
                    default_action_type="query_pha_portal",
                    impact_description="PHA Section 8 voucher waitlist status unknown requiring agency registry inquiry.",
                )
            ])

        elif program == ProgramId.APPEALS:
            temporal_windows["appeal_window_days"] = profile.appeal_window_days
            ambiguity_vectors.extend([
                AmbiguityVector(
                    vector_id="appeals:ambiguous_temporal_dates",
                    program=ProgramId.APPEALS,
                    ambiguity_type="ambiguous_temporal_dates",
                    statutory_basis="7 C.F.R. § 273.15 Fair Hearings 90-Day Window",
                    target_evidence_type=EvidenceType.DAYS_SINCE_DENIAL,
                    default_action_type="request_agency_denial_record",
                    impact_description="Disputed postmark date versus notice issue date placing appeal filing at 89 vs 94 days.",
                )
            ])

        return ProgramResearchProfile(
            program=program,
            jurisdiction=jurisdiction,
            required_criteria=ruleset.required_ids,
            rule_summaries=rule_summaries,
            income_thresholds=income_thresholds,
            asset_thresholds=asset_thresholds,
            temporal_windows=temporal_windows,
            ambiguity_vectors=ambiguity_vectors,
        )

    def parse_all_programs(self, jurisdiction: str = "EX") -> dict[ProgramId, ProgramResearchProfile]:
        """Parse rules across all registered statutory programs."""
        return {p: self.parse_statutory_rules(p, jurisdiction) for p in _ALL_ORDER}


# --------------------------------------------------------------------------- #
# Scenario Agent: Multi-Turn Case Synthesis with Hidden States
# --------------------------------------------------------------------------- #


class ScenarioAgent:
    """Synthesizes complex, multi-turn factual cases embedded with hidden latent states.

    Embeds:
    - Undisclosed secondary household income.
    - Contradictory asset statements between self-attestation and banking audits.
    - Ambiguous temporal dependency dates relative to statutory appeal filing windows.
    - Contested dependent custody arrangements.
    """

    def __init__(self, seed: int = 7, research_agent: ResearchAgent | None = None) -> None:
        self.seed = seed
        self.research_agent = research_agent or ResearchAgent()
        self.rng = random.Random(seed)

    def synthesize_latent_facts(
        self,
        case_id: str,
        target_programs: list[ProgramId],
        ambiguity_types: list[str],
        base_situation: ApplicantSituation,
    ) -> list[LatentFact]:
        """Synthesize concrete latent facts corresponding to requested ambiguity types."""
        facts: list[LatentFact] = []

        if "hidden_secondary_income" in ambiguity_types:
            facts.append(
                LatentFact(
                    variable_id=f"{case_id}:sec_income",
                    name="Undisclosed Gig Economy / Secondary Monthly Earnings",
                    true_value=850.0,
                    is_revealed=False,
                    discovery_action_type="request_w2_wages",
                    statutory_impact="Increases gross monthly income by $850.00, pushing household past statutory limits",
                    ambiguity_type="hidden_secondary_income",
                    revealed_evidence_type=EvidenceType.MONTHLY_INCOME,
                    contradiction_notes="Intake reported $0 secondary income; discovery uncovers active rideshare 1099 wages.",
                )
            )

        if "contradictory_assets" in ambiguity_types or "unverified_assets" in ambiguity_types:
            facts.append(
                LatentFact(
                    variable_id=f"{case_id}:liquid_assets",
                    name="Liquid Asset Bank Account Audit",
                    true_value=3200.0 if "contradictory_assets" in ambiguity_types else 1200.0,
                    is_revealed=False,
                    discovery_action_type="asset_audit",
                    statutory_impact="Audits liquid asset holdings against statutory resource limits ($2,750 threshold)",
                    ambiguity_type="contradictory_assets" if "contradictory_assets" in ambiguity_types else "unverified_assets",
                    revealed_evidence_type=EvidenceType.LIQUID_ASSETS,
                    contradiction_notes="Self-reported liquid assets of $500 contradicted by $3,200 verified checking account balance.",
                )
            )

        if "ambiguous_temporal_dates" in ambiguity_types:
            facts.append(
                LatentFact(
                    variable_id=f"{case_id}:denial_notice_date",
                    name="Official Agency Denial Postmark & Service Verification",
                    true_value=82.0,
                    is_revealed=False,
                    discovery_action_type="request_agency_denial_record",
                    statutory_impact="Verifies actual service date was 82 days ago (within 90-day window) rather than 95 days",
                    ambiguity_type="ambiguous_temporal_dates",
                    revealed_evidence_type=EvidenceType.DAYS_SINCE_DENIAL,
                    contradiction_notes="Initial caseworker note estimated 95 days post-denial; official agency mail register certifies 82 days.",
                )
            )

        if "ambiguous_custody" in ambiguity_types:
            facts.append(
                LatentFact(
                    variable_id=f"{case_id}:custody_order",
                    name="Court-Ordered Shared Custody Decree Verification",
                    true_value=True,
                    is_revealed=False,
                    discovery_action_type="verify_custody_order",
                    statutory_impact="Confirms primary physical custody >50% for MAGI household and dependent child criteria",
                    ambiguity_type="ambiguous_custody",
                    revealed_evidence_type=EvidenceType.HAS_DEPENDENT_CHILD,
                    contradiction_notes="Informal joint parenting agreement clarified by formal family court custody decree.",
                )
            )

        if "undocumented_earnings" in ambiguity_types:
            facts.append(
                LatentFact(
                    variable_id=f"{case_id}:base_earnings",
                    name="Seasonal Cash Earnings & Wage Audit",
                    true_value=4500.0,
                    is_revealed=False,
                    discovery_action_type="cross_check_tax_records",
                    statutory_impact="Establishes qualifying base-period earnings exceeding UI statutory threshold",
                    ambiguity_type="undocumented_earnings",
                    revealed_evidence_type=EvidenceType.BASE_PERIOD_EARNINGS,
                    contradiction_notes="Unreported cash-in-hand agricultural earnings verified through state 1099-MISC repository.",
                )
            )

        return facts

    def synthesize_documents(
        self,
        case_id: str,
        situation: ApplicantSituation,
        target_programs: list[ProgramId],
        latent_facts: list[LatentFact] | None = None,
    ) -> list[RawDocument]:
        """Synthesize rich, multi-field raw application documents and supporting statements."""
        relevant = set()
        for p in target_programs:
            relevant.update(labelers.relevant_evidence(p))

        fields: dict[str, str] = {}
        for ev in build_all_evidence(situation):
            if ev.type in relevant:
                fields[ev.type.value] = _str_value(ev.value)

        # Format multi-section legal intake text
        intake_lines = [
            f"=== BENEFIT APPLICATION & INTAKE RECORD: {case_id} ===",
            f"Jurisdiction: {situation.jurisdiction}",
            f"Target Programs: {', '.join(p.value.upper() for p in target_programs)}",
            "--- APPLICANT DISCLOSURES ---",
        ]
        for k, v in fields.items():
            intake_lines.append(f"{k}: {v}")

        if situation.days_since_denial is not None:
            intake_lines.append(f"Notice of Action Denial: ~{situation.days_since_denial} days ago")
        if situation.appeal_grounds:
            intake_lines.append(f"Statement of Appeal Grounds: {situation.appeal_grounds}")

        docs = [
            RawDocument(
                doc_id=f"{case_id}:intake",
                doc_type="application_intake",
                text="\n".join(intake_lines),
                fields=fields,
            )
        ]

        if situation.days_since_denial is not None:
            docs.append(
                RawDocument(
                    doc_id=f"{case_id}:notice_of_action",
                    doc_type="agency_denial_notice",
                    text=f"STATE DEPARTMENT OF HUMAN SERVICES\nNOTICE OF ADVERSE ACTION\nCase: {case_id}\nDays elapsed: {situation.days_since_denial}\nGrounds: Eligibility threshold not met.",
                    fields={"days_since_denial": str(situation.days_since_denial)},
                )
            )

        return docs


# --------------------------------------------------------------------------- #
# Dual-Agent Scenario Miner & Generator
# --------------------------------------------------------------------------- #


class DualAgentScenarioMiner:
    """Orchestrates ResearchAgent and ScenarioAgent to generate complex synthetic test environments."""

    def __init__(self, seed: int = 7, gateway: TieredRoutingGateway | None = None) -> None:
        self.seed = seed
        self.gateway = gateway or TieredRoutingGateway()
        self.research_agent = ResearchAgent()
        self.scenario_agent = ScenarioAgent(seed=seed, research_agent=self.research_agent)

    def mine_scenario_environment(
        self,
        case_id: str,
        jurisdiction: str,
        overrides: dict,
        target_programs: list[ProgramId],
        latent_ambiguities: list[str] | None = None,
    ) -> tuple[SyntheticCase, SyntheticEnvironment]:
        """Mine a grounded synthetic case paired with an interactive multi-turn discovery environment."""
        fields = dict(_DEFAULTS)
        fields.update(overrides)
        situation = ApplicantSituation(case_id=case_id, jurisdiction=jurisdiction, **fields)
        profile = get_profile(jurisdiction)

        ambiguities = latent_ambiguities or ["hidden_secondary_income", "ambiguous_custody"]
        latent_facts = self.scenario_agent.synthesize_latent_facts(
            case_id=case_id,
            target_programs=target_programs,
            ambiguity_types=ambiguities,
            base_situation=situation,
        )

        documents = self.scenario_agent.synthesize_documents(
            case_id=case_id,
            situation=situation,
            target_programs=target_programs,
            latent_facts=latent_facts,
        )
        evidence = StructuredIngest().ingest_many(documents)
        ground_truth = {p: labelers.ground_truth(p, situation, profile) for p in target_programs}

        case = SyntheticCase(
            case_id=case_id,
            jurisdiction=jurisdiction,
            situation=situation,
            evidence=evidence,
            documents=documents,
            ground_truth=ground_truth,
            target_programs=target_programs,
        )

        env = SyntheticEnvironment(
            case_id=case_id,
            jurisdiction=jurisdiction,
            initial_situation=situation,
            initial_evidence=evidence,
            latent_facts=latent_facts,
        )
        return case, env


class SyntheticCaseGenerator:
    """High-assurance synthetic case generator backed by the Dual-Agent Scenario Mining architecture."""

    routing_intent: str = "synthetic_casegen"
    default_engine: str = "glm-5.3-flash"

    def __init__(self, seed: int = 7, gateway: TieredRoutingGateway | None = None) -> None:
        self.seed = seed
        self.gateway = gateway or TieredRoutingGateway()
        self.miner = DualAgentScenarioMiner(seed=seed, gateway=self.gateway)
        self.research_agent = self.miner.research_agent
        self.scenario_agent = self.miner.scenario_agent

    def generate_bulk_variations(
        self,
        base_prompt: str,
        n_variations: int = 5,
        target_program: ProgramId = ProgramId.SNAP,
        counterfactual_constraints: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate bulk scenario permutations routed to cost-deflated open MoE backends (GLM-5.3-Flash)."""
        return self.gateway.generate_bulk_scenarios(
            base_scenario_prompt=base_prompt,
            n_variations=n_variations,
            target_program=target_program.value,
            counterfactual_constraints=counterfactual_constraints,
        )

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
        return self.miner.mine_scenario_environment(
            case_id=case_id,
            jurisdiction=jurisdiction,
            overrides=overrides,
            target_programs=target_programs,
            latent_ambiguities=latent_ambiguities,
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


__all__ = [
    "LatentFact",
    "DiscoveryAction",
    "SyntheticEnvironment",
    "AmbiguityVector",
    "ProgramResearchProfile",
    "ResearchAgent",
    "ScenarioAgent",
    "DualAgentScenarioMiner",
    "SyntheticCaseGenerator",
]
