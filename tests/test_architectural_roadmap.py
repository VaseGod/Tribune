"""Unit & Integration Tests for Project Tribune Architectural Roadmap.

Tests:
1. Dual-Agent Scenario Mining (ResearchAgent, ScenarioAgent, DualAgentScenarioMiner).
2. Dynamic Programmatic Verification, Tripartite Self-Testing Suite, & Step-level Advantage Optimization (SAO).
3. Autoresearch Optimization Ratchet (Time-bounded experiment execution, mutation benchmarking, and acceptance gating).
4. Deterministic Systems Correctness (Bitwise Logprob Parity) & Air-Gapped Sovereign Hybrid Routing.
"""

import math
import pytest
from tribune.agents.verifier import (
    DynamicVerificationFunctionGenerator,
    StepAdvantageOptimizer,
    TripartiteCheckResult,
    TripartiteSelfTestSuite,
    Verifier,
    VerifierAgent,
)
from tribune.casegen.synthetic import (
    DualAgentScenarioMiner,
    ProgramResearchProfile,
    ResearchAgent,
    ScenarioAgent,
    SyntheticCaseGenerator,
)
from tribune.corpus.rule_store import LocalRuleStore
from tribune.eval.appeals_eval import run_appeals_eval
from tribune.orchestration.continual_optimizer import (
    AutoresearchRatchetLoop,
    ContinualOptimizer,
    MutationType,
    RatchetAcceptanceGate,
    RatchetExperimentResult,
    RatchetMutationProposal,
)
from tribune.orchestration.pipeline import TrajectoryBuffer, TrajectoryFrame
from tribune.providers.local_rules import LocalRulesProvider
from tribune.providers.router import (
    BitwiseParityEnforcer,
    DataSovereigntyLevel,
    ModelRouter,
)
from tribune.providers.base import SynthesisRequest
from tribune.types import (
    ApplicantSituation,
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    Provenance,
    RecommendedAction,
    SMState,
)


# =========================================================================== #
# Pillar 1: Dual-Agent Scenario Mining Tests
# =========================================================================== #


def test_research_agent_parses_statutory_rules_and_ambiguity_vectors():
    """Verify that ResearchAgent parses raw statutory rules and extracts structured research profiles."""
    store = LocalRuleStore()
    research_agent = ResearchAgent(rule_store=store)

    # 1. Test SNAP program parsing
    snap_profile = research_agent.parse_statutory_rules(ProgramId.SNAP, "EX")
    assert isinstance(snap_profile, ProgramResearchProfile)
    assert snap_profile.program == ProgramId.SNAP
    assert snap_profile.jurisdiction == "EX"
    assert "gross_income" in snap_profile.required_criteria
    assert len(snap_profile.rule_summaries) >= 4
    assert len(snap_profile.ambiguity_vectors) >= 2
    assert any(v.ambiguity_type == "hidden_secondary_income" for v in snap_profile.ambiguity_vectors)

    # 2. Test Appeals program parsing
    appeals_profile = research_agent.parse_statutory_rules(ProgramId.APPEALS, "EX")
    assert appeals_profile.program == ProgramId.APPEALS
    assert "appeal_window_days" in appeals_profile.temporal_windows
    assert any(v.ambiguity_type == "ambiguous_temporal_dates" for v in appeals_profile.ambiguity_vectors)

    # 3. Parse all programs
    all_profiles = research_agent.parse_all_programs("EX")
    assert len(all_profiles) == 5
    assert set(all_profiles.keys()) == {
        ProgramId.SNAP,
        ProgramId.UNEMPLOYMENT,
        ProgramId.MEDICAID,
        ProgramId.HOUSING,
        ProgramId.APPEALS,
    }


def test_scenario_agent_embeds_hidden_states_and_synthesizes_documents():
    """Verify that ScenarioAgent synthesizes multi-turn cases with embedded latent states."""
    scenario_agent = ScenarioAgent(seed=42)
    situation = SyntheticCaseGenerator().build_case(
        case_id="case_sao_test",
        jurisdiction="EX",
        overrides={
            "household_size": 2,
            "monthly_income": 1200.0,
            "liquid_assets": 500.0,
            "days_since_denial": 85,
            "appeal_grounds": "Disputed denial calculation.",
        },
        target_programs=[ProgramId.SNAP, ProgramId.APPEALS],
    ).situation

    # 1. Synthesize latent facts across multiple ambiguity vectors
    ambiguities = ["hidden_secondary_income", "contradictory_assets", "ambiguous_temporal_dates"]
    latent_facts = scenario_agent.synthesize_latent_facts(
        case_id="case_sao_test",
        target_programs=[ProgramId.SNAP, ProgramId.APPEALS],
        ambiguity_types=ambiguities,
        base_situation=situation,
    )

    assert len(latent_facts) == 3
    sec_inc = next(f for f in latent_facts if f.ambiguity_type == "hidden_secondary_income")
    assert sec_inc.true_value == 850.0
    assert sec_inc.discovery_action_type == "request_w2_wages"

    contra_asset = next(f for f in latent_facts if f.ambiguity_type == "contradictory_assets")
    assert contra_asset.true_value == 3200.0
    assert contra_asset.discovery_action_type == "asset_audit"

    temp_date = next(f for f in latent_facts if f.ambiguity_type == "ambiguous_temporal_dates")
    assert temp_date.true_value == 82.0
    assert temp_date.discovery_action_type == "request_agency_denial_record"

    # 2. Synthesize rich documents
    docs = scenario_agent.synthesize_documents(
        case_id="case_sao_test",
        situation=situation,
        target_programs=[ProgramId.SNAP, ProgramId.APPEALS],
        latent_facts=latent_facts,
    )
    assert len(docs) >= 2
    intake_doc = next(d for d in docs if d.doc_type == "application_intake")
    assert "BENEFIT APPLICATION & INTAKE RECORD" in intake_doc.text
    assert "household_size: 2" in intake_doc.text


def test_dual_agent_scenario_miner_integration():
    """Verify DualAgentScenarioMiner and SyntheticCaseGenerator integration with structured schema."""
    miner = DualAgentScenarioMiner(seed=123)
    case, env = miner.mine_scenario_environment(
        case_id="case_miner_test",
        jurisdiction="EX",
        overrides={"monthly_income": 950.0, "household_size": 1},
        target_programs=[ProgramId.SNAP, ProgramId.MEDICAID],
        latent_ambiguities=["hidden_secondary_income", "ambiguous_custody"],
    )

    assert case.case_id == "case_miner_test"
    assert env.has_unresolved_blocking_ambiguities() is True
    actions = env.get_available_actions()
    assert len(actions) == 2

    # Step through discovery
    new_ev, cost, resolved, record = env.step(actions[0])
    assert new_ev is not None
    assert cost == 0.005
    assert not resolved

    new_ev2, cost2, resolved2, record2 = env.step(actions[1])
    assert new_ev2 is not None
    assert resolved2 is True
    assert env.has_unresolved_blocking_ambiguities() is False


# =========================================================================== #
# Pillar 2: Programmatic Verifier Self-Testing & SAO Tests
# =========================================================================== #


def test_dynamic_verification_code_generation():
    """Verify dynamic programmatic verification code generation without reference solutions."""
    code_snap = DynamicVerificationFunctionGenerator.generate_verification_code(ProgramId.SNAP, "EX")
    assert "def verify_candidate_determination" in code_snap
    assert "gross_limit" in code_snap
    assert "recomputed_criteria" in code_snap

    code_appeals = DynamicVerificationFunctionGenerator.generate_verification_code(ProgramId.APPEALS, "EX")
    assert "appeal_window_days" in code_appeals
    assert "timely_filing" in code_appeals


def test_tripartite_self_test_suite_execution():
    """Verify that TripartiteSelfTestSuite executes Oracle, No-Op, and Unsolved-State checks."""
    store = LocalRuleStore()
    provider = LocalRulesProvider(role="verifier")
    verifier = VerifierAgent(provider=provider, rule_store=store)

    suite_result = verifier.run_self_testing_suite(program=ProgramId.SNAP, jurisdiction="EX")
    assert suite_result.all_passed is True
    assert suite_result.composite_score >= 0.90
    assert suite_result.oracle_result.passed is True
    assert suite_result.noop_result.passed is True
    assert suite_result.unsolved_state_result.passed is True


def test_step_level_advantage_optimization_sao_analysis():
    """Verify Step-level Advantage Optimization (SAO) assigns advantages to reasoning trajectories."""
    store = LocalRuleStore()
    provider = LocalRulesProvider(role="verifier")
    verifier = Verifier(provider=provider, rule_store=store)

    prov = Provenance(source_doc_id="doc_test", ingest_method=IngestMethod.STRUCTURED, anonymized=True, content_hash="hash_1")
    evidence = [
        Evidence(evidence_id="ev_inc", type=EvidenceType.MONTHLY_INCOME, value=1100.0, provenance=prov),
        Evidence(evidence_id="ev_hh", type=EvidenceType.HOUSEHOLD_SIZE, value=2.0, provenance=prov),
    ]

    citations = store.all_citations(ProgramId.SNAP, "EX")

    # 1. Grounded, well-cited trajectory -> positive SAO advantage
    good_trajectory = TrajectoryBuffer(
        static_prefix="Grounded Prefix",
        frames=(
            TrajectoryFrame(
                frame_id="f1",
                state=SMState.GATHER,
                agent="gatherer",
                action="Ingested verified income and household size",
                data={"evidence": evidence},
            ),
            TrajectoryFrame(
                frame_id="f2",
                state=SMState.ASSESS,
                agent="proposer",
                action="Evaluated SNAP gross income test against statutory rule",
                data={"citations": [citations[0]]},
            ),
        ),
    )

    sao_good = verifier.analyze_step_advantages(good_trajectory, evidence, "EX", ProgramId.SNAP)
    assert sao_good.is_advantageous is True
    assert sao_good.mean_advantage > 0.3
    assert len(sao_good.critical_defects) == 0
    assert sao_good.grounded_steps_count == 2
    assert sao_good.valid_citation_steps_count == 2

    # 2. Trajectory with ungrounded fact, fake citation, and reasoning monologue -> negative SAO advantage
    fake_ev = Evidence(evidence_id="ev_fake", type=EvidenceType.LIQUID_ASSETS, value=999.0, provenance=prov)
    fake_cit = Citation(citation_id="fake:statute:999", program=ProgramId.SNAP, jurisdiction="EX", source="Fake", title="Fake", text="Fake")

    bad_trajectory = TrajectoryBuffer(
        static_prefix="Bad Prefix",
        frames=(
            TrajectoryFrame(
                frame_id="f1",
                state=SMState.ASSESS,
                agent="proposer",
                action="Asserted ungrounded liquid assets <think>internal speculation</think>",
                data={"evidence": [fake_ev], "citations": [fake_cit]},
            ),
        ),
    )

    sao_bad = verifier.analyze_step_advantages(bad_trajectory, evidence, "EX", ProgramId.SNAP)
    assert sao_bad.is_advantageous is False
    assert sao_bad.mean_advantage < 0.0
    assert len(sao_bad.critical_defects) >= 2


# =========================================================================== #
# Pillar 3: Autoresearch Optimization Ratchet Tests
# =========================================================================== #


def test_autoresearch_ratchet_proposals_and_acceptance_gate():
    """Verify AutoresearchRatchetLoop creates proposals and enforces strict acceptance gating."""
    optimizer = ContinualOptimizer()
    ratchet = AutoresearchRatchetLoop(optimizer=optimizer, time_bound_seconds=120.0, min_accuracy_improvement=0.0)

    # 1. Create prompt mutation proposal
    prompt_prop = ratchet.propose_prompt_mutation(
        target_agent="proposer",
        template="Strict statutory compliance directive with exact citations.",
        description="Enhance citation grounding",
    )
    assert prompt_prop.mutation_type == MutationType.PROMPT_TEMPLATE
    assert prompt_prop.time_limit_sec == 120.0

    # 2. Create tool pruning proposal
    prune_prop = ratchet.propose_tool_pruning_mutation(
        target_program=ProgramId.SNAP,
        active_tools=["ProgrammaticSnapTools"],
        pruned_tools=["ProgrammaticMedicaidTools", "ProgrammaticHousingTools"],
    )
    assert prune_prop.mutation_type == MutationType.TOOL_PRUNING

    # 3. Create routing proposal
    route_prop = ratchet.propose_routing_mutation(
        tier=1,
        primary_model="qwen2.5-distill-legal-7b",
        fallback_model="qwen3.8-27b",
    )
    assert route_prop.mutation_type == MutationType.ROUTING_CONFIG

    # 4. Test Acceptance Gate on passing experiment result
    passing_res = RatchetExperimentResult(
        run_id="run_pass_1",
        proposal_id=prompt_prop.proposal_id,
        passed_gate=True,
        baseline_accuracy=0.92,
        mutated_accuracy=0.96,
        accuracy_delta=0.04,
        cost_usd=0.012,
        latency_ms=450.0,
        duration_seconds=12.5,
        timed_out=False,
        canary_passed=True,
        blocked_egress_count=0,
        reasons=[],
    )
    passed, reasons = ratchet.gate.evaluate(passing_res)
    assert passed is True

    # 5. Test Acceptance Gate rejecting egress security violation or canary failure
    failing_res = RatchetExperimentResult(
        run_id="run_fail_1",
        proposal_id=prompt_prop.proposal_id,
        passed_gate=False,
        baseline_accuracy=0.92,
        mutated_accuracy=0.90,
        accuracy_delta=-0.02,
        cost_usd=0.012,
        latency_ms=450.0,
        duration_seconds=12.5,
        timed_out=False,
        canary_passed=False,
        blocked_egress_count=1,
        reasons=[],
    )
    passed_fail, reasons_fail = ratchet.gate.evaluate(failing_res)
    assert passed_fail is False
    assert any("egress" in r for r in reasons_fail)
    assert any("Canary" in r for r in reasons_fail)
    assert any("accuracy delta" in r for r in reasons_fail)


# =========================================================================== #
# Pillar 4: Deterministic Systems Correctness & Hybrid Sovereign Routing Tests
# =========================================================================== #


def test_bitwise_logprob_parity_reduction():
    """Verify bitwise logprob parity enforcer mitigates floating-point non-associative drift."""
    # Two permutations of the same floating point logprob values (simulating parallel rank reduction order differences)
    logprobs_rank_a = [-0.123456789, -1.987654321, -0.000456789, -2.345678901, -0.000001234]
    logprobs_rank_b = [-0.000001234, -2.345678901, -0.123456789, -0.000456789, -1.987654321]

    # Standard sum in Python may suffer from float ordering discrepancies; math.fsum ensures exact summation
    sum_a = BitwiseParityEnforcer.deterministic_logprob_sum(logprobs_rank_a)
    sum_b = BitwiseParityEnforcer.deterministic_logprob_sum(logprobs_rank_b)
    assert math.isclose(sum_a, sum_b, rel_tol=1e-15, abs_tol=1e-15)

    # Test parity verification
    parity_check = BitwiseParityEnforcer.verify_bitwise_logprob_parity(
        rollout_a=logprobs_rank_a,
        rollout_b=logprobs_rank_a,  # Identical aligned rollout
        epsilon=1e-7,
    )
    assert parity_check["parity_verified"] is True
    assert parity_check["max_drift"] == 0.0


def test_air_gapped_sovereign_hybrid_routing():
    """Verify that ModelRouter enforces data sovereignty by routing sensitive requests to local air-gapped models."""
    router = ModelRouter()

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[],
        required_total=4,
        coverage_complete=False,
        evidence_summary="Confidential applicant PII",
        citations=[],
    )

    # Route with strict Air-Gapped Local constraint
    res = router.route_with_sovereignty(req, DataSovereigntyLevel.AIR_GAPPED_LOCAL)
    assert res is not None
    assert router.stats["sovereignty_enforced_calls"] >= 1
    assert router.stats["air_gapped_calls"] >= 1

    # Verify cost attributions recorded with sovereign level
    cost_attrs = router.get_cost_attributions()
    assert len(cost_attrs) >= 1
    sovereign_attr = next(a for a in cost_attrs if "sovereign" in a.task_intent)
    assert sovereign_attr.sovereignty_level == "air_gapped_local"
    assert sovereign_attr.estimated_cost_usd == 0.0
