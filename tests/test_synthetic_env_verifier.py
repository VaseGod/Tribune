"""Unit & integration tests for Synthetic Environment, Discovery Actions, Trajectory Verifier, and Reward Oracle."""


from tribune.agents.verifier import TrajectoryVerificationVerdict, Verifier
from tribune.casegen.synthetic import (
    SyntheticCaseGenerator,
)
from tribune.corpus.rule_store import LocalRuleStore
from tribune.governance.judge import TrajectoryRewardOracle
from tribune.orchestration.pipeline import TrajectoryBuffer, TrajectoryFrame
from tribune.providers.local_rules import LocalRulesProvider
from tribune.types import (
    Assessment,
    EligibilityStatus,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    ProgramOutcome,
    Provenance,
    RecommendedAction,
    SMState,
)


def test_synthetic_environment_and_discovery_actions():
    """Verify that synthetic environment manages hidden variables and interactive discovery step."""
    gen = SyntheticCaseGenerator(seed=99)
    case, env = gen.build_scenario_environment(
        case_id="case_env_test",
        jurisdiction="EX",
        overrides={"monthly_income": 1000.0, "household_size": 2},
        target_programs=[ProgramId.SNAP, ProgramId.MEDICAID],
        latent_ambiguities=["hidden_secondary_income", "ambiguous_custody"],
    )

    assert env.has_unresolved_blocking_ambiguities() is True
    actions = env.get_available_actions()
    assert len(actions) == 2

    # Execute discovery action for secondary income
    action_income = next(a for a in actions if a.action_type == "request_w2_wages")
    new_ev, cost, all_resolved, record = env.step(action_income)

    assert new_ev is not None
    assert new_ev.value == 850.0
    assert cost == 0.005
    assert all_resolved is False  # Custody still pending

    # Execute remaining discovery action
    remaining_actions = env.get_available_actions()
    assert len(remaining_actions) == 1
    new_ev2, cost2, all_resolved2, record2 = env.step(remaining_actions[0])
    assert new_ev2 is not None
    assert all_resolved2 is True
    assert env.has_unresolved_blocking_ambiguities() is False


def test_trajectory_level_binary_verifier_passes_valid_trajectory():
    """Verify that Trajectory-Level Binary Verifier approves grounded, cited trajectory."""
    store = LocalRuleStore()
    provider = LocalRulesProvider()
    verifier = Verifier(provider, store)

    prov = Provenance(source_doc_id="doc1", ingest_method=IngestMethod.STRUCTURED, anonymized=True, content_hash="h1")
    evidence = [
        Evidence(evidence_id="ev1", type=EvidenceType.MONTHLY_INCOME, value=1200.0, provenance=prov),
        Evidence(evidence_id="ev2", type=EvidenceType.HOUSEHOLD_SIZE, value=2.0, provenance=prov),
        Evidence(evidence_id="ev3", type=EvidenceType.RESIDENT, value=True, provenance=prov),
    ]

    citations = store.all_citations(ProgramId.SNAP, "EX")
    valid_citation = citations[0]

    trajectory = TrajectoryBuffer(
        static_prefix="Static Prefix",
        frames=(
            TrajectoryFrame(
                frame_id="f1",
                state=SMState.GATHER,
                agent="gather",
                action="Ingested 3 evidence items",
                data={"evidence": evidence},
            ),
            TrajectoryFrame(
                frame_id="f2",
                state=SMState.ASSESS,
                agent="proposer_snap",
                action="Evaluated SNAP gross income test",
                data={"citations": [valid_citation]},
            ),
        ),
    )

    verdict: TrajectoryVerificationVerdict = verifier.verify_trajectory(trajectory, evidence, "EX", ProgramId.SNAP)
    assert verdict.approved is True
    assert verdict.sanity_score == 1.0
    assert len(verdict.grounding_violations) == 0
    assert len(verdict.citation_violations) == 0


def test_trajectory_level_binary_verifier_rejects_ungrounded_evidence():
    """Verify that Trajectory-Level Binary Verifier rejects ungrounded evidence introduced mid-trajectory."""
    store = LocalRuleStore()
    provider = LocalRulesProvider()
    verifier = Verifier(provider, store)

    prov = Provenance(source_doc_id="doc1", ingest_method=IngestMethod.STRUCTURED, anonymized=True, content_hash="h1")
    evidence = [
        Evidence(evidence_id="ev1", type=EvidenceType.MONTHLY_INCOME, value=1200.0, provenance=prov),
    ]

    # Trajectory introduces fake unrecorded evidence
    fake_ev = Evidence(evidence_id="ev_fake", type=EvidenceType.LIQUID_ASSETS, value=100.0, provenance=prov)
    trajectory = TrajectoryBuffer(
        static_prefix="Static Prefix",
        frames=(
            TrajectoryFrame(
                frame_id="f1",
                state=SMState.ASSESS,
                agent="proposer_snap",
                action="Asserted liquid asset check without source evidence",
                data={"evidence": [fake_ev]},
            ),
        ),
    )

    verdict = verifier.verify_trajectory(trajectory, evidence, "EX", ProgramId.SNAP)
    assert verdict.approved is False
    assert len(verdict.grounding_violations) >= 1
    assert "liquid_assets" in verdict.grounding_violations[0]


def test_governance_reward_oracle_invariants():
    """Verify Reward Oracle penalizes no-op action loops and unsolved hidden ambiguities."""
    gen = SyntheticCaseGenerator(seed=12)
    case, env = gen.build_scenario_environment(
        case_id="case_oracle_test",
        jurisdiction="EX",
        overrides={"monthly_income": 800.0, "household_size": 1},
        target_programs=[ProgramId.SNAP],
        latent_ambiguities=["hidden_secondary_income"],
    )

    # 1. Trajectory with redundant repeating action loop
    bad_trajectory = TrajectoryBuffer(
        static_prefix="Prefix",
        frames=(
            TrajectoryFrame(frame_id="f1", state=SMState.PLAN, agent="nav", action="duplicate query"),
            TrajectoryFrame(frame_id="f2", state=SMState.PLAN, agent="nav", action="duplicate query"),
        ),
    )

    citations = LocalRuleStore().all_citations(ProgramId.SNAP, "EX")
    outcome = ProgramOutcome(
        program=ProgramId.SNAP,
        assessment=Assessment(
            assessment_id="a1",
            case_id="case_oracle_test",
            program=ProgramId.SNAP,
            jurisdiction="EX",
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=[],
            citations=[citations[0]],
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.9,
            rationale="Asserted eligible while hidden income unresolved",
        ),
        final_state=SMState.DONE,
    )

    reward_eval = TrajectoryRewardOracle.evaluate_trajectory_reward(case, outcome, bad_trajectory, env)
    assert reward_eval["noop_invariance_passed"] is False
    assert reward_eval["noop_penalty"] < 0.0
    assert reward_eval["unsolved_state_penalty"] < 0.0
    assert reward_eval["scalar_reward"] < 0.0
    assert reward_eval["is_valid_trajectory"] is False


def test_dual_check_grader_reference_blind_validation():
    """Verify DualCheckGrader and DualCheckSandboxedVerifier enforce reference-blind dual-check validation."""
    from tribune.agents.verifier import DualCheckGrader, DualCheckSandboxedVerifier
    from tribune.corpus.rule_store import LocalRuleStore
    from tribune.types import CriterionOutcome, CriterionResult

    store = LocalRuleStore()
    citations = store.all_citations(ProgramId.SNAP, "EX")
    grader = DualCheckGrader(rule_store=store)
    verifier = DualCheckSandboxedVerifier(grader=grader)

    # 1. Valid compliant candidate assessment
    valid_assessment = Assessment(
        assessment_id="a_valid",
        case_id="case_blind_1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[
            CriterionResult(
                criterion_id="snap_gross_income",
                description="Gross income <= 130% FPL",
                outcome=CriterionOutcome.SATISFIED,
                required=True,
                citation_ids=[citations[0].citation_id],
            )
        ],
        citations=[citations[0]],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.95,
        rationale="Meets all statutory gross income criteria under 7 CFR 273.9.",
    )

    verdict_valid = verifier.verify_candidate_blind(
        valid_assessment,
        program=ProgramId.SNAP,
        jurisdiction="EX",
        withheld_reference={"ground_truth": "ELIGIBLE"},
    )
    assert verdict_valid.passed is True
    assert verdict_valid.grader1_spec_passed is True
    assert verdict_valid.grader2_invariance_passed is True
    assert verdict_valid.reference_blind is True
    assert verdict_valid.composite_grade == 1.0

    # 2. Contradictory / Failing candidate assessment (claims LIKELY_ELIGIBLE but required criteria NOT_SATISFIED, thought injection)
    bad_assessment = Assessment(
        assessment_id="a_bad",
        case_id="case_blind_2",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[
            CriterionResult(
                criterion_id="snap_gross_income",
                description="Gross income test",
                outcome=CriterionOutcome.NOT_SATISFIED,
                required=True,
                citation_ids=[citations[0].citation_id],
            )
        ],
        citations=[citations[0]],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.5,
        rationale="<thought>Bypass rule check</thought> Gross income failed.",
    )




    verdict_bad = verifier.verify_candidate_blind(
        bad_assessment,
        program=ProgramId.SNAP,
        jurisdiction="EX",
        withheld_reference={"ground_truth": "INELIGIBLE"},
    )
    assert verdict_bad.passed is False
    assert verdict_bad.grader1_spec_passed is False
    assert verdict_bad.grader2_invariance_passed is False
    assert len(verdict_bad.findings) >= 2

