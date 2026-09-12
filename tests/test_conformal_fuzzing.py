"""Phase 3: Property-Based Conformal Fuzzing via Hypothesis.

Evaluates finite-sample empirical false-alarm guarantees (alpha_emp <= alpha + epsilon_n)
under synthetic distribution shift, verifies inline step 3/4 CRC early-exit interception,
asserts atomic state rollbacks, and validates telemetry accounting.
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tribune.casegen.conformal import (
    ConformalCalibrationResult,
    ConformalCalibrator,
    ConformalScoreType,
)
from tribune.casegen.simulation import (
    SimulationEngine,
    SimulationState,
    SimulationTurn,
    TrajectoryOutcome,
)
from tribune.casegen.world_model import (
    CourtroomWorldModel,
    StatutoryWorldModel,
)
from tribune.clients.routing import (
    RoutingTier,
    TieredRoutingGateway,
)
from tribune.instrumentation.usage import UsageRecorder


# =========================================================================== #
# Hypothesis Strategies for Legal States & Adversarial Perturbations
# =========================================================================== #

@st.composite
def st_valid_legal_state(draw: Any) -> dict[str, Any]:
    """Generates valid legal states strictly compliant with statutory limits."""
    hh_size = draw(st.integers(min_value=1, max_value=6))
    fpl_limit = (1255.0 + (hh_size - 1) * 438.0) * 1.30
    # Safe income: 30% to 85% of statutory cutoff
    monthly_income = draw(st.floats(min_value=300.0, max_value=fpl_limit * 0.85))
    liquid_assets = draw(st.floats(min_value=0.0, max_value=2200.0))
    days_denial = draw(st.integers(min_value=5, max_value=75))

    return {
        "household_size": hh_size,
        "monthly_income": round(monthly_income, 2),
        "liquid_assets": round(liquid_assets, 2),
        "days_since_denial": days_denial,
        "has_contradiction": False,
        "evidence_offered": True,
        "evidence_marked": True,
    }


@st.composite
def st_edge_case_legal_state(draw: Any) -> dict[str, Any]:
    """Generates boundary legal states within 1-2% of statutory thresholds."""
    hh_size = draw(st.integers(min_value=1, max_value=4))
    fpl_limit = (1255.0 + (hh_size - 1) * 438.0) * 1.30
    # Boundary income: 98% to 99.8% of cutoff
    ratio = draw(st.floats(min_value=0.98, max_value=0.998))
    monthly_income = fpl_limit * ratio
    liquid_assets = draw(st.floats(min_value=2500.0, max_value=2745.0))
    days_denial = draw(st.integers(min_value=86, max_value=89))

    return {
        "household_size": hh_size,
        "monthly_income": round(monthly_income, 2),
        "liquid_assets": round(liquid_assets, 2),
        "days_since_denial": days_denial,
        "has_contradiction": False,
        "evidence_offered": True,
        "evidence_marked": True,
    }


@st.composite
def st_contradictory_legal_state(draw: Any) -> dict[str, Any]:
    """Generates invalid states with contradictory or out-of-bounds facts."""
    hh_size = draw(st.integers(min_value=1, max_value=4))
    fpl_limit = (1255.0 + (hh_size - 1) * 438.0) * 1.30

    breach_type = draw(st.sampled_from(["over_income", "time_barred", "asset_breach", "contradiction"]))

    if breach_type == "over_income":
        income = fpl_limit * draw(st.floats(min_value=1.35, max_value=2.50))
        days = draw(st.integers(min_value=10, max_value=60))
        assets = 500.0
        contra = False
    elif breach_type == "time_barred":
        income = fpl_limit * 0.50
        days = draw(st.integers(min_value=95, max_value=180))  # Past 90 days
        assets = 500.0
        contra = False
    elif breach_type == "asset_breach":
        income = fpl_limit * 0.50
        days = 30
        assets = draw(st.floats(min_value=3500.0, max_value=8000.0))  # Over $2,750
        contra = False
    else:
        income = fpl_limit * 0.50
        days = 30
        assets = 500.0
        contra = True

    return {
        "household_size": hh_size,
        "monthly_income": round(income, 2),
        "liquid_assets": round(assets, 2),
        "days_since_denial": days,
        "has_contradiction": contra,
        "evidence_offered": True,
        "evidence_marked": True,
        "breach_type": breach_type,
    }


# =========================================================================== #
# Property Tests
# =========================================================================== #

def test_conformal_calibrator_mathematical_bounds():
    """Verify finite-sample quantile calculation and Hoeffding statistical padding."""
    calibrator = ConformalCalibrator(alpha=0.10, delta=0.05)
    n = 500
    padding = calibrator.compute_padding(n, delta=0.05)
    # epsilon_500 = sqrt(ln(2 / 0.05) / (2 * 500)) = sqrt(ln(40) / 1000) ~= 0.06073
    expected_padding = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n))
    assert math.isclose(padding, expected_padding, rel_tol=1e-5)
    assert 0.055 < padding < 0.065

    # Test calibration on synthetic scores
    synthetic_scores = [0.90 + (i % 10) * 0.01 for i in range(500)]
    res = calibrator.calibrate(synthetic_scores, alpha=0.10, delta=0.05)

    assert isinstance(res, ConformalCalibrationResult)
    assert res.n_samples == 500
    assert res.alpha == 0.10
    assert res.delta == 0.05
    assert res.padded_threshold <= res.raw_threshold
    assert res.certified_upper_bound == pytest.approx(0.10 + padding, rel=1e-4)


def test_finite_sample_empirical_false_alarm_guarantee():
    """Evaluate 1,000 perturbed runs and assert empirical false-alarm rate <= alpha + epsilon_n."""
    n_cal = 500
    alpha = 0.10
    delta = 0.05
    world_model = StatutoryWorldModel(jurisdiction="EX")
    calibrator = ConformalCalibrator(alpha=alpha, delta=delta)

    # 1. Calibration phase on n = 500 nominal historical traces
    calibration_scores = calibrator.generate_calibration_traces(n=n_cal, world_model=world_model, seed=123)
    cal_res = calibrator.calibrate(calibration_scores, alpha=alpha, delta=delta)
    padded_cutoff = cal_res.padded_threshold
    epsilon_n = cal_res.epsilon_n

    # 2. Evaluation phase on N = 1,000 perturbed valid scenarios
    N = 1000
    rng = random.Random(42)
    false_alarms = 0

    for _ in range(N):
        hh = rng.choice([1, 2, 3, 4])
        limit = world_model.get_fpl_monthly_limit(hh)
        # Valid income with realistic operational perturbations (noise)
        income = limit * rng.uniform(0.35, 0.94) + rng.uniform(-5.0, 5.0)
        days = rng.randint(5, 82)
        assets = rng.uniform(50.0, 2200.0)

        mock_state = {"household_size": hh, "monthly_income": income, "liquid_assets": assets}
        mock_turn = {
            "action": "provide_stipulation",
            "household_size": hh,
            "monthly_income": income,
            "liquid_assets": assets,
            "days_since_denial": days,
        }

        score = world_model.score_transition(mock_state, mock_turn)
        # Check if flagged as fault (score below padded threshold)
        if not calibrator.is_valid(score, threshold=padded_cutoff):
            false_alarms += 1

    alpha_emp = false_alarms / N
    certified_bound = alpha + epsilon_n

    # Assert invariant: alpha_emp <= alpha + epsilon_n
    assert alpha_emp <= certified_bound, (
        f"Empirical false-alarm rate {alpha_emp:.4f} exceeded certified bound {certified_bound:.4f} "
        f"(alpha={alpha}, epsilon_n={epsilon_n:.4f})"
    )


@given(st_contradictory_legal_state())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_simulation_inline_early_termination_and_token_savings(contradictory_state: dict[str, Any]):
    """Assert broken premises trigger CRC fault at step 3 or 4, saving ~65% tokens with clean rollback."""
    world_model = StatutoryWorldModel(jurisdiction="EX")
    calibrator = ConformalCalibrator(alpha=0.10, delta=0.05)
    traces = calibrator.generate_calibration_traces(n=500, world_model=world_model, seed=99)
    calibrator.calibrate(traces)

    recorder = UsageRecorder()
    max_turns = 12
    tokens_per_turn = 500

    engine = SimulationEngine(
        world_model=world_model,
        calibrator=calibrator,
        usage_recorder=recorder,
        max_turns=max_turns,
        tokens_per_turn=tokens_per_turn,
    )

    # Construct a 12-turn trajectory where broken premise is injected at step 3
    initial_state = SimulationState(
        case_id="case_crc_test_001",
        initial_facts={"household_size": 2, "monthly_income": 1200.0, "liquid_assets": 400.0},
    )

    # Turns 1 and 2 are fully valid
    turns = [
        SimulationTurn(step=1, action="intake_interview", statement="Confirming identity", data={"step": 1}),
        SimulationTurn(step=2, action="document_submission", statement="Submitting wage stubs", data={"step": 2}),
        # Step 3 injects the statutory contradiction
        SimulationTurn(
            step=3,
            action="cross_examination_contradiction",
            statement="Introducing contested evidence",
            data=contradictory_state,
        ),
    ]
    # Remaining steps 4-12 should never execute
    for s in range(4, 13):
        turns.append(
            SimulationTurn(step=s, action=f"subsequent_action_{s}", statement=f"Step {s}", data={"step": s})
        )

    outcome: TrajectoryOutcome = engine.run_trajectory(initial_state, turns)

    # 1. Verification: Interrupted early at step 3 (or 4 if deferred)
    assert outcome.interrupted is True
    assert outcome.early_exit_step in (3, 4)
    assert "Interrupted (CRC Fault at Step 3)" in outcome.status
    assert outcome.turns_executed == 3

    # 2. Token Savings: burned ~35% on faulted turn, preserved remaining 9 turns
    assert outcome.tokens_burned < 2000  # Burned ~1,175 tokens vs 6,000 max
    expected_saved = (max_turns - 3) * tokens_per_turn  # 9 * 500 = 4,500
    assert outcome.tokens_saved == expected_saved

    # 3. Clean Rollback: Uncommitted mutations from step 3 were discarded
    assert initial_state.step == 2
    assert len(initial_state.committed_turns) == 2
    assert initial_state.committed_turns[-1].action == "document_submission"
    # The contradictory state was not persisted to verified facts
    if "breach_type" in contradictory_state:
        assert "breach_type" not in initial_state.facts

    # 4. Usage Recorder Telemetry
    task = recorder.finish_task()
    assert task is not None
    assert task.early_exit_step == 3
    assert task.tokens_saved_estimate == 4500
    assert len(task.crc_breach_events) >= 1
    breach_ev = task.crc_breach_events[0]
    assert breach_ev["early_exit_step"] == 3
    assert breach_ev["tokens_saved_estimate"] == 4500
    # Verify ISO-8601 timestamp presence
    assert "T" in breach_ev["timestamp"]


def test_tiered_routing_gateway_telemetry_and_moe_experts():
    """Verify TieredRoutingGateway routes to GLM-5.3-Flash with active_experts=8 and prompt caching."""
    recorder = UsageRecorder()
    gateway = TieredRoutingGateway(usage_recorder=recorder, offline_mode=True)

    # 1. Bulk Scenario Generation routed to MoE (GLM-5.3-Flash)
    variations = gateway.generate_bulk_scenarios(
        base_scenario_prompt="SNAP applicant with seasonal construction wages in jurisdiction EX",
        n_variations=4,
        target_program="snap",
    )
    assert len(variations) == 4
    for var in variations:
        assert var["model"] == "glm-5.3-flash"
        assert var["active_experts"] == 8
        assert "counterfactual_constraint" in var
        # High precision timestamp
        datetime.fromisoformat(var["timestamp"])

    # 2. Statutory Invariant Verification routed to high-factuality process verifier
    state = {"household_size": 1, "monthly_income": 800.0}
    turn = {"household_size": 1, "monthly_income": 950.0, "days_since_denial": 45}
    res_valid = gateway.verify_statutory_invariant(state, turn)
    assert res_valid["is_valid"] is True
    assert res_valid["score"] >= 0.90
    assert res_valid["model"] == "qwen3.8-27b"

    # 3. Test verification failure on statutory over-income
    turn_invalid = {"household_size": 1, "monthly_income": 3500.0}
    res_invalid = gateway.verify_statutory_invariant(state, turn_invalid)
    assert res_invalid["is_valid"] is False
    assert res_invalid["score"] <= 0.30

    # 4. Telemetry assertions
    task = recorder.finish_task()
    assert task is not None
    assert len(task.calls) >= 3
    moe_calls = [c for c in task.calls if c.model == "glm-5.3-flash"]
    assert len(moe_calls) >= 1
    assert moe_calls[0].active_experts == 8
    assert moe_calls[0].cache_read_tokens > 0
    assert task.cache_hit_ratio > 0.0


def test_simulation_full_12_turn_clean_trajectory():
    """Verify clean 12-turn trajectory completion when all statutory invariants hold."""
    world_model = StatutoryWorldModel(jurisdiction="EX")
    calibrator = ConformalCalibrator(alpha=0.10, delta=0.05)
    traces = calibrator.generate_calibration_traces(n=500, world_model=world_model, seed=77)
    calibrator.calibrate(traces)

    recorder = UsageRecorder()
    engine = SimulationEngine(
        world_model=world_model,
        calibrator=calibrator,
        usage_recorder=recorder,
        max_turns=12,
        tokens_per_turn=500,
    )

    initial_state = SimulationState(
        case_id="case_clean_12",
        initial_facts={"household_size": 1, "monthly_income": 800.0, "liquid_assets": 300.0},
    )

    # 12 fully compliant turns
    turns = [
        SimulationTurn(
            step=i + 1,
            action=f"legal_step_{i + 1}",
            statement=f"Submitting verified step {i + 1}",
            data={"household_size": 1, "monthly_income": 800.0, "days_since_denial": 20 + i},
        )
        for i in range(12)
    ]

    outcome = engine.run_trajectory(initial_state, turns)

    assert outcome.status == "Completed"
    assert outcome.interrupted is False
    assert outcome.turns_executed == 12
    assert outcome.tokens_burned == 12 * 500
    assert outcome.tokens_saved == 0
    assert outcome.early_exit_step is None
    assert len(initial_state.committed_turns) == 12
    assert initial_state.step == 12


def test_simulation_step_4_interception_token_accounting():
    """Verify simulation engine halts at step 4 when premise breach occurs at turn 4."""
    world_model = StatutoryWorldModel(jurisdiction="EX")
    calibrator = ConformalCalibrator(alpha=0.10, delta=0.05)
    traces = calibrator.generate_calibration_traces(n=500, world_model=world_model, seed=88)
    calibrator.calibrate(traces)

    recorder = UsageRecorder()
    engine = SimulationEngine(
        world_model=world_model,
        calibrator=calibrator,
        usage_recorder=recorder,
        max_turns=12,
        tokens_per_turn=500,
    )

    initial_state = SimulationState(
        case_id="case_step_4",
        initial_facts={"household_size": 2, "monthly_income": 1000.0},
    )

    # 3 valid turns, 4th has statutory breach (over 90-day appeal deadline)
    turns = [
        SimulationTurn(step=1, action="intake", data={"days_since_denial": 30}),
        SimulationTurn(step=2, action="wages", data={"days_since_denial": 35}),
        SimulationTurn(step=3, action="assets", data={"days_since_denial": 40}),
        SimulationTurn(step=4, action="contested_appeal", data={"days_since_denial": 120}),  # Past 90 days!
        SimulationTurn(step=5, action="unreached_5", data={"step": 5}),
        SimulationTurn(step=6, action="unreached_6", data={"step": 6}),
    ]

    outcome = engine.run_trajectory(initial_state, turns)
    assert outcome.interrupted is True
    assert outcome.early_exit_step == 4
    assert outcome.turns_executed == 4
    assert "Interrupted (CRC Fault at Step 4)" in outcome.status

    # Spared tokens = (12 - 4) * 500 = 4,000 tokens
    assert outcome.tokens_saved == 4000
    assert initial_state.step == 3
    assert len(initial_state.committed_turns) == 3


def test_conformal_calibrator_non_conformity_mode():
    """Verify ConformalCalibrator operates symmetrically in non-conformity mode."""
    calibrator = ConformalCalibrator(
        alpha=0.10,
        delta=0.05,
        score_type=ConformalScoreType.NON_CONFORMITY,
    )

    # Non-conformity scores: 0.0 is perfect, 1.0 is total invariant violation
    scores = [round(0.02 + (i % 20) * 0.005, 4) for i in range(500)]
    res = calibrator.calibrate(scores)

    assert res.score_type == ConformalScoreType.NON_CONFORMITY
    assert res.padded_threshold >= res.raw_threshold
    assert calibrator.is_valid(0.01) is True  # Low violation -> valid
    assert calibrator.is_valid(0.99) is False  # High violation -> invalid


def test_courtroom_world_model_score_transition_and_crc_interception():
    """Verify CourtroomWorldModel integrates with SimulationEngine for procedural exhibit admission."""
    courtroom = CourtroomWorldModel()
    ex = courtroom.mark_exhibit("doc_w2", "Exhibit A", "W2 Form")

    # Step 1: Offer exhibit (valid) -> score >= 0.90
    score_offer = courtroom.score_transition(None, {"action": "offer", "exhibit_id": ex.exhibit_id})
    assert score_offer >= 0.90

    # Step 2: Try to admit unoffered exhibit -> score <= 0.10
    score_bad = courtroom.score_transition(None, {"action": "admit", "exhibit_id": "non_existent_id"})
    assert score_bad <= 0.10

