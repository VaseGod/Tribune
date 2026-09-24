"""Unit Tests for Dual-Agent Trajectory Cost Model.

Validates:
- Trajectory cost formula: Cost = sum_t [ C_action(t) + I_eval(t) * C_mem(t) ]
- Indicator function behavior: auxiliary cost is strictly zero when I_eval(t) = 0
- Non-empty vs empty trajectory handling
- Pricing overrides for action and auxiliary models
- Auxiliary overhead ratio and avoided failure loop savings calculation
- Net trajectory efficiency score
"""

from __future__ import annotations

import pytest

from tribune.eval.costmodel import (
    CostModel,
    DualAgentTrajectoryCost,
    TrajectoryCostModel,
    default_cost_model,
)


def test_dual_agent_cost_formula_and_indicator_function():
    """Verify that auxiliary model cost is accounted for if and only if I_eval(t) == 1."""
    traj_model = TrajectoryCostModel()

    # 4 turns: Action model generates 1000 input, 200 output per turn.
    action_in = [1000, 1000, 1000, 1000]
    action_out = [200, 200, 200, 200]
    # Memory sidecar invoked only on turn 2 and turn 4 (k=2)
    i_eval = [0, 1, 0, 1]
    mem_in = [0, 200, 0, 200]
    mem_out = [0, 50, 0, 50]

    # Rates:
    # Action: $0.002 / 1k in, $0.004 / 1k out
    # Memory: $0.0005 / 1k in, $0.001 / 1k out
    cost_res = traj_model.compute_dual_agent_trajectory(
        action_turns_input_tokens=action_in,
        action_turns_output_tokens=action_out,
        memory_eval_indicators=i_eval,
        memory_turns_input_tokens=mem_in,
        memory_turns_output_tokens=mem_out,
        action_price_per_1k_input=0.002,
        action_price_per_1k_output=0.004,
        memory_price_per_1k_input=0.0005,
        memory_price_per_1k_output=0.0010,
        avoided_loop_count=2,
    )

    # Manual calculation:
    # Action per turn: (1000 * 0.002 + 200 * 0.004) / 1000 = (2.0 + 0.8) / 1000 = 0.0028
    # Action total (4 turns) = 4 * 0.0028 = 0.0112
    assert cost_res.action_input_tokens == 4000
    assert cost_res.action_output_tokens == 800
    assert abs(cost_res.action_estimated_cost - 0.0112) < 1e-6

    # Memory per turn (when I_eval=1): (200 * 0.0005 + 50 * 0.001) / 1000 = (0.1 + 0.05) / 1000 = 0.00015
    # Memory total (2 invocations) = 2 * 0.00015 = 0.00030
    assert cost_res.memory_invocation_count == 2
    assert cost_res.memory_input_tokens == 400
    assert cost_res.memory_output_tokens == 100
    assert abs(cost_res.memory_estimated_cost - 0.0003) < 1e-6

    # Total trajectory cost: 0.0112 + 0.0003 = 0.0115
    assert abs(cost_res.total_trajectory_cost - 0.0115) < 1e-6

    # Overhead ratio: 0.0003 / 0.0112 ~= 0.0268 (~2.7% overhead)
    assert 0.02 < cost_res.auxiliary_overhead_ratio < 0.03

    # Avoided loop savings: 2 avoided loops * avg_turn_cost (0.0028) * 1.5 = 0.0084
    assert abs(cost_res.avoided_failure_loop_savings_estimate - 0.0084) < 1e-5


def test_zero_memory_invocations():
    """Verify that when sidecar is disabled (all I_eval == 0), memory cost is exactly 0."""
    traj_model = TrajectoryCostModel()
    action_in = [500, 500]
    action_out = [100, 100]
    i_eval = [0, 0]

    cost_res = traj_model.compute_dual_agent_trajectory(
        action_turns_input_tokens=action_in,
        action_turns_output_tokens=action_out,
        memory_eval_indicators=i_eval,
        action_price_per_1k_input=0.001,
        action_price_per_1k_output=0.002,
    )
    assert cost_res.memory_invocation_count == 0
    assert cost_res.memory_estimated_cost == 0.0
    assert cost_res.total_trajectory_cost == cost_res.action_estimated_cost
    assert cost_res.auxiliary_overhead_ratio == 0.0


def test_empty_trajectory_handling():
    """Verify clean handling of 0-turn trajectories."""
    traj_model = TrajectoryCostModel()
    res = traj_model.compute_dual_agent_trajectory(
        action_turns_input_tokens=[],
        action_turns_output_tokens=[],
        memory_eval_indicators=[],
    )
    assert res.total_turns == 0
    assert res.total_trajectory_cost == 0.0
    assert res.auxiliary_overhead_ratio == 0.0
    assert res.net_trajectory_efficiency_score == 1.0


def test_serialization_and_to_dict():
    """Verify to_dict outputs all required fields with proper formatting."""
    traj_model = TrajectoryCostModel()
    res = traj_model.compute_dual_agent_trajectory(
        action_turns_input_tokens=[100],
        action_turns_output_tokens=[50],
        memory_eval_indicators=[1],
    )
    data = res.to_dict()
    assert "total_trajectory_cost" in data
    assert "auxiliary_overhead_ratio" in data
    assert "avoided_failure_loop_savings_estimate" in data
    assert "net_trajectory_efficiency_score" in data
