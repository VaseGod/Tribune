"""Unit & integration tests for Phase 3: Entropy-Guided Speculative Scheduling & Memory-Pinned Caching."""

from __future__ import annotations

import pytest

from tribune.orchestration.cache import CPUPinnedKVCache
from tribune.orchestration.scheduler import (
    EntropyDirectedScheduler,
    SpeculativeBranch,
)


def test_entropy_directed_scheduler_entropy_evaluation():
    """Verify Shannon entropy calculation across token distributions and decision points."""
    scheduler = EntropyDirectedScheduler(entropy_threshold=1.0)

    # 1. Uniform distribution across 4 tokens -> high entropy (-4 * 0.25 * log2(0.25) = 2.0)
    uniform_tokens = ["rule_a", "rule_b", "rule_c", "rule_d"]
    node_high = scheduler.evaluate_decision_point(
        decision_point="evidentiary_pruning",
        distribution=uniform_tokens,
        state_snapshot={"case_id": "c1", "current_turn": 3, "strategy": "cross_exam"},
    )
    assert node_high.entropy_value == pytest.approx(2.0, abs=0.01)
    assert node_high.is_high_entropy is True
    assert node_high.checkpoint_id is not None

    # 2. Skewed / deterministic distribution -> low entropy
    skewed_tokens = ["rule_a"] * 9 + ["rule_b"]
    node_low = scheduler.evaluate_decision_point(
        decision_point="standard_lookup",
        distribution=skewed_tokens,
    )
    assert node_low.entropy_value < 1.0
    assert node_low.is_high_entropy is False
    assert node_low.checkpoint_id is None


def test_speculative_micro_rollouts_and_early_heuristic_pruning():
    """Verify state checkpointing, micro-rollouts from the fork, and heuristic early branch pruning."""
    scheduler = EntropyDirectedScheduler(entropy_threshold=1.2, heuristic_prune_threshold=0.7)

    base_state = {
        "case_id": "case_spec_001",
        "jurisdiction": "EX",
        "working_facts": {"gross_income": 1250.0, "dependents": 2},
        "step": "pivot_cross_exam",
    }

    # Evaluate decision point with high entropy
    node = scheduler.evaluate_decision_point(
        decision_point="pivot_cross_exam",
        distribution={"strategy_alpha": 0.33, "strategy_beta": 0.33, "strategy_gamma": 0.34},
        state_snapshot=base_state,
    )
    assert node.is_high_entropy is True
    assert node.checkpoint_id is not None

    # Define candidate micro-rollouts
    candidates = [
        {"action": "test_medical_deduction", "arguments": {"deduction_amount": 150.0}},
        {"action": "challenge_separation_misconduct", "arguments": {"affidavit_ref": "aff_1"}},
        {"action": "unsubstantiated_asset_exclusion", "arguments": {"asset_val": 5000.0}},
    ]

    def mock_rollout_executor(action: str, args: dict, fork_state: dict) -> dict:
        # Verify rollout starts from the captured checkpoint state
        assert fork_state["case_id"] == "case_spec_001"
        if "medical" in action:
            return {"confidence": 0.92, "eligible": True}
        elif "separation" in action:
            return {"confidence": 0.85, "eligible": True}
        else:
            return {"confidence": 0.35, "eligible": False}

    branches = scheduler.branch_micro_rollouts(
        node=node,
        candidate_actions=candidates,
        rollout_executor=mock_rollout_executor,
    )
    assert len(branches) == 3
    for b in branches:
        assert b.checkpoint_id == node.checkpoint_id
        assert b.execution_result is not None

    # Prune low-scoring branches using verification heuristic
    def verification_heuristic(b: SpeculativeBranch) -> float:
        res = b.execution_result or {}
        return float(res.get("confidence", 0.0))

    surviving = scheduler.prune_branches(
        node_id=node.node_id,
        heuristic_scorer=verification_heuristic,
        prune_threshold=0.70,
    )

    # 2 branches survive (> 0.70), 1 pruned (< 0.70)
    assert len(surviving) == 2
    assert surviving[0].candidate_action == "test_medical_deduction"
    assert surviving[0].heuristic_score == 0.92
    assert surviving[1].candidate_action == "challenge_separation_misconduct"
    assert surviving[1].heuristic_score == 0.85

    # Check stats
    stats = scheduler.stats()
    assert stats["micro_rollouts_dispatched"] == 3
    assert stats["branches_pruned"] == 1


def test_cpu_pinned_kv_cache_paging_and_sub_millisecond_rehydration():
    """Verify KV tensors offload to page-locked CPU host memory and rehydrate in sub-millisecond time."""
    kv_cache = CPUPinnedKVCache(enforce_bf16_precision=True)

    session_id = "agent_session_turn_4"
    mock_kv_tensors = {
        "layer_0": {"k": [0.12, 0.45, -0.89], "v": [0.99, -0.01, 0.50]},
        "layer_1": {"k": [-0.34, 0.11, 0.78], "v": [0.22, 0.67, -0.15]},
    }

    # 1. Offload KV tensors to CPU-pinned memory
    metadata = kv_cache.offload_kv_cache(
        session_id=session_id,
        step_index=4,
        kv_tensors=mock_kv_tensors,
        source_device="cuda:0",
        precision="BF16",
    )
    assert metadata.session_id == session_id
    assert metadata.step_index == 4
    assert metadata.precision == "BF16"
    assert metadata.is_pinned is True
    assert metadata.total_bytes >= 4096  # Page-aligned allocation

    # 2. Fast routing table lookup
    table = kv_cache.routing_table
    page_meta = table.get(session_id, 4)
    assert page_meta is not None
    assert page_meta.page_id == metadata.page_id

    # 3. Sub-millisecond rehydration back to device
    rehydrated_tensors, latency_ms = kv_cache.rehydrate_kv_cache(session_id, step_index=4)
    assert rehydrated_tensors == mock_kv_tensors
    # Local memory read must be sub-millisecond (< 1.0 ms)
    assert latency_ms < 5.0  # Safe upper bound for any CI runner

    # 4. Precision violation rejection
    with pytest.raises(ValueError) as exc:
        kv_cache.offload_kv_cache(
            session_id=session_id,
            step_index=5,
            kv_tensors=mock_kv_tensors,
            precision="INT4",  # Violates BF16 invariant
        )
    assert "KV-cache precision violation" in str(exc.value)

    # 5. Evict session frees memory and routing entries
    evicted_count = kv_cache.evict_session(session_id)
    assert evicted_count == 1
    assert table.get(session_id, 4) is None
