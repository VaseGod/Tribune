"""Tests for Speculative Programmatic Tool Calling (sPTC) and Speculator Pipeline."""

import time

import numpy as np
import pytest

import tribune.providers.local_rules  # noqa: F401 - register speculative tools
from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.orchestration.mtp import (
    EntropyAwareDepthScaler,
    NativeMTPBackbone,
)
from tribune.orchestration.pipeline import (
    CasePipeline,
    ShadowREPL,
    Speculator,
    StreamingTokenSpeculationParser,
)
from tribune.providers.local_rules import SPECULATIVE_TOOLS_REGISTRY
from tribune.providers.router import (
    ExecutionTier,
    ModelRouter,
    SpeculativeInferenceRunner,
    SpeculativeQueueGovernor,
)
from tribune.types import ProgramId


def test_speculative_tool_registration():
    """Verify @spec.tool decorator registers pure functions in SPECULATIVE_TOOLS_REGISTRY."""
    assert "evaluate_statutory_predicate" in SPECULATIVE_TOOLS_REGISTRY
    assert "lookup_program_rules" in SPECULATIVE_TOOLS_REGISTRY
    assert "cross_evaluate_rule_citations" in SPECULATIVE_TOOLS_REGISTRY

    meta = SPECULATIVE_TOOLS_REGISTRY["evaluate_statutory_predicate"]
    assert meta["speculatable"] is True
    assert meta["pure"] is True
    assert meta["ttl"] > 0


def test_shadow_repl_isolation_and_rollback():
    """Verify ShadowREPL executes pure operations in isolation and allows snapshot rollback."""
    repl = ShadowREPL(sandbox_id="test_shadow_repl")
    repl.set("x", 10)
    assert repl.get("x") == 10

    snap_id = repl.snapshot()
    repl.set("x", 20)
    assert repl.get("x") == 20

    # Rollback to snapshot
    restored = repl.rollback_to_snapshot(snap_id)
    assert restored["x"] == 10
    assert repl.get("x") == 10

    # Execute pure function
    res = repl.execute_pure(lambda a, b: a + b, 5, 7)
    assert res == 12

    repl.prune()
    assert repl.get("x") is None


def test_streaming_speculation_token_parser():
    """Verify StreamingTokenSpeculationParser extracts candidate tool calls from partial streams."""
    parser = StreamingTokenSpeculationParser()

    # Ingest partial tokens
    parser.feed_token('{"tool": "lookup_program_rules", "arguments": {"program": "snap"')
    calls = parser.parse_candidate_calls()
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "lookup_program_rules"
    assert calls[0]["arguments"]["program"] == "snap"

    parser.reset()
    parser.feed_token('evaluate_statutory_predicate(program="medicaid", age=45)')
    ast_calls = parser.parse_candidate_calls()
    assert len(ast_calls) == 1
    assert ast_calls[0]["tool_name"] == "evaluate_statutory_predicate"


def test_speculator_hit_miss_and_cancellation():
    """Verify Speculator dispatches background tasks, records hits, and cancels divergent branches."""
    speculator = Speculator(max_workers=2)

    # 1. Dispatch speculative lookup
    task_id = speculator.speculate(
        tool_name="lookup_program_rules",
        arguments={"program": "snap", "jurisdiction": "EX"},
        jurisdiction="EX",
    )
    assert task_id is not None

    # Wait briefly for execution
    time.sleep(0.05)

    # Consume hit
    result = speculator.consume_hit(
        tool_name="lookup_program_rules",
        arguments={"program": "snap", "jurisdiction": "EX"},
    )
    assert result is not None
    assert result["program"] == "snap"
    assert len(result["rules"]) > 0

    # Verify stats
    stats = speculator.stats()
    assert stats["dispatched"] >= 1
    assert stats["hits"] == 1

    # 2. Test miss
    miss_res = speculator.consume_hit(
        tool_name="lookup_program_rules",
        arguments={"program": "housing", "jurisdiction": "NY"},
    )
    assert miss_res is None
    assert speculator.stats()["misses"] == 1

    # 3. Test branch cancellation
    _task_id2 = speculator.speculate(
        tool_name="evaluate_statutory_predicate",
        arguments={"program": "snap", "income": 50000},
    )
    speculator.cancel_divergent_branches(active_candidate_keys=[])
    speculator.prune_all()


def test_pipeline_speculative_telemetry():
    """Verify CasePipeline runs with speculative execution and tracks telemetry metrics."""
    generator = SyntheticCaseGenerator(seed=42)
    case = generator.build_case(
        case_id="spec_case_01",
        jurisdiction="EX",
        overrides={"monthly_income": 900.0, "household_size": 2, "age": 35},
        target_programs=[ProgramId.SNAP, ProgramId.MEDICAID],
    )

    pipeline = CasePipeline()
    result = pipeline.run_case(case)

    # Validate speculative telemetry recorded in result
    assert result.speculative_dispatched >= 2
    assert result.speculative_hits >= 1
    assert result.speculative_latency_saved_ms >= 0.0


def test_queue_governor_and_execution_tiers():
    """Verify SpeculativeQueueGovernor throttles excessive tasks and ModelRouter respects tiers."""
    governor = SpeculativeQueueGovernor(max_queue_depth=3, cost_budget_usd=0.002)

    assert not governor.should_throttle_speculation()
    governor.record_speculative_dispatch(estimated_cost=0.001)
    governor.record_speculative_dispatch(estimated_cost=0.0008)

    # Next call exceeds budget
    assert governor.should_throttle_speculation(estimated_cost=0.0005)
    assert governor.throttled_count >= 1

    router = ModelRouter()
    tier, provider, reason = router.route_execution_tier(
        task_intent="snap_eligibility", is_speculative=False
    )
    assert tier == ExecutionTier.TIER_2_FRONTIER
    assert "primary_frontier_root" in reason

    spec_tier, spec_provider, spec_reason = router.route_execution_tier(
        task_intent="speculative_predicate", is_speculative=True
    )
    assert spec_tier in (ExecutionTier.TIER_0_LOCAL_DENSE, ExecutionTier.TIER_1_ROUTINE_FAST)


def test_native_mtp_prediction_heads_and_shared_memory():
    """Verify Native MTP integrates prediction heads into backbone, projects K in [1, 4]
    in shared tensor memory, and deprecates the dual-model draft/target runner."""
    # 1. Verify deprecation warning on dual-model SpeculativeInferenceRunner
    with pytest.deprecated_call():
        _ = SpeculativeInferenceRunner(
            draft_provider=None,  # type: ignore
            target_provider=None,  # type: ignore
        )

    # 2. Native MTP backbone execution across depth K in [1, 4] within shared memory
    backbone = NativeMTPBackbone(vocab_size=1000, hidden_dim=256, max_speculative_depth=4)
    hidden_state = np.ones(256, dtype=np.float32)

    # Project candidates in shared tensor memory
    branch = backbone.project_candidates(
        current_sequence=[10, 20, 30],
        hidden_state=hidden_state,
        content_hint="json",
    )

    # Assert depth K is within [1, 4]
    assert 1 <= branch.depth_k <= 4
    assert len(branch.candidate_tokens) == branch.depth_k
    assert branch.logits.shape == (branch.depth_k, 1000)

    # Assert shared memory slice was allocated with zero IPC serialization
    assert branch.shared_memory_offset >= 0
    # Data is directly accessible in shared memory
    assert np.array_equal(
        backbone.shared_memory.token_ids[
            branch.shared_memory_offset : branch.shared_memory_offset + branch.depth_k
        ],
        branch.candidate_tokens,
    )


def test_dynamic_entropy_aware_depth_scaling():
    """Verify dynamic entropy-aware depth scaling:
    - Low entropy (structured code, JSON schemas) expands to K = 4 (~180+ tok/s).
    - High entropy (ambiguous prose, branching natural language) throttles to K <= 1
      or falls back to single-token autoregression."""
    scaler = EntropyAwareDepthScaler(tau_low=0.85, tau_high=1.75, max_depth=4, min_depth=1)

    # 1. Low Entropy distribution (sharp/peaked, e.g. JSON schema / structured tokens)
    sharp_dist = np.zeros(100, dtype=np.float32)
    sharp_dist[42] = 0.98
    sharp_dist[43] = 0.02
    k_low, h_low, telem_low = scaler.scale_depth(sharp_dist, content_hint="json")

    assert h_low < 0.85
    assert k_low == 4
    assert telem_low["is_low_entropy"] is True
    assert telem_low["estimated_tok_per_sec"] >= 180.0

    # 2. High Entropy distribution (uniform, ambiguous prose / branching language)
    uniform_dist = np.ones(20, dtype=np.float32) / 20.0
    k_high, h_high, telem_high = scaler.scale_depth(uniform_dist)

    assert h_high >= 1.75
    assert k_high <= 1
    assert telem_high["is_high_entropy"] is True
    assert telem_high["is_autoregressive_fallback"] is True

    # 3. Intermediate Entropy
    mid_dist = np.zeros(100, dtype=np.float32)
    mid_dist[1] = 0.6
    mid_dist[2] = 0.3
    mid_dist[3] = 0.1
    k_mid, h_mid, telem_mid = scaler.scale_depth(mid_dist)
    assert 0.85 <= h_mid < 1.75
    assert 1 <= k_mid <= 4


def test_non_stalling_asynchronous_rollback():
    """Verify non-stalling rollback pattern:
    - Eliminates synchronous H2D verification barriers upon rejected draft tokens.
    - Pipelines tree-mask verification and candidate projections concurrently."""
    backbone = NativeMTPBackbone(vocab_size=500, hidden_dim=128, max_speculative_depth=4)
    hidden_state = np.zeros(128, dtype=np.float32)

    # Simulate verifier where token at index 2 is rejected
    # Index 0 accepted, Index 1 accepted, Index 2 rejected -> m = 2 accepted out of K = 4
    def mock_verifier(step_idx: int, token: int) -> bool:
        return step_idx < 2

    step_result = backbone.execute_speculative_step(
        current_sequence=[5, 15, 25],
        hidden_state=hidden_state,
        target_token_verifier=mock_verifier,
        content_hint="json",
    )

    assert step_result["depth_k"] == 4
    assert step_result["accepted_tokens_count"] == 2
    assert step_result["acceptance_rate"] == 0.5
    assert step_result["non_stalling_rollback"] is True

    # Check non-stalling pipeline stats
    stats = backbone.rollback_pipeline.stats()
    assert stats["accepted_tokens"] == 2
    assert stats["rejected_tokens"] >= 2
    assert stats["synchronous_stalls_avoided"] >= 1
    assert stats["h2d_barriers_eliminated"] >= 1
