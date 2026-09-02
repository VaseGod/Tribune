"""Tests for Speculative Programmatic Tool Calling (sPTC) and Speculator Pipeline."""

import json
import time

import pytest

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.corpus.programs import all_programs
from tribune.orchestration.pipeline import (
    CasePipeline,
    ShadowREPL,
    SpeculativeTask,
    SpeculativeTaskStatus,
    Speculator,
    StreamingTokenSpeculationParser,
)
from tribune.providers.local_rules import (
    SPECULATIVE_TOOLS_REGISTRY,
    cross_evaluate_rule_citations,
    evaluate_statutory_predicate,
    lookup_program_rules,
    spec,
)
from tribune.providers.router import ExecutionTier, ModelRouter, SpeculativeQueueGovernor
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
    task_id2 = speculator.speculate(
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
