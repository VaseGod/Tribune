"""Comprehensive Test Suite for Tribune DeepSeek-V4.1-Flash Roadmap Integration.

Covers:
1. Provider Layer (DeepSeekProvider, Cost Accounting, Generation Parameters, Response Parsing, Router Policy).
2. Preparer Agent (Invariant Prefix Anchoring, Hash Stability, PARSER Scatter-Gather, Token Budgeting).
3. Navigator Agent (Elastic Horizon p90, Rollout Halting, 512-Token Ceiling, Loop Detection, Single-Agent Topology).
4. Verifier Agent (Failure Localization, Single-Turn Delta Rewriting, Retention Protection, ToolGrad Assertions).
5. Metrics & Telemetry (Safe Cache Hit Rate, Cost Calculations, Event Emission).
"""

import pytest

from tribune.agents.navigator import (
    MAX_INTERMEDIATE_TOOL_TOKENS,
    ElasticHorizonController,
    LoopDetector,
    Navigator,
    TrajectoryPersistenceStore,
)
from tribune.agents.preparer import (
    DynamicTailSpec,
    InvariantPrefixSerializer,
    PARSERScatterGatherEngine,
    Preparer,
    StablePrefixSpec,
)
from tribune.agents.verifier import (
    FailureTurnLocator,
    FailureType,
    SingleTurnDeltaRepairContract,
    SingleTurnDeltaRewriter,
    ToolGradAssertionChecker,
)
from tribune.governance.action_gate import ActionGate
from tribune.metrics import get_roadmap_metrics, reset_roadmap_metrics
from tribune.providers.base import get_provider_for_role
from tribune.providers.deepseek import (
    DeepSeekCostCalculator,
    DeepSeekProvider,
    parse_provider_usage,
)
from tribune.providers.router import ModelRouter
from tribune.types import ProgramId

# =========================================================================== #
# 1. PROVIDER LAYER TESTS
# =========================================================================== #


class DummyFailingProvider:
    name = "dummy_failing"
    version = "1.0"

    def complete(self, req):
        raise RuntimeError("API timeout on endpoint")


def test_deepseek_provider_registration():
    """Verify DeepSeekProvider is registered and accessible via factory."""
    provider = get_provider_for_role("deepseek")
    assert isinstance(provider, DeepSeekProvider)
    assert provider.model in ("deepseek-flash", "deepseek-v4.1-flash", "deepseek-v4-pro")


def test_deepseek_cost_calculator_multi_tier_arithmetic():
    """Verify exact multi-tier cost accounting:
    Uncached: $0.30/M, Cached: $0.006/M, Output: $1.20/M.
    """
    # 1,000,000 uncached, 1,000,000 cached, 1,000,000 output
    # Cost = 0.30 + 0.006 + 1.20 = 1.506 USD
    cost = DeepSeekCostCalculator.calculate_cost(
        uncached_input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(1.506, abs=1e-6)

    # Partial calculation
    # 500,000 uncached ($0.15), 500,000 cached ($0.003), 100,000 output ($0.12)
    # Total = 0.15 + 0.003 + 0.12 = 0.273 USD
    cost_partial = DeepSeekCostCalculator.calculate_cost(
        uncached_input_tokens=500_000,
        cached_input_tokens=500_000,
        output_tokens=100_000,
    )
    assert cost_partial == pytest.approx(0.273, abs=1e-6)


def test_deepseek_generation_parameters_defaults():
    """Verify temperature is 1.0 and high reasoning effort is set by default."""
    provider = DeepSeekProvider()
    assert provider.temperature == 1.0
    assert provider.reasoning_effort == "high"


def test_deepseek_usage_parser_handles_multiple_response_formats():
    """Verify parse_provider_usage handles diverse provider response structures."""
    # Standard OpenAI format with prompt_tokens_details
    fmt1 = {
        "prompt_tokens": 10000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 8000},
    }
    p1 = parse_provider_usage(fmt1)
    assert p1.prompt_tokens == 10000
    assert p1.cached_tokens == 8000
    assert p1.uncached_tokens == 2000
    assert p1.completion_tokens == 500
    assert p1.cache_hit_rate == pytest.approx(0.80, abs=1e-4)

    # DeepSeek native prompt_cache_hit_tokens format
    fmt2 = {
        "prompt_tokens": 50000,
        "completion_tokens": 1000,
        "prompt_cache_hit_tokens": 45000,
        "prompt_cache_miss_tokens": 5000,
    }
    p2 = parse_provider_usage(fmt2)
    assert p2.cached_tokens == 45000
    assert p2.uncached_tokens == 5000
    assert p2.cache_hit_rate == pytest.approx(0.90, abs=1e-4)

    # Legacy cached_tokens format
    fmt3 = {
        "prompt_tokens": 2000,
        "completion_tokens": 200,
        "cached_tokens": 1500,
    }
    p3 = parse_provider_usage(fmt3)
    assert p3.cached_tokens == 1500
    assert p3.uncached_tokens == 500


def test_router_sends_ingestion_and_ast_to_deepseek():
    """Verify dual-tier router routes ingestion, AST extraction, and summarization to DeepSeekProvider."""
    router = ModelRouter()
    res_ingest = router.route_codebase_task(task_type="ingestion", prompt="Ingest repository structure")
    assert "deepseek" in res_ingest["model"].lower() or "flash" in res_ingest["model"].lower()
    assert res_ingest["tier"] == 1

    res_ast = router.route_codebase_task(task_type="ast_extraction", prompt="Parse AST nodes")
    assert "deepseek" in res_ast["model"].lower() or "flash" in res_ast["model"].lower()
    assert res_ast["tier"] == 1

    res_summary = router.route_codebase_task(task_type="summarization", prompt="Summarize architecture")
    assert "deepseek" in res_summary["model"].lower() or "flash" in res_summary["model"].lower()
    assert res_summary["tier"] == 1


def test_router_reserves_frontier_models_for_arbitration():
    """Verify high-level arbitration and ambiguous decisions route exclusively to frontier tier."""
    router = ModelRouter()
    decision_arb = router.determine_economic_route("high_level_arbitration")
    assert decision_arb.is_frontier is True
    assert decision_arb.tier == 2
    assert "gpt" in decision_arb.target_model.lower() or "grok" in decision_arb.target_model.lower()

    decision_ambig = router.determine_economic_route("ambiguous_system_decision")
    assert decision_ambig.is_frontier is True
    assert decision_ambig.tier == 2


def test_router_cost_aware_fallback_never_routes_ingestion_to_frontier():
    """Verify router fallback does not silently redirect large ingestion workloads to frontier models."""
    router = ModelRouter(deepseek_provider=DummyFailingProvider())
    res = router.route_codebase_task(
        task_type="ingestion",
        prompt="Parse 100k tokens of source code",
        frontier_fallback_allowed=False,
    )
    # Must fall back to safe secondary provider, not expensive frontier
    assert router.stats["deepseek_fallbacks"] == 1
    assert "frontier" not in res["provider"].lower()


# =========================================================================== #
# 2. PREPARER INVARIANT PREFIX ANCHORING & PARSER TESTS
# =========================================================================== #


def test_invariant_prefix_stability_and_determinism():
    """Verify invariant prefix remains byte-identical across repeated calls with unchanged stable inputs."""
    gate = ActionGate()
    preparer = Preparer(action_gate=gate)

    stable_spec = StablePrefixSpec(
        system_prompt="You are Tribune's public benefits ingestion engine.",
        framework_version="0.1.0",
        repository_file_hierarchy=["tribune/providers/router.py", "tribune/agents/preparer.py", "tribune/config.py"],
        database_schemas={"users": "id INT, name TEXT", "assessments": "id INT, status TEXT"},
        tool_specifications={"tools": ["ingest", "verify"]},
        static_project_constraints=["Enforce zero PII leakage", "Offline deterministic execution"],
        architectural_metadata={"domain": "public_benefits", "security_tier": "air_gapped"},
    )

    # Dynamic inputs differ across invocations
    dyn1 = DynamicTailSpec(user_request="Session 1: Process SNAP claim.", failing_test_output="None")
    dyn2 = DynamicTailSpec(user_request="Session 2: Process Medicaid claim.", failing_test_output="AssertionError: Line 42")

    payload1 = preparer.build_anchored_prompt(stable_spec, dyn1)
    payload2 = preparer.build_anchored_prompt(stable_spec, dyn2)

    # Stable prefix buffers and SHA-256 digests MUST be byte-identical
    assert payload1.stable_prefix == payload2.stable_prefix
    assert payload1.prefix_hash == payload2.prefix_hash

    # Dynamic tails must differ
    assert payload1.dynamic_tail != payload2.dynamic_tail
    assert payload1.full_prompt != payload2.full_prompt


def test_prefix_hash_changes_only_when_stable_content_changes():
    """Verify prefix digest is sensitive exclusively to changes in stable prefix inputs."""
    serializer = InvariantPrefixSerializer()

    base_spec = StablePrefixSpec(
        system_prompt="Base system prompt",
        repository_file_hierarchy=["a.py", "b.py"],
    )
    modified_spec = StablePrefixSpec(
        system_prompt="Base system prompt",
        repository_file_hierarchy=["a.py", "b.py", "c.py"],  # Added file
    )

    p_base = serializer.build_anchored_prompt(base_spec, DynamicTailSpec())
    p_mod = serializer.build_anchored_prompt(modified_spec, DynamicTailSpec())

    assert p_base.prefix_hash != p_mod.prefix_hash


def test_parser_scatter_gather_deterministic_merge():
    """Verify PARSER scatter-gather merges architectural modules in deterministic sorted order."""
    engine = PARSERScatterGatherEngine(max_parallel_workers=2)
    fake_files = [
        "tribune/providers/router.py",
        "tribune/providers/deepseek.py",
        "tribune/agents/preparer.py",
        "tribune/agents/navigator.py",
        "tribune/corpus/rule_store.py",
    ]

    res = engine.execute(fake_files)
    assert len(res.partitions) > 0
    # Modules must appear in alphabetical order in merged context
    keys = [p.partition_key for p in res.partitions]
    assert keys == sorted(keys)
    assert res.speedup_factor_vs_serial >= 1.0


def test_preparer_token_budget_enforcement():
    """Verify budget limits prevent context overflow."""
    serializer = InvariantPrefixSerializer()
    huge_files = [f"path/to/module_{i}/component_{j}.py" for i in range(100) for j in range(10)]
    spec = StablePrefixSpec(
        system_prompt="Budget test",
        repository_file_hierarchy=huge_files,
    )
    payload = serializer.build_anchored_prompt(spec, DynamicTailSpec(), max_prefix_tokens=500)
    assert payload.prefix_tokens_estimate <= 600


# =========================================================================== #
# 3. NAVIGATOR ELASTIC HORIZON & ANTI-VERBOSITY TESTS
# =========================================================================== #


def test_elastic_horizon_p90_and_rollout_halting():
    """Verify Elastic Horizon calculates p90 from history and halts excessive rollouts."""
    store = TrajectoryPersistenceStore()
    # Feed 10 successful trajectories of lengths 5, 6, 7, 8, 9, 10, 11, 12, 13, 20
    for length in [5, 6, 7, 8, 9, 10, 11, 12, 13, 20]:
        store.record(length, success=True)

    controller = ElasticHorizonController(
        minimum_samples=5,
        safety_multiplier=1.25,
        store=store,
    )

    p90 = controller.calculate_p90()
    assert p90 == pytest.approx(13.0, abs=1.0)
    bound = controller.get_current_horizon()
    # ceil(13 * 1.25) = 17
    assert bound == 17

    # Current step within horizon
    is_breached, _, _ = controller.check_bound(current_step=15)
    assert is_breached is False

    # Current step exceeding horizon
    is_breached, bound_val, reason = controller.check_bound(current_step=18)
    assert is_breached is True
    assert "elastic horizon limit" in reason


def test_navigator_512_token_intermediate_thought_ceiling():
    """Verify intermediate tool thoughts respect the 512-token ceiling."""
    nav = Navigator()
    verbose_thought = "This is an unnecessarily wordy explanation. " * 80  # ~320 words, ~400+ chars
    very_verbose = verbose_thought * 10  # ~4000 chars, well over 512 tokens

    step_res = nav.execute_trajectory_step(
        current_step=1,
        intermediate_thought=very_verbose,
        tool_name="verify_document",
        tool_args={"path": "doc.txt"},
    )
    assert step_res["thought_tokens"] <= MAX_INTERMEDIATE_TOOL_TOKENS
    assert nav.telemetry.verbosity_violations == 1


def test_navigator_loop_detection_identifies_repeated_behavior():
    """Verify loop detector flags repeated identical tool calls and no-ops."""
    detector = LoopDetector(max_consecutive_duplicates=2)

    # First call
    loop1, _ = detector.record_and_check("query_rule", {"rule_id": "SNAP_01"}, "Searching...")
    assert loop1 is False

    # Second consecutive identical call: triggers loop detection
    loop2, reason = detector.record_and_check("query_rule", {"rule_id": "SNAP_01"}, "Searching...")
    assert loop2 is True
    assert "Verbosity loop detected" in reason

    # Empty no-op step
    detector.clear()
    loop_noop, noop_reason = detector.record_and_check(None, None, "")
    assert loop_noop is True
    assert "No-op step detected" in noop_reason


def test_navigator_defaults_to_single_agent_topology():
    """Verify single-agent execution topology is enforced by default."""
    from types import SimpleNamespace

    nav = Navigator(enable_subagents=False)
    case = SimpleNamespace(
        case_id="case_101",
        jurisdiction="EX",
        target_programs=[ProgramId.SNAP, ProgramId.MEDICAID],
    )
    dag = nav.plan(case)
    # In unified single-agent mode, tasks must not assign subagents
    for task in dag.tasks.values():
        assert task.subagent_id is None
    assert nav.telemetry.subagent_executions == 0


# =========================================================================== #
# 4. VERIFIER SINGLE-TURN DELTA REWRITE & TOOLGRAD TESTS
# =========================================================================== #


def test_verifier_failure_localization_identifies_exact_turn():
    """Verify failure locator accurately identifies failing turn ID and failure classification."""
    history = [
        {"turn_id": "t1", "status": "success", "content": "Step 1 complete"},
        {"turn_id": "t2", "status": "success", "content": "Step 2 complete"},
        {
            "turn_id": "t3",
            "status": "failure",
            "error_message": "AssertionError: Expected eligibility status likely_eligible but got indeterminate",
            "code_block": "assert assessment.status == EligibilityStatus.LIKELY_ELIGIBLE",
        },
        {"turn_id": "t4", "status": "pending", "content": "Step 4 pending"},
    ]

    report = FailureTurnLocator.locate_failure(history)
    assert report is not None
    assert report.turn_id == "t3"
    assert report.turn_index == 2
    assert report.error_type == FailureType.UNIT_TEST_ASSERTION_FAILURE


def test_verifier_single_turn_delta_rewrite_preserves_surrounding_history():
    """Verify single-turn delta rewrite replaces only the failing turn without full trajectory regeneration."""
    history = [
        {"turn_id": "turn_1", "status": "success", "content": "Initial intake"},
        {"turn_id": "turn_2", "status": "failure", "error": "Schema violation: missing param 'limit'"},
        {"turn_id": "turn_3", "status": "pending", "content": "Finalize adjudication"},
    ]

    repair = SingleTurnDeltaRepairContract(
        turn_id="turn_2",
        repair_type="tool_call",
        original_content="query_documents(user_id=123)",
        replacement_content="query_documents(user_id=123, limit=10)",
        explanation="Added missing required limit argument",
        confidence=0.99,
    )

    rewriter = SingleTurnDeltaRewriter()
    updated, applied = rewriter.apply_delta_repair(history, repair)

    assert applied is True
    assert len(updated) == 3

    # Preceding and subsequent turns must be completely unchanged
    assert updated[0]["turn_id"] == "turn_1"
    assert updated[0]["content"] == "Initial intake"
    assert updated[2]["turn_id"] == "turn_3"
    assert updated[2]["content"] == "Finalize adjudication"

    # Targeted turn repaired
    assert updated[1]["turn_id"] == "turn_2"
    assert updated[1]["repaired"] is True
    assert updated[1]["content"] == "query_documents(user_id=123, limit=10)"
    assert updated[1]["status"] == "success"


def test_toolgrad_assertions_block_schema_violations():
    """Verify ToolGrad blocks tool invocations missing required parameters or invalid types."""
    spec = {
        "required": ["jurisdiction", "income"],
        "properties": {
            "jurisdiction": {"type": "string"},
            "income": {"type": "integer"},
            "category": {"type": "string", "enum": ["earned", "unearned"]},
        },
    }

    # Missing required 'income'
    bad_call_1 = {"jurisdiction": "EX"}
    res1 = ToolGradAssertionChecker.validate_tool_call("assess_income", bad_call_1, tool_spec=spec)
    assert res1.passed is False
    assert any("Missing required parameter 'income'" in v for v in res1.violations)

    # Invalid enum
    bad_call_2 = {"jurisdiction": "EX", "income": 1500, "category": "invalid_category"}
    res2 = ToolGradAssertionChecker.validate_tool_call("assess_income", bad_call_2, tool_spec=spec)
    assert res2.passed is False
    assert any("not in permitted enum values" in v for v in res2.violations)


def test_toolgrad_assertions_block_unsafe_path_traversal():
    """Verify ToolGrad blocks path traversal directory escapes."""
    bad_path_call = {"path": "../../etc/shadow", "data": "leak"}
    res = ToolGradAssertionChecker.validate_tool_call(
        "write_to_file",
        bad_path_call,
        repository_root="/var/app",
    )
    assert res.passed is False
    assert any("Directory traversal escape detected" in v for v in res.violations)


# =========================================================================== #
# 5. METRICS & TELEMETRY TESTS
# =========================================================================== #


def test_metrics_collector_safe_cache_hit_rate_and_cost():
    """Verify cache hit rate division is safe when input tokens are zero, and cost accumulates accurately."""
    reset_roadmap_metrics()
    collector = get_roadmap_metrics()

    # Zero tokens edge case: safe division without ZeroDivisionError
    hit_rate_zero = DeepSeekCostCalculator.compute_cache_hit_rate(cached_input_tokens=0, total_input_tokens=0)
    assert hit_rate_zero == 0.0

    # Record provider call
    collector.record_provider_call(
        provider_name="deepseek:deepseek-flash",
        model_name="deepseek-flash",
        task_type="ingestion",
        input_tokens=10000,
        output_tokens=1000,
        cached_tokens=8000,
    )

    snapshot = collector.get_snapshot()
    assert snapshot.total_input_tokens == 10000
    assert snapshot.cached_input_tokens == 8000
    assert snapshot.uncached_input_tokens == 2000
    assert snapshot.output_tokens == 1000
    assert snapshot.cache_hit_rate == pytest.approx(0.80, abs=1e-4)
    assert snapshot.estimated_cost_usd > 0.0
