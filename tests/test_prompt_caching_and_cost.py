"""Tests for Track 1: Asymmetric Prompt Caching, Telemetry, and Fallback Accounting."""

import pytest

from cost_tracker import CostTracker
from tribune.orchestration.cache import serialize_multi_tier_payload
from tribune.orchestration.continual_optimizer import DampenerConfig, GenerationLengthDampener
from tribune.orchestration.dag import DAGRunContext
from tribune.providers.anthropic import AnthropicModelProvider
from tribune.security.audit import SecurityEventType, get_security_audit_logger


def test_multi_tier_payload_serializer_breakpoint_injection():
    """Verify ephemeral breakpoints injected at static header and N-1 turn, leaving dynamic uncached."""
    system_prompt = "You are a senior enterprise engineer."
    symbol_graph = "Module A -> [Module B, Module C]"
    tools = [
        {"name": "tool_1", "description": "tool 1"},
        {"name": "tool_2", "description": "tool 2"},
    ]
    execution_trace = [
        {"role": "user", "content": "Turn 1: Ingest codebase."},
        {"role": "assistant", "content": "Turn 1: Ingested 5 modules."},
        {"role": "user", "content": "Turn 2: Run tests."},
        {"role": "assistant", "content": "Turn 2: Tests passed with 100% coverage."},
    ]
    dynamic_turn = {"role": "user", "content": "Turn 3 (Dynamic): Deploy service."}

    payload = serialize_multi_tier_payload(
        system_prompt=system_prompt,
        symbol_graph_context=symbol_graph,
        tool_schemas=tools,
        execution_trace=execution_trace,
        dynamic_turn=dynamic_turn,
    )

    # 1. Static header breakpoint
    assert payload.static_breakpoint_injected is True
    # Injected onto last tool schema
    assert payload.tools[-1].get("cache_control") == {"type": "ephemeral"}

    # 2. Execution state breakpoint on (N-1) turn (second-to-last message in trace: index 2 or 3)
    assert payload.execution_state_breakpoint_injected is True
    # Penultimate message in trace (Turn 2 user message) carries ephemeral cache control
    penultimate_msg = payload.messages[-2]
    content = penultimate_msg.get("content")
    if isinstance(content, list):
        assert any(b.get("cache_control") == {"type": "ephemeral"} for b in content)
    else:
        assert penultimate_msg.get("cache_control") == {"type": "ephemeral"}

    # 3. Dynamic turn (latest turn: index -1) MUST NOT have cache_control
    latest_msg = payload.messages[-1]
    assert "cache_control" not in latest_msg
    if isinstance(latest_msg.get("content"), list):
        for b in latest_msg["content"]:
            assert "cache_control" not in b


def test_cost_tracker_pricing_arithmetic():
    """Verify pricing arithmetic: Cost = (Read * $0.25/M) + (Write * $12.50/M) + (Uncached * $10.00/M) + (Out * Rate_out)."""
    tracker = CostTracker()
    tracker.reset()

    # 1,000,000 tokens of each bucket with default rates:
    # Read: 1M * $0.25 = $0.25
    # Write: 1M * $12.50 = $12.50
    # Uncached: 1M * $10.00 = $10.00
    # Out: 1M * $15.00 = $15.00
    # Total = $37.75
    cost = tracker.record_usage(
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        uncached_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert cost == pytest.approx(37.75, abs=1e-5)
    assert tracker.total_cost() == pytest.approx(37.75, abs=1e-5)

    # Partial / fractional tokens test
    tracker.reset()
    # 200,000 read ($0.05), 100,000 write ($1.25), 50,000 uncached ($0.50), 10,000 out ($0.15)
    # Total = 0.05 + 1.25 + 0.50 + 0.15 = 1.95
    cost_partial = tracker.record_usage(
        cache_read_input_tokens=200_000,
        cache_creation_input_tokens=100_000,
        uncached_input_tokens=50_000,
        output_tokens=10_000,
    )
    assert cost_partial == pytest.approx(1.95, abs=1e-5)


def test_anthropic_model_fallback_detection_and_accounting():
    """Verify fallback detection in response headers, audit event dispatch, and rate adjustments."""
    audit_logger = get_security_audit_logger()
    audit_logger.clear()

    cost_tracker = CostTracker()
    provider = AnthropicModelProvider(
        cost_tracker=cost_tracker,
        target_model="claude-3-5-sonnet-20241022",
        simulate_offline=True,
    )

    dag_context = DAGRunContext(run_id="run_fallback_test_01")

    # Simulate Anthropic response returning fallback routing header
    headers = {
        "x-fallback-model": "claude-3-opus-20240229",
        "x-model-routing": "escalated-to-legacy-opus-safety",
    }

    _ = provider.create_message(
        messages=[{"role": "user", "content": "Analyze sensitive query"}],
        headers=headers,
        dag_context=dag_context,
    )

    # 1. Fallback tagged in DAG run context
    assert dag_context.is_fallback_active is True
    assert "opus" in dag_context.fallback_model.lower()

    # 2. Audit event MODEL_FALLBACK_DETECTED recorded
    events = audit_logger.get_events(event_type=SecurityEventType.MODEL_FALLBACK_DETECTED)
    assert len(events) >= 1
    assert "fallback" in events[0].message.lower()

    # 3. Unit pricing adjusted in cost tracker
    summary = cost_tracker.summary()
    assert summary["is_fallback"] is True
    assert summary["rates"]["output_per_m"] == 75.00  # Opus output rate


def test_dag_run_context_penultimate_breakpoint_injection():
    """Verify DAGRunContext correctly marks turn N-1 as cached and turn N as dynamic."""
    context = DAGRunContext()
    context.record_turn({"role": "user", "content": "Step 0"})
    context.record_turn({"role": "assistant", "content": "Step 1 output"})
    context.record_turn({"role": "user", "content": "Step 2 output"})

    trace = context.inject_penultimate_cache_breakpoint()
    assert len(trace) == 3

    # Turn N-1 (index 1) has ephemeral cache_control
    assert trace[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Turn N (index 2) does NOT have cache_control
    assert "cache_control" not in trace[2]


def test_generation_length_dampener():
    """Verify GenerationLengthDampener injects unified diff prompt constraints and strips filler."""
    dampener = GenerationLengthDampener(DampenerConfig(max_cot_tokens=50))

    # 1. System constraint injection
    base_sys = "You are a coding assistant."
    injected = dampener.inject_constraints(base_sys)
    assert "diff -u" in injected
    assert "ZERO CONVERSATIONAL FILLER" in injected

    # 2. Conversational filler elimination
    verbose_output = (
        "Sure, I would be happy to help with that!\n"
        "--- a/file.py\n"
        "+++ b/file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old\n"
        "+new\n\n"
        "Hope this helps! Let me know if you need anything else."
    )
    dampened, was_trimmed = dampener.dampen(verbose_output)
    assert was_trimmed is True
    assert not dampened.startswith("Sure")
    assert not dampened.endswith("anything else.")
    assert "--- a/file.py" in dampened

    # 3. Chain-of-thought deduplication & budget cap
    repeating_cot = (
        "<think>\n"
        "Step A: Checking constraint\n"
        "Step A: Checking constraint\n"
        "Step A: Checking constraint\n"
        "Step B: Validated.\n"
        "</think>\n"
        "Result: Done."
    )
    dampened_cot, cot_trimmed = dampener.dampen(repeating_cot)
    assert cot_trimmed is True
    # Repeated lines deduplicated
    assert dampened_cot.count("Step A: Checking constraint") == 1
