"""Tests for the harness loop execution engine."""

from tribune.harness.context import ContextCompactor
from tribune.harness.loop import HarnessLoop
from tribune.harness.state import HarnessState, RunStatus, StepType
from tribune.harness.tools import ToolDefinition, ToolDispatcher
from tribune.inference.base import InferenceResponse
from tribune.inference.registry import MockOfflineProvider


def test_harness_loop_basic_execution():
    provider = MockOfflineProvider(canned_response="Administrative form completed.")
    loop = HarnessLoop(provider=provider, max_steps=4)

    state = loop.run(
        task_id="task_001",
        initial_messages=[{"role": "user", "content": "Draft intake form."}],
    )

    assert state.status == RunStatus.COMPLETED
    assert state.step_count > 0
    assert len(state.steps) > 0
    assert state.steps[0].step_type == StepType.INFERENCE
    assert state.steps[0].outcome == "SUCCESS"


def test_harness_loop_with_tool_invocation():
    dispatcher = ToolDispatcher()
    tool_called = False

    def dummy_tool(query: str):
        nonlocal tool_called
        tool_called = True
        return {"status": "ok", "match": query}

    dispatcher.register_tool(
        ToolDefinition(
            name="search_statute",
            description="Search statute rules",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=dummy_tool,
        )
    )

    class ToolCallingProvider(MockOfflineProvider):
        def __init__(self):
            super().__init__()
            self.first = True

        def complete(self, request):
            if self.first:
                self.first = False
                return InferenceResponse(
                    text="",
                    tool_calls=[{
                        "id": "call_1",
                        "function": {"name": "search_statute", "arguments": {"query": "SNAP"}},
                    }],
                    model="mock",
                    provider_id="mock",
                )
            return InferenceResponse(text="Found statute.", model="mock", provider_id="mock")

    loop = HarnessLoop(provider=ToolCallingProvider(), tool_dispatcher=dispatcher, max_steps=4)
    state = loop.run(task_id="task_tool_01", initial_messages=[{"role": "user", "content": "Search SNAP"}])

    assert tool_called is True
    assert state.status == RunStatus.COMPLETED
    # Should have INFERENCE -> TOOL -> INFERENCE
    step_types = [s.step_type for s in state.steps]
    assert StepType.INFERENCE in step_types
    assert StepType.TOOL in step_types


def test_harness_loop_kill_switch():
    provider = MockOfflineProvider()
    loop = HarnessLoop(provider=provider, max_steps=10)

    state = HarnessState(task_id="task_kill_test")
    state.request_kill("Safety containment breach")

    result_state = loop.run(
        task_id="task_kill_test",
        initial_messages=[{"role": "user", "content": "Continue run"}],
        existing_state=state,
    )

    assert result_state.status == RunStatus.KILLED
    assert result_state.kill_requested is True


def test_harness_loop_checkpoint_and_resumption():
    state = HarnessState(task_id="task_resume_01")
    state.variables["applicant_income"] = 1200.0
    state.cumulative_cost_usd = 0.05
    state.status = RunStatus.RUNNING

    snapshot = state.checkpoint()
    restored = HarnessState.restore(snapshot)

    assert restored.task_id == "task_resume_01"
    assert restored.variables["applicant_income"] == 1200.0
    assert restored.cumulative_cost_usd == 0.05
    assert restored.status == RunStatus.RUNNING


def test_context_compaction_preserves_verifier_failures():
    compactor = ContextCompactor(max_context_tokens=50, intermediate_thought_ceiling_chars=50)

    messages = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Here is a very long text with lots of words " * 20},
        {"role": "user", "content": "=== [VERIFIER_GATE_REJECTION] ===\nFAILED: Malformed Citation\n================================="},
        {"role": "assistant", "content": "Final turn answer."},
    ]

    compacted = compactor.compact(messages)

    # Assert system prompt preserved
    assert any(m.get("role") == "system" for m in compacted)
    # Assert verifier rejection preserved
    assert any("[VERIFIER_GATE_REJECTION]" in str(m.get("content", "")) for m in compacted)


def test_context_staged_tool_output_elision():
    """Verify bulky middle tool outputs are deterministically elided to short stubs."""
    compactor = ContextCompactor(
        max_context_tokens=16_000,
        preserve_recent_turns=2,
        tool_output_ceiling_chars=500,
    )

    bulky_tool_data = "SECTION 201 STATUTE: " + ("Rule paragraph text. " * 100)  # ~2,200 chars
    recent_tool_data = "SHORT STATUTE: " + ("Short rule. " * 50)  # ~600 chars

    messages = [
        {"role": "system", "content": "You are a legal assistant."},
        # Older turns (idx 0 to 2 in conversation_msgs)
        {"role": "user", "content": "Search for Section 201."},
        {"role": "assistant", "content": "Invoking search_statute tool."},
        {
            "role": "tool",
            "name": "search_statute",
            "tool_call_id": "call_statute_1",
            "content": bulky_tool_data,
        },
        # Recent turns (last 2 conversation_msgs)
        {"role": "assistant", "content": "I found Section 201. Now checking recent rule."},
        {
            "role": "tool",
            "name": "search_recent",
            "tool_call_id": "call_statute_2",
            "content": recent_tool_data,
        },
    ]

    compacted = compactor.compact(messages)

    # 1. System prompt preserved
    assert compacted[0]["role"] == "system"

    # 2. Find the elided middle tool message
    tool_msgs = [m for m in compacted if m.get("role") == "tool"]
    assert len(tool_msgs) == 2

    older_tool = tool_msgs[0]
    recent_tool = tool_msgs[1]

    # Older bulky tool output MUST be elided into deterministic stub
    assert "[TOOL OUTPUT ELIDED: name=search_statute, tool_call_id=call_statute_1" in older_tool["content"]
    assert older_tool["name"] == "search_statute"
    assert older_tool["tool_call_id"] == "call_statute_1"
    assert len(older_tool["content"]) < 200

    # Recent tool output MUST be preserved intact
    assert recent_tool["content"] == recent_tool_data
    assert recent_tool["name"] == "search_recent"

    # Token/character savings check
    orig_chars = sum(len(str(m.get("content", ""))) for m in messages)
    compacted_chars = sum(len(str(m.get("content", ""))) for m in compacted)
    reduction_pct = (orig_chars - compacted_chars) / orig_chars
    assert reduction_pct > 0.50  # Over 50% character/token reduction


def test_staged_tool_elision_respects_critical_anchors():
    """Verify tool messages containing critical verifier markers are never elided."""
    compactor = ContextCompactor(
        max_context_tokens=16_000,
        preserve_recent_turns=1,
        tool_output_ceiling_chars=100,
    )

    critical_tool_data = (
        "=== [VERIFIER_GATE_REJECTION] ===\n"
        "GATE STATUS: FAILED\n"
        + ("Detailed statutory violation description. " * 30)
    )

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Verify submission."},
        {
            "role": "tool",
            "name": "audit_tool",
            "tool_call_id": "call_crit_1",
            "content": critical_tool_data,
        },
        {"role": "assistant", "content": "Recent response."},
    ]

    compacted = compactor.compact(messages)

    tool_msg = next(m for m in compacted if m.get("role") == "tool")
    # Must preserve the exact verifier rejection content despite being older and bulky
    assert "[VERIFIER_GATE_REJECTION]" in tool_msg["content"]
    assert tool_msg["content"] == critical_tool_data


def test_harness_loop_with_staged_tool_output_elision_integration():
    """Integration test: HarnessLoop executes multi-turn trajectory with tool elision."""
    dispatcher = ToolDispatcher()
    bulky_result = "STATUTE CORPUS DATA: " + ("Extremely long legal text segment. " * 200)

    def mock_bulk_search(query: str):
        return bulky_result

    dispatcher.register_tool(
        ToolDefinition(
            name="bulk_search",
            description="Returns bulky statute data",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=mock_bulk_search,
        )
    )

    call_count = 0

    class MultiTurnProvider(MockOfflineProvider):
        def complete(self, request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return InferenceResponse(
                    text="",
                    tool_calls=[{
                        "id": "call_bulk_1",
                        "function": {"name": "bulk_search", "arguments": {"query": "SNAP eligibility"}},
                    }],
                    model="mock",
                    provider_id="mock",
                )
            elif call_count == 2:
                # Issue second non-tool response, leading to more turns
                return InferenceResponse(
                    text="Parsed bulky data. Generating additional questions.",
                    model="mock",
                    provider_id="mock",
                )
            return InferenceResponse(text="Assessment finalized.", model="mock", provider_id="mock")

    # Compactor with preserve_recent_turns=1 and tool_output_ceiling_chars=500
    compactor = ContextCompactor(
        max_context_tokens=16_000,
        preserve_recent_turns=1,
        tool_output_ceiling_chars=500,
    )

    loop = HarnessLoop(
        provider=MultiTurnProvider(),
        tool_dispatcher=dispatcher,
        compactor=compactor,
        max_steps=6,
    )

    state = loop.run(
        task_id="task_staged_elision_integration",
        initial_messages=[{"role": "user", "content": "Evaluate SNAP"}],
    )

    assert state.status == RunStatus.COMPLETED
    assert state.step_count >= 2
    # Verify execution finished cleanly with zero errors
    assert state.failure_reason is None

