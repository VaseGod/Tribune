"""Tests for the harness loop execution engine."""

import pytest
from tribune.harness.context import ContextCompactor
from tribune.harness.loop import HarnessLoop
from tribune.harness.policies import HarnessPolicyEnforcer
from tribune.harness.state import HarnessState, RunStatus, StepType
from tribune.harness.tools import ToolDefinition, ToolDispatcher
from tribune.inference.base import InferenceRequest, InferenceResponse, ProviderUsage
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
