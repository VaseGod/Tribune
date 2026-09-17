"""Harness Execution Loop.

Decoupled three-tier execution engine separating:
1. Inference dispatch
2. Tool invocation
3. Verifier execution
4. State tracking & checkpoints
5. Context compaction
6. Cryptographic trace emission
7. Failure handling without recursive retry loops
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..inference.base import InferenceProvider, InferenceRequest, ProviderError
from .context import ContextCompactor
from .policies import HarnessPolicyEnforcer
from .state import HarnessState, RunStatus, StepRecord, StepType
from .tools import ToolDispatcher, ToolExecutionResult

logger = logging.getLogger(__name__)


class HarnessLoop:
    """Core execution engine managing agent trajectories with deterministic verification."""

    def __init__(
        self,
        provider: InferenceProvider,
        tool_dispatcher: ToolDispatcher | None = None,
        verifier_callback: Callable[[HarnessState, Any], dict[str, Any]] | None = None,
        compactor: ContextCompactor | None = None,
        policy_enforcer: HarnessPolicyEnforcer | None = None,
        trace_emitter: Callable[[StepRecord, HarnessState], None] | None = None,
        max_steps: int = 16,
        max_error_retries_per_step: int = 2,
    ) -> None:
        self.provider = provider
        self.tool_dispatcher = tool_dispatcher or ToolDispatcher()
        self.verifier_callback = verifier_callback
        self.compactor = compactor or ContextCompactor()
        self.policy_enforcer = policy_enforcer or HarnessPolicyEnforcer()
        self.trace_emitter = trace_emitter
        self.max_steps = max_steps
        self.max_error_retries_per_step = max_error_retries_per_step

    def run(
        self,
        task_id: str,
        initial_messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        initial_variables: dict[str, Any] | None = None,
        existing_state: HarnessState | None = None,
    ) -> HarnessState:
        """Execute the harness trajectory to completion, verifier gate, or stop signal."""
        # Initialize or resume state
        if existing_state:
            state = existing_state
            state.status = RunStatus.RUNNING
        else:
            state = HarnessState(
                task_id=task_id,
                max_steps=self.max_steps,
                context_messages=list(initial_messages),
                variables=dict(initial_variables or {}),
                status=RunStatus.RUNNING,
            )

        if system_prompt and not any(m.get("role") == "system" for m in state.context_messages):
            state.context_messages.insert(0, {"role": "system", "content": system_prompt})

        logger.info(f"[HarnessLoop] Starting task '{task_id}' (run_id={state.run_id})")

        while state.step_count < state.max_steps:
            current_step_num = state.step_count + 1

            # --------------------------------------------------------------- #
            # 1. Stop & Kill Signal Evaluation
            # --------------------------------------------------------------- #
            if state.kill_requested:
                state.status = RunStatus.KILLED
                self._record_step(
                    state=state,
                    step_type=StepType.SYSTEM,
                    outcome="BLOCKED",
                    error="Execution killed by kill switch",
                    metadata={"reason": state.failure_reason},
                )
                break

            if state.stop_requested:
                state.status = RunStatus.INTERRUPTED
                self._record_step(
                    state=state,
                    step_type=StepType.SYSTEM,
                    outcome="SUCCESS",
                    metadata={"reason": state.failure_reason or "Graceful stop requested"},
                )
                break

            # --------------------------------------------------------------- #
            # 2. Pre-Inference Policy Gate
            # --------------------------------------------------------------- #
            policy_res = self.policy_enforcer.evaluate_step(state, StepType.INFERENCE, {})
            if not policy_res.allowed:
                state.status = RunStatus.FAILED
                state.failure_reason = policy_res.violation_reason
                self._record_step(
                    state=state,
                    step_type=StepType.POLICY_CHECK,
                    outcome="BLOCKED",
                    error=policy_res.violation_reason,
                    metadata={"policy": policy_res.policy_name, **policy_res.details},
                )
                break

            # --------------------------------------------------------------- #
            # 3. Context Compaction
            # --------------------------------------------------------------- #
            state.context_messages = self.compactor.compact(state.context_messages)

            # --------------------------------------------------------------- #
            # 4. Inference Dispatch (Bounded, Non-Recursive)
            # --------------------------------------------------------------- #
            inference_start = time.perf_counter()
            tool_schemas = self.tool_dispatcher.get_tool_schemas()
            request = InferenceRequest(
                messages=state.context_messages,
                tools=tool_schemas if tool_schemas else [],
                timeout_s=30.0,
            )

            try:
                # Bounded retries: strict limit, never recurses into model loops
                response = self.provider.execute_with_bounded_retries(
                    request, max_retries=self.max_error_retries_per_step
                )
            except ProviderError as exc:
                duration_ms = (time.perf_counter() - inference_start) * 1000.0
                state.status = RunStatus.FAILED
                state.failure_reason = f"Provider error: {exc}"
                self._record_step(
                    state=state,
                    step_type=StepType.INFERENCE,
                    provider=self.provider.provider_id,
                    duration_ms=duration_ms,
                    outcome="FAILED",
                    error=str(exc),
                )
                break

            inference_dur_ms = (time.perf_counter() - inference_start) * 1000.0
            self._record_step(
                state=state,
                step_type=StepType.INFERENCE,
                provider=self.provider.provider_id,
                model=response.model,
                duration_ms=inference_dur_ms,
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                tokens_cached=response.usage.cached_tokens,
                cost_usd=response.cost_usd,
                outcome="SUCCESS",
                output_data={"text": response.text, "tool_calls_count": len(response.tool_calls)},
            )

            # Append assistant message to context
            state.context_messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": response.tool_calls if response.tool_calls else None,
            })

            # --------------------------------------------------------------- #
            # 5. Tool Invocations
            # --------------------------------------------------------------- #
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            import json

                            tool_args = json.loads(tool_args)
                        except Exception:
                            tool_args = {"raw": tool_args}

                    # Check policy for tool invocation
                    tool_policy = self.policy_enforcer.evaluate_step(
                        state, StepType.TOOL, {"tool_name": tool_name, "arguments": tool_args}
                    )
                    if not tool_policy.allowed:
                        self._record_step(
                            state=state,
                            step_type=StepType.TOOL,
                            outcome="BLOCKED",
                            error=tool_policy.violation_reason,
                            input_data={"tool": tool_name, "arguments": tool_args},
                            metadata={"policy": tool_policy.policy_name},
                        )
                        # Feed tool error back to context
                        state.context_messages.append({
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call.get("id", f"call_{current_step_num}"),
                            "content": f"BLOCKED BY POLICY: {tool_policy.violation_reason}",
                        })
                        continue

                    # Execute tool safely
                    tool_res: ToolExecutionResult = self.tool_dispatcher.execute(tool_name, tool_args)
                    self._record_step(
                        state=state,
                        step_type=StepType.TOOL,
                        duration_ms=tool_res.duration_ms,
                        outcome="SUCCESS" if tool_res.success else "FAILED",
                        input_data={"tool": tool_name, "arguments": tool_args},
                        output_data={"result": tool_res.result_data},
                        error=tool_res.error_message,
                    )
                    # Feed tool result back to context
                    state.context_messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call.get("id", f"call_{current_step_num}"),
                        "content": str(tool_res.result_data if tool_res.success else tool_res.error_message),
                    })

            # --------------------------------------------------------------- #
            # 6. Verifier Execution Gate
            # --------------------------------------------------------------- #
            if self.verifier_callback is not None:
                verifier_start = time.perf_counter()
                try:
                    ver_res = self.verifier_callback(state, response)
                except Exception as exc:
                    ver_res = {
                        "is_valid": False,
                        "status": "VERIFIER_EXCEPTION",
                        "failure_reason": f"Verifier exception: {exc}",
                        "remediation_hint": "Inspect verifier configuration",
                    }

                verifier_dur = (time.perf_counter() - verifier_start) * 1000.0
                is_valid = ver_res.get("is_valid", True)
                gate_status = "SUCCESS" if is_valid else "FAILED"

                self._record_step(
                    state=state,
                    step_type=StepType.VERIFIER,
                    duration_ms=verifier_dur,
                    outcome=gate_status,
                    output_data=ver_res,
                    error=ver_res.get("failure_reason") if not is_valid else None,
                )

                if not is_valid:
                    # Deterministic verifier gate failure: immediately halt or inject feedback
                    mandatory_fail_fast = ver_res.get("fail_fast", True)
                    if mandatory_fail_fast:
                        state.status = RunStatus.FAILED
                        state.failure_reason = ver_res.get("failure_reason", "Verifier gate failed")
                        logger.warning(
                            f"[HarnessLoop] Mandatory verifier gate failed: {state.failure_reason}. Halting loop."
                        )
                        break
                    else:
                        # Non-fatal: Inject structured feedback into context
                        state.context_messages = self.compactor.inject_verifier_feedback(
                            state.context_messages, ver_res
                        )

            # If model produced completion without tool calls and verifier passed: completed
            if not response.tool_calls:
                state.status = RunStatus.COMPLETED
                break

        if state.status == RunStatus.RUNNING:
            state.status = RunStatus.TIMED_OUT
            state.failure_reason = f"Max step horizon reached ({state.max_steps} steps)"

        logger.info(
            f"[HarnessLoop] Run finished with status={state.status.value}, "
            f"steps={state.step_count}, cost=${state.cumulative_cost_usd:.4f}"
        )
        return state

    def _record_step(
        self,
        state: HarnessState,
        step_type: StepType,
        outcome: str = "SUCCESS",
        provider: str = "",
        model: str = "",
        duration_ms: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_cached: int = 0,
        cost_usd: float = 0.0,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepRecord:
        record = StepRecord(
            step_id=state.step_count + 1,
            step_type=step_type,
            provider=provider or getattr(self.provider, "provider_id", ""),
            model=model,
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_cached=tokens_cached,
            cost_usd=cost_usd,
            outcome=outcome,
            input_data=input_data or {},
            output_data=output_data or {},
            error=error,
            metadata=metadata or {},
        )
        state.add_step(record)

        # Emit trace event if handler registered
        if self.trace_emitter:
            try:
                self.trace_emitter(record, state)
            except Exception as exc:
                logger.error(f"[HarnessLoop] Error in trace emitter: {exc}")

        return record
