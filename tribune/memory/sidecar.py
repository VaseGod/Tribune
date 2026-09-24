"""Proactive Memory Sidecar Engine for Quantized Model Execution.

Operates in the auxiliary intelligence tier, updating the five-track state bank
at interval k (default k=2 turns) and evaluating the sparse intervention policy
to prevent quantized models from catastrophic loops, path hallucination, and premature completion.
"""

from __future__ import annotations

import logging
from typing import Any

from ..security.token_broker import TokenBroker, get_token_broker
from .aux_backend import AuxiliaryBackend, HeuristicAuxBackend
from .intervention_policy import InterventionDecision, InterventionPolicy, SILENCE_TOKEN
from .state_bank import MemoryStateBank, TaskRequirement
from .trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)


class ProactiveMemorySidecar:
    """Sidecar maintaining structured execution memory and sparse intervention gates."""

    def __init__(
        self,
        task_id: str = "default_task",
        session_id: str = "default_session",
        interval_k: int = 2,
        enabled: bool = True,
        aux_backend: AuxiliaryBackend | None = None,
        intervention_policy: InterventionPolicy | None = None,
        token_broker: TokenBroker | None = None,
        max_context_chars: int = 4000,
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.interval_k = max(1, interval_k)
        self.enabled = enabled
        self.max_context_chars = max_context_chars

        self.state_bank = MemoryStateBank()
        self.trajectory_buffer = TrajectoryBuffer(task_id=task_id, session_id=session_id)
        self.aux_backend = aux_backend or HeuristicAuxBackend()
        self.policy = intervention_policy or InterventionPolicy()
        self.token_broker = token_broker or get_token_broker()

        # Telemetry metrics
        self._total_invocations: int = 0
        self._interventions_issued: int = 0
        self._silence_count: int = 0
        self._avoided_loop_count: int = 0

    def load_task_requirements(self, requirements: list[TaskRequirement]) -> None:
        """Initialize the state bank with formal requirements for the task."""
        self.state_bank.unmet_task_requirements = list(requirements)

    def record_turn_and_evaluate(
        self,
        turn_index: int,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_seconds: float = 0.0,
        candidate_next_command: str | None = None,
        agent_declared_complete: bool = False,
    ) -> InterventionDecision:
        """Record turn outcome, update state bank at interval k, and evaluate intervention policy."""
        if not self.enabled:
            return InterventionDecision.silence(turn_index=turn_index, reason="sidecar_disabled")

        # 1. Append to trajectory buffer
        clean_cmd = self.token_broker.redact_text(command)
        clean_stdout = self.token_broker.redact_text(stdout)
        clean_stderr = self.token_broker.redact_text(stderr)

        self.trajectory_buffer.append_turn(
            turn_index=turn_index,
            command=command,
            exit_code=exit_code,
            stdout=clean_stdout,
            stderr=clean_stderr,
            duration_seconds=duration_seconds,
            redacted_command=clean_cmd,
        )

        last_turn_data = {
            "turn_index": turn_index,
            "command": clean_cmd,
            "exit_code": exit_code,
            "stdout": clean_stdout,
            "stderr": clean_stderr,
            "duration": duration_seconds,
        }

        # 2. Update state bank at interval k or on non-zero error / completion attempt
        should_update_state = (
            (turn_index % self.interval_k == 0)
            or (exit_code != 0)
            or agent_declared_complete
        )

        if should_update_state:
            self._total_invocations += 1
            delta_ctx = self.trajectory_buffer.to_localized_context(max_chars=self.max_context_chars)
            self.state_bank, _ = self.aux_backend.update_state_bank(
                state_bank=self.state_bank,
                trajectory_context=delta_ctx,
                last_turn_data=last_turn_data,
            )

        # 3. Evaluate intervention gate
        recent_cmds = [t.command for t in self.trajectory_buffer.get_window(window_size=4)]
        decision = self.policy.evaluate(
            turn_index=turn_index,
            state_bank=self.state_bank,
            candidate_command=candidate_next_command,
            agent_declared_complete=agent_declared_complete,
            recent_commands=recent_cmds,
        )

        # 4. Update metrics
        if decision.should_intervene:
            self._interventions_issued += 1
            if decision.reason_code in ("REPEATED_FAILED_COMMAND", "COMMAND_LOOP_DETECTED"):
                self._avoided_loop_count += 1
            # Record intervention text in trajectory turn
            last_turn = self.trajectory_buffer.get_last_turn()
            if last_turn:
                last_turn.intervention_applied = True
                last_turn.intervention_text = decision.injected_text
        else:
            self._silence_count += 1

        return decision

    def get_metrics(self) -> dict[str, Any]:
        """Return operational telemetry metrics for cost and reliability reporting."""
        return {
            "total_turns": self.trajectory_buffer.total_turns,
            "sidecar_invocations": self._total_invocations,
            "interventions_issued": self._interventions_issued,
            "silence_count": self._silence_count,
            "avoided_loops_estimate": self._avoided_loop_count,
            "unmet_requirements_count": len([r for r in self.state_bank.unmet_task_requirements if not r.satisfied]),
            "failed_commands_cached": len(self.state_bank.failed_commands_cache.entries),
            "mutations_recorded": len(self.state_bank.successful_mutations),
        }
