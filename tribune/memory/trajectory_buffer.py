"""Trajectory Buffer for Multi-Turn Execution History.

Tracks chronological execution actions, shell inputs, outputs, exit codes, and durations
to provide localized context windows (Δctx_t) to the auxiliary intelligence tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryTurn:
    """A single execution turn in the trajectory."""

    turn_index: int
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    redacted_command: str = ""
    intervention_applied: bool = False
    intervention_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class TrajectoryBuffer:
    """Thread-safe append-only buffer for evaluation trajectory events."""

    def __init__(self, task_id: str = "default_task", session_id: str = "default_session") -> None:
        self.task_id = task_id
        self.session_id = session_id
        self._turns: list[TrajectoryTurn] = []

    def append_turn(
        self,
        turn_index: int,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_seconds: float = 0.0,
        redacted_command: str = "",
        intervention_applied: bool = False,
        intervention_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryTurn:
        """Record an execution turn into the trajectory buffer."""
        turn = TrajectoryTurn(
            turn_index=turn_index,
            command=command,
            exit_code=exit_code,
            stdout=stdout[:2000],  # Bounded preview to prevent memory explosion
            stderr=stderr[:2000],
            duration_seconds=duration_seconds,
            redacted_command=redacted_command or command,
            intervention_applied=intervention_applied,
            intervention_text=intervention_text,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        return turn

    @property
    def turns(self) -> list[TrajectoryTurn]:
        """Return all recorded turns."""
        return list(self._turns)

    @property
    def total_turns(self) -> int:
        return len(self._turns)

    def get_last_turn(self) -> TrajectoryTurn | None:
        return self._turns[-1] if self._turns else None

    def get_window(self, window_size: int = 3) -> list[TrajectoryTurn]:
        """Return the most recent `window_size` turns."""
        return self._turns[-window_size:] if self._turns else []

    def count_command_occurrences(self, command: str) -> int:
        """Count how many times a normalized command has been executed."""
        norm = command.strip().lower()
        return sum(1 for t in self._turns if t.command.strip().lower() == norm)

    def to_localized_context(self, max_turns: int = 3, max_chars: int = 2000) -> str:
        """Format a localized trajectory window (Δctx_t) for auxiliary sidecar input."""
        recent = self.get_window(window_size=max_turns)
        if not recent:
            return "No previous execution history."

        lines: list[str] = [f"--- Trajectory Window (Task: {self.task_id}) ---"]
        for t in recent:
            status = "SUCCESS" if t.exit_code == 0 else f"FAILED (exit {t.exit_code})"
            lines.append(f"[Turn {t.turn_index}] Cmd: {t.redacted_command} -> {status}")
            if t.stderr.strip():
                lines.append(f"  stderr: {t.stderr.strip()[:200]}")
            if t.stdout.strip():
                lines.append(f"  stdout: {t.stdout.strip()[:200]}")
            if t.intervention_applied:
                lines.append(f"  [Intervention Active]: {t.intervention_text}")

        content = "\n".join(lines)
        if len(content) > max_chars:
            return content[-max_chars:]
        return content
