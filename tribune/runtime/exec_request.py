"""Data contracts and interfaces for dual-zone execution partitioning.

Separates host orchestration (trusted zone) from sandboxed model execution (unprivileged zone).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecRequest:
    """Request to execute a command within the sandboxed runtime."""

    command: str | list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    session_id: str = "default_session"
    task_id: str = "default_task"
    model_backend_id: str = "quant_model"
    requested_network_destinations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_command_str(self) -> str:
        """Return a string representation of the command."""
        if isinstance(self.command, list):
            return " ".join(self.command)
        return str(self.command)


@dataclass
class ExecResult:
    """Outcome of sandboxed command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    redacted_command: str = ""
    surrogate_env_keys: list[str] = field(default_factory=list)
    sentinel_decision: str = "ALLOWED"  # "ALLOWED" | "DENIED" | "BLOCKED"
    intervention_metadata: dict[str, Any] | None = None
    isolation_mode: str = "container"  # "container" | "local_fallback" | "legacy"
    timed_out: bool = False
    killed: bool = False
    unsafe_for_production: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.killed


class ShellRunner(ABC):
    """Abstract interface for shell runners in the dual-zone architecture."""

    @abstractmethod
    def execute(self, request: ExecRequest) -> ExecResult:
        """Execute the command specified in request in the appropriate runtime zone."""
        raise NotImplementedError
