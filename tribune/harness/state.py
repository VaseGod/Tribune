"""Execution Engine State Primitives & Step Tracking.

Provides typed state representations, step lifecycle records, and resumable
checkpoints for the harness loop.
"""

from __future__ import annotations

import copy
import enum
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class RunStatus(str, enum.Enum):
    """Lifecycle status of a harness execution run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    KILLED = "KILLED"
    TIMED_OUT = "TIMED_OUT"


class StepType(str, enum.Enum):
    """Discrete functional steps within the harness loop."""

    INFERENCE = "INFERENCE"
    TOOL = "TOOL"
    VERIFIER = "VERIFIER"
    COMPACTION = "COMPACTION"
    POLICY_CHECK = "POLICY_CHECK"
    SYSTEM = "SYSTEM"


@dataclass
class StepRecord:
    """Detailed audit record of an individual step executed by the harness."""

    step_id: int
    step_type: StepType
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    provider: str = ""
    model: str = ""
    duration_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    outcome: str = "SUCCESS"  # SUCCESS | FAILED | BLOCKED | SKIPPED
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessState:
    """Execution state machine container supporting resumable checkpoints."""

    task_id: str
    run_id: str = field(default_factory=lambda: f"run_{secrets.token_hex(6)}")
    status: RunStatus = RunStatus.PENDING
    step_count: int = 0
    max_steps: int = 24
    steps: list[StepRecord] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    context_messages: list[dict[str, Any]] = field(default_factory=list)
    cumulative_cost_usd: float = 0.0
    cumulative_tokens: int = 0
    kill_requested: bool = False
    stop_requested: bool = False
    failure_reason: str | None = None

    def add_step(self, record: StepRecord) -> None:
        """Append a step record and update cumulative metrics."""
        self.steps.append(record)
        self.step_count += 1
        self.cumulative_cost_usd += record.cost_usd
        self.cumulative_tokens += (record.tokens_input + record.tokens_output)

    def request_stop(self, reason: str = "Graceful stop requested") -> None:
        """Signal graceful loop termination at the next safe boundary."""
        self.stop_requested = True
        self.failure_reason = reason

    def request_kill(self, reason: str = "Immediate kill triggered") -> None:
        """Signal immediate emergency loop termination."""
        self.kill_requested = True
        self.status = RunStatus.KILLED
        self.failure_reason = reason

    def checkpoint(self) -> dict[str, Any]:
        """Produce an immutable serialized snapshot for state resumption."""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "variables": copy.deepcopy(self.variables),
            "context_messages": copy.deepcopy(self.context_messages),
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "cumulative_tokens": self.cumulative_tokens,
            "kill_requested": self.kill_requested,
            "stop_requested": self.stop_requested,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> HarnessState:
        """Reconstruct state from a saved checkpoint snapshot."""
        state = cls(
            task_id=snapshot["task_id"],
            run_id=snapshot["run_id"],
            status=RunStatus(snapshot["status"]),
            step_count=snapshot["step_count"],
            max_steps=snapshot["max_steps"],
            variables=copy.deepcopy(snapshot.get("variables", {})),
            context_messages=copy.deepcopy(snapshot.get("context_messages", [])),
            cumulative_cost_usd=snapshot.get("cumulative_cost_usd", 0.0),
            cumulative_tokens=snapshot.get("cumulative_tokens", 0),
            kill_requested=snapshot.get("kill_requested", False),
            stop_requested=snapshot.get("stop_requested", False),
            failure_reason=snapshot.get("failure_reason"),
        )
        return state
