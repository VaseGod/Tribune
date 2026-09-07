"""Manager/subagent DAG with async concurrency and wave-based scheduling.

The navigator (manager) decomposes a person's situation into a DAG of sub-tasks:
a single GATHER task (ingest + consolidate evidence) that every per-program ASSESS
task depends on.

The execution engine supports:
1. :class:`DAGRunner`: Synchronous topological execution.
2. :class:`AsyncDAGRunner`: Asynchronous wave-based execution with `asyncio.gather`
   fan-out across independent subagent tasks (e.g. parallel Medicaid and SNAP evaluations)
   and deterministic error boundaries.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..types import ProgramId


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    task_id: str
    kind: str  # "gather" | "assess" | "verify" | "prepare"
    deps: list[str] = field(default_factory=list)
    program: ProgramId | None = None
    subagent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "deps": list(self.deps),
            "program": self.program.value if self.program else None,
            "subagent_id": self.subagent_id,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        prog = ProgramId(data["program"]) if data.get("program") else None
        status = TaskStatus(data.get("status", "pending"))
        return cls(
            task_id=data["task_id"],
            kind=data["kind"],
            deps=list(data.get("deps", [])),
            program=prog,
            subagent_id=data.get("subagent_id"),
            status=status,
            error=data.get("error"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


class DAG:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    @property
    def tasks(self) -> dict[str, Task]:
        return dict(self._tasks)

    def topological_order(self) -> list[Task]:
        """Kahn's algorithm; raises on a cycle or a missing dependency."""
        indeg = {tid: 0 for tid in self._tasks}
        for t in self._tasks.values():
            for d in t.deps:
                if d not in self._tasks:
                    raise ValueError(f"task '{t.task_id}' depends on unknown task '{d}'")
                indeg[t.task_id] += 1
        queue = [tid for tid, n in indeg.items() if n == 0]
        order: list[Task] = []
        while queue:
            tid = queue.pop(0)
            order.append(self._tasks[tid])
            for t in self._tasks.values():
                if tid in t.deps:
                    indeg[t.task_id] -= 1
                    if indeg[t.task_id] == 0:
                        queue.append(t.task_id)
        if len(order) != len(self._tasks):
            raise ValueError("DAG contains a cycle")
        return order

    def topological_waves(self) -> list[list[Task]]:
        """Group tasks into independent execution waves for safe async fan-out.

        Wave 0 contains root tasks with 0 dependencies. Wave k contains tasks whose
        dependencies all reside in waves < k.
        """
        self.topological_order()  # Validates dependencies and cycle-freedom
        completed: set[str] = set()
        waves: list[list[Task]] = []

        while len(completed) < len(self._tasks):
            current_wave_ids = [
                tid for tid, t in self._tasks.items()
                if tid not in completed and all(d in completed for d in t.deps)
            ]
            if not current_wave_ids:
                raise ValueError("DAG deadlock: unable to form next execution wave")
            waves.append([self._tasks[tid] for tid in current_wave_ids])
            completed.update(current_wave_ids)

        return waves

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()}
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DAG:
        dag = cls()
        tasks_data = data.get("tasks", {})
        for tdict in tasks_data.values():
            dag.add(Task.from_dict(tdict))
        return dag


@dataclass
class DAGRunContext:
    """Execution context for a DAG run with execution trace and prompt caching support."""

    run_id: str = field(default_factory=lambda: f"dag_run_{int(time.time() * 1000)}")
    is_fallback_active: bool = False
    fallback_model: str | None = None
    fallback_reason: str = ""
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_turn(self, turn_data: dict[str, Any]) -> None:
        """Record an executed turn (message, tool invocation, or observation)."""
        self.execution_trace.append(dict(turn_data))

    def tag_fallback(self, fallback_model: str, reason: str = "") -> None:
        """Tag run context with detected upstream model fallback."""
        self.is_fallback_active = True
        self.fallback_model = fallback_model
        self.fallback_reason = reason
        self.metadata["model_fallback"] = {
            "fallback_model": fallback_model,
            "reason": reason,
            "tagged_at": time.time(),
        }

    def inject_penultimate_cache_breakpoint(self) -> list[dict[str, Any]]:
        """Identify the second-to-last turn (N-1) and inject an ephemeral cache breakpoint.

        Returns a deepcopy of the execution trace where turn N-1 carries
        cache_control: {"type": "ephemeral"}, while the latest active turn N remains uncached.
        """
        import copy
        trace_copy = copy.deepcopy(self.execution_trace)
        if not trace_copy:
            return trace_copy

        # If 2 or more turns, target index is N-2 (second to last)
        # If 1 turn, target index is 0
        target_idx = max(0, len(trace_copy) - 2) if len(trace_copy) >= 2 else 0
        target_item = trace_copy[target_idx]

        cache_ctrl = {"type": "ephemeral"}
        content = target_item.get("content")
        if isinstance(content, str):
            target_item["content"] = [{"type": "text", "text": content, "cache_control": cache_ctrl}]
        elif isinstance(content, list) and content:
            if isinstance(content[-1], dict):
                content[-1]["cache_control"] = cache_ctrl
        elif isinstance(content, dict):
            content["cache_control"] = cache_ctrl
        else:
            target_item["cache_control"] = cache_ctrl

        # Ensure the latest turn (index -1 if len >= 2) has NO cache_control (dynamic turn)
        if len(trace_copy) >= 2:
            latest_item = trace_copy[-1]
            latest_item.pop("cache_control", None)
            lcontent = latest_item.get("content")
            if isinstance(lcontent, list):
                for blk in lcontent:
                    if isinstance(blk, dict):
                        blk.pop("cache_control", None)
            elif isinstance(lcontent, dict):
                lcontent.pop("cache_control", None)

        return trace_copy


class DAGRunner:
    """Executes a DAG synchronously in topological order, collecting per-task results."""

    def run(
        self,
        dag: DAG,
        executor: Callable[[Task], object],
        context: DAGRunContext | None = None,
    ) -> dict[str, object]:
        results: dict[str, object] = {}
        for task in dag.topological_order():
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            try:
                res = executor(task)
                task.result = res
                task.status = TaskStatus.COMPLETED
                results[task.task_id] = res
                if context:
                    context.record_turn({
                        "task_id": task.task_id,
                        "kind": task.kind,
                        "role": "assistant",
                        "content": str(res),
                        "status": task.status.value,
                    })
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                if context:
                    context.record_turn({
                        "task_id": task.task_id,
                        "kind": task.kind,
                        "role": "system",
                        "content": f"Task failed: {exc}",
                        "status": task.status.value,
                    })
                raise
            finally:
                task.completed_at = time.time()
        return results


class AsyncDAGRunner:
    """Executes a DAG asynchronously in parallel waves, handling fan-out and safe merge-back."""

    async def run_async(
        self,
        dag: DAG,
        async_executor: Callable[[Task], Awaitable[object]],
        context: DAGRunContext | None = None,
    ) -> dict[str, object]:
        results: dict[str, object] = {}
        waves = dag.topological_waves()

        for wave in waves:
            async def _run_single(task: Task) -> tuple[str, object]:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                try:
                    res = await async_executor(task)
                    task.result = res
                    task.status = TaskStatus.COMPLETED
                    if context:
                        context.record_turn({
                            "task_id": task.task_id,
                            "kind": task.kind,
                            "role": "assistant",
                            "content": str(res),
                            "status": task.status.value,
                        })
                    return task.task_id, res
                except Exception as exc:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    if context:
                        context.record_turn({
                            "task_id": task.task_id,
                            "kind": task.kind,
                            "role": "system",
                            "content": f"Task failed: {exc}",
                            "status": task.status.value,
                        })
                    raise
                finally:
                    task.completed_at = time.time()

            wave_results = await asyncio.gather(
                *[_run_single(t) for t in wave],
                return_exceptions=False,
            )
            for tid, res in wave_results:
                results[tid] = res

        return results


__all__ = [
    "Task",
    "TaskStatus",
    "DAG",
    "DAGRunContext",
    "DAGRunner",
    "AsyncDAGRunner",
]
