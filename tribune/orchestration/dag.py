"""Manager/subagent DAG.

The navigator (manager) decomposes a person's situation into a DAG of sub-tasks:
a single GATHER task (ingest + consolidate evidence) that every per-program ASSESS
task depends on. The runner executes tasks in dependency order. Failure handling is
two-tiered: each per-program task runs its own REPLAN state machine internally, and
the DAG runner additionally fail-safes any uncaught task error into a safe
abstention rather than letting a case crash.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..types import ProgramId


@dataclass
class Task:
    task_id: str
    kind: str  # "gather" | "assess"
    deps: list[str] = field(default_factory=list)
    program: ProgramId | None = None


class DAG:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.task_id] = task

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


class DAGRunner:
    """Executes a DAG in topological order, collecting per-task results."""

    def run(self, dag: DAG, executor: Callable[[Task], object]) -> dict[str, object]:
        results: dict[str, object] = {}
        for task in dag.topological_order():
            results[task.task_id] = executor(task)
        return results
