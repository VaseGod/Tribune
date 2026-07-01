"""Navigator (manager agent).

Decomposes a person's situation into a sub-task DAG: one GATHER task that ingests
and consolidates the available evidence, and one ASSESS task per target program
that depends on it. The per-program assessment work is delegated to the eligibility
proposer / verifier / preparer via the state machine.
"""

from __future__ import annotations

from ..orchestration.dag import DAG, Task
from ..types import ProgramId, SyntheticCase


class Navigator:
    def plan(self, case: SyntheticCase) -> DAG:
        dag = DAG()
        dag.add(Task(task_id="gather", kind="gather"))
        for program in case.target_programs:
            dag.add(
                Task(
                    task_id=f"assess:{program.value}",
                    kind="assess",
                    deps=["gather"],
                    program=program,
                )
            )
        return dag

    @staticmethod
    def target_programs(case: SyntheticCase) -> list[ProgramId]:
        return list(case.target_programs)
