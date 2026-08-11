"""Navigator (manager agent).

Decomposes a person's situation into a sub-task DAG: one GATHER task that ingests
and consolidates the available evidence, and one ASSESS task per target program
that depends on it. The per-program assessment work is delegated to the eligibility
proposer / verifier / preparer via the state machine.
"""

from __future__ import annotations

from ..orchestration.dag import DAG, Task
from ..types import ProgramId, SyntheticCase


class ProgrammaticNavigatorTools:
    """Typed Python stubs executed directly in-code by navigator agent loops."""

    @staticmethod
    def build_dag(target_programs: list[str]) -> dict:
        """Construct execution DAG tasks for evidence gathering and per-program assessment."""
        tasks = [{"task_id": "gather", "kind": "gather"}]
        for prog in target_programs:
            tasks.append({"task_id": f"assess:{prog}", "kind": "assess", "deps": ["gather"], "program": prog})
        return {"tasks": tasks}

    @classmethod
    def get_tool_signatures(cls) -> str:
        """Expose typed Python signatures for model prompt generation."""
        return (
            "class ProgrammaticNavigatorTools:\n"
            "    @staticmethod\n"
            "    def build_dag(target_programs: list[str]) -> dict: ...\n"
        )


class Navigator:
    def __init__(self) -> None:
        self.tools = ProgrammaticNavigatorTools()

    def generate_prompt(self, target_programs: list[ProgramId]) -> str:
        """Generate prompt incorporating programmatic Python tool signatures."""
        progs = [p.value for p in target_programs]
        return (
            f"You are the navigator agent planning tasks for programs: {progs}.\n"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticNavigatorTools.get_tool_signatures()}\n"
            "Use these tools directly to decompose workflow plans into execution DAGs."
        )

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

