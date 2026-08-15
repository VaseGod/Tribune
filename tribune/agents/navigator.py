from __future__ import annotations

from typing import TYPE_CHECKING

from ..orchestration.dag import DAG, Task
from ..types import ProgramId, SyntheticCase

if TYPE_CHECKING:
    from ..corpus.rule_store import RuleStore


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
    def __init__(self, rule_store: RuleStore | None = None) -> None:
        self.rule_store = rule_store
        self.tools = ProgrammaticNavigatorTools()

    def generate_prompt(self, target_programs: list[ProgramId], jurisdiction: str = "EX") -> str:
        """Generate prompt incorporating programmatic Python tool signatures strictly scoped to target programs."""
        progs = [p.value for p in target_programs]
        scoped_schemas = ""
        if self.rule_store is not None:
            schemas = [self.rule_store.get_scoped_schema(p, jurisdiction) for p in target_programs]
            scoped_schemas = f"\nScoped Program Schemas (Unselected domain schemas pruned): {schemas}\n"
        return (
            f"You are the navigator agent planning tasks for programs: {progs}.\n"
            f"{scoped_schemas}"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticNavigatorTools.get_tool_signatures()}\n"
            "Use these tools directly to decompose workflow plans into execution DAGs."
        )

    def plan(self, case: SyntheticCase) -> DAG:
        """Dynamically bind and expose only DAG tasks for programs in the case intake payload."""
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

