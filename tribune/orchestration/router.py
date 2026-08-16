"""Routing by case complexity.

The router decides the retrieval breadth (top-k) and provider tier for a program,
and how to escalate on REPLAN. Complex programs (Medicaid, Housing) start with a
deliberately *narrow* retrieval so the verifier's coverage check exercises the
REPLAN path; on escalation the router widens retrieval to cover every governing
rule and (in a served deployment) can route the re-derivation to a stronger model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..corpus import programs as program_registry
from ..governance.audit import CheckpointManager
from ..types import AuditRecord, ProgramId, ProgramOutcome
from .dag import DAG, TaskStatus

_COMPLEX = {ProgramId.MEDICAID, ProgramId.HOUSING}


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    k: int
    note: str
    target_model: str = "DeepSeek V4 Pro"
    cost_per_m_input: float = 0.435


@dataclass
class RecoveryPlan:
    """Reconstructed execution plan and state generated from a recovered checkpoint."""

    case_id: str
    jurisdiction: str
    completed_task_ids: list[str]
    pending_task_ids: list[str]
    reconstructed_dag: DAG
    restored_outcomes: list[ProgramOutcome]
    restored_audit: list[AuditRecord]
    memory_snapshot: dict[str, Any]
    checkpoint_timestamp: str


class Router:
    def initial(self, program: ProgramId) -> RouteDecision:
        total = len(program_registry.get_ruleset(program).rules)
        if program in _COMPLEX:
            # Start narrow: omit one governing rule so coverage is incomplete and the
            # verifier triggers a REPLAN (a realistic retrieval-recall failure).
            return RouteDecision(
                tier=0,
                k=max(1, total - 1),
                note="complex program: narrow first pass",
                target_model="DeepSeek V4 Pro",
                cost_per_m_input=0.435,
            )
        return RouteDecision(
            tier=1,
            k=total,
            note="standard program: full retrieval",
            target_model="DeepSeek V4 Pro",
            cost_per_m_input=0.435,
        )

    def escalate(self, program: ProgramId, current: RouteDecision) -> RouteDecision:
        total = len(program_registry.get_ruleset(program).rules)
        return RouteDecision(
            tier=current.tier + 1,
            k=total,
            note="escalated: full retrieval over all governing rules (and stronger Grok 4.6 model if served)",
            target_model="Grok 4.6",
            cost_per_m_input=2.00,
        )

    def inspect_checkpoint(self, path: str = ".tribune/last_run.json") -> RecoveryPlan | None:
        """Inspect checkpoint file upon startup, detect incomplete runs, and reconstruct in-memory DAG and state."""
        chk = CheckpointManager.load_checkpoint(path)
        if not chk:
            return None

        dag_dict = chk.get("dag", {})
        dag = DAG.from_dict(dag_dict)
        completed_ids = set(chk.get("completed_task_ids", []))
        all_ids = set(dag.tasks.keys())
        pending_ids = [tid for tid in dag.topological_order() if tid.task_id not in completed_ids]

        if not pending_ids and completed_ids == all_ids:
            # All tasks completed in checkpoint
            return None

        for tid_str in completed_ids:
            task = dag.get(tid_str)
            if task:
                task.status = TaskStatus.COMPLETED

        restored_audit = [AuditRecord.model_validate(r) for r in chk.get("audit_records", [])]
        restored_outcomes = [ProgramOutcome.model_validate(o) for o in chk.get("outcomes", [])]

        return RecoveryPlan(
            case_id=chk["case_id"],
            jurisdiction=chk.get("jurisdiction", "EX"),
            completed_task_ids=list(completed_ids),
            pending_task_ids=[t.task_id for t in pending_ids],
            reconstructed_dag=dag,
            restored_outcomes=restored_outcomes,
            restored_audit=restored_audit,
            memory_snapshot=chk.get("memory_snapshot", {}),
            checkpoint_timestamp=chk.get("timestamp", ""),
        )

    def route_task(
        self,
        task_type: str,
        context_length: int = 0,
        program: ProgramId | None = None,
    ) -> RouteDecision:
        """Dynamic, task-aware work allocation:

        - High-volume document parsing, multi-modal intake processing, and synthetic case generation
          routed to DeepSeek V4 Pro ($0.435/M input tokens).
        - Complex statutory eligibility determinations and independent verifications
          routed to Grok 4.6 ($2.00/M input tokens).
        """
        task_norm = (task_type or "").lower().strip()

        tier1_tasks = {
            "parsing",
            "document_parsing",
            "intake",
            "multimodal_intake",
            "ocr",
            "synthetic",
            "synthetic_casegen",
            "casegen",
            "classification",
            "formatting",
            "search_query_generation",
            "extraction",
            "utility",
        }

        tier2_tasks = {
            "eligibility",
            "statutory_determination",
            "complex_synthesis",
            "verifier",
            "verification",
            "multi_step_verification",
            "rederivation",
            "reasoning",
            "edge_case_analysis",
        }

        if task_norm in tier1_tasks:
            return RouteDecision(
                tier=1,
                k=8,
                note=f"high-volume task '{task_type}' routed to DeepSeek V4 Pro",
                target_model="DeepSeek V4 Pro",
                cost_per_m_input=0.435,
            )

        if task_norm in tier2_tasks or (program in _COMPLEX and task_norm not in tier1_tasks) or context_length > 4000:
            return RouteDecision(
                tier=2,
                k=8,
                note=f"complex statutory reasoning task '{task_type}' routed to Grok 4.6",
                target_model="Grok 4.6",
                cost_per_m_input=2.00,
            )

        return RouteDecision(
            tier=1,
            k=8,
            note=f"standard task '{task_type}' routed to DeepSeek V4 Pro",
            target_model="DeepSeek V4 Pro",
            cost_per_m_input=0.435,
        )


__all__ = ["RouteDecision", "RecoveryPlan", "Router"]
