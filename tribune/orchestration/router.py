"""Routing by case complexity.

The router decides the retrieval breadth (top-k) and provider tier for a program,
and how to escalate on REPLAN. Complex programs (Medicaid, Housing) start with a
deliberately *narrow* retrieval so the verifier's coverage check exercises the
REPLAN path; on escalation the router widens retrieval to cover every governing
rule and (in a served deployment) can route the re-derivation to a stronger model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..corpus import programs as program_registry
from ..types import ProgramId

_COMPLEX = {ProgramId.MEDICAID, ProgramId.HOUSING}


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    k: int
    note: str
    target_model: str = "DeepSeek V4 Pro"
    cost_per_m_input: float = 0.435


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
