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


class Router:
    def initial(self, program: ProgramId) -> RouteDecision:
        total = len(program_registry.get_ruleset(program).rules)
        if program in _COMPLEX:
            # Start narrow: omit one governing rule so coverage is incomplete and the
            # verifier triggers a REPLAN (a realistic retrieval-recall failure).
            return RouteDecision(tier=0, k=max(1, total - 1), note="complex program: narrow first pass")
        return RouteDecision(tier=1, k=total, note="standard program: full retrieval")

    def escalate(self, program: ProgramId, current: RouteDecision) -> RouteDecision:
        total = len(program_registry.get_ruleset(program).rules)
        return RouteDecision(
            tier=current.tier + 1,
            k=total,
            note="escalated: full retrieval over all governing rules (and stronger model if served)",
        )
