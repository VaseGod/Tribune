"""Reasoning Budget Metadata & Telemetry Tracker.

Attaches calibrated reasoning budgets to each task run:
- Task complexity class
- Selected model and tier
- Expected token budget vs. actual usage
- Routing rationale
- Confidence / complexity scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .task_classifier import TaskComplexityClass, TaskRoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class TaskReasoningBudget:
    """Telemetry record documenting dynamic reasoning allocation for a task."""

    task_id: str
    task_class: TaskComplexityClass
    selected_tier: str
    selected_model: str
    expected_token_budget: int
    actual_tokens_input: int = 0
    actual_tokens_output: int = 0
    actual_cost_usd: float = 0.0
    routing_reason: str = ""
    complexity_score: float = 0.0
    budget_exceeded: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_actual_tokens(self) -> int:
        return self.actual_tokens_input + self.actual_tokens_output

    @property
    def token_variance(self) -> int:
        """Difference between actual token consumption and expected budget."""
        return self.total_actual_tokens - self.expected_token_budget

    @classmethod
    def from_decision(cls, task_id: str, decision: TaskRoutingDecision) -> TaskReasoningBudget:
        return cls(
            task_id=task_id,
            task_class=decision.task_class,
            selected_tier=decision.selected_tier,
            selected_model=decision.selected_model,
            expected_token_budget=decision.expected_token_budget,
            routing_reason=decision.routing_reason,
            complexity_score=decision.complexity_score,
            metadata=decision.signals,
        )

    def record_completion(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.actual_tokens_input = tokens_in
        self.actual_tokens_output = tokens_out
        self.actual_cost_usd = cost_usd
        if self.total_actual_tokens > self.expected_token_budget:
            self.budget_exceeded = True
            logger.warning(
                f"[ReasoningBudget] Task '{self.task_id}' exceeded expected budget: "
                f"{self.total_actual_tokens} tokens consumed > {self.expected_token_budget} expected budget."
            )
