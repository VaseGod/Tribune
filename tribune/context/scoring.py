"""Calibrated Scoring Kernels and Utility Attribution for Working Memory Nodes.

Provides continuous utility ranking across sliding windows using calibrated kernels
combining recency, frequency, graph centrality, task relevance, semantic relevance,
escalation status, and downstream reference count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ContextNode:
    """A scored unit of context memory (evidence item, proposition, or dialogue turn)."""

    node_id: str
    text: str
    token_count: int
    created_step: int
    last_accessed_step: int
    access_count: int = 1
    graph_centrality: float = 0.1
    task_relevance: float = 0.5
    semantic_relevance: float = 0.5
    in_escalation: bool = False
    downstream_references: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    is_evicted: bool = False
    summary: str = ""


@dataclass
class ScoringWeights:
    """Configurable weights for the calibrated utility scoring kernel."""

    w_recency: float = 0.20
    w_frequency: float = 0.15
    w_centrality: float = 0.15
    w_task: float = 0.20
    w_semantic: float = 0.10
    w_escalation: float = 0.10
    w_downstream: float = 0.10

    def normalized(self) -> ScoringWeights:
        total = (
            self.w_recency
            + self.w_frequency
            + self.w_centrality
            + self.w_task
            + self.w_semantic
            + self.w_escalation
            + self.w_downstream
        )
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            w_recency=self.w_recency / total,
            w_frequency=self.w_frequency / total,
            w_centrality=self.w_centrality / total,
            w_task=self.w_task / total,
            w_semantic=self.w_semantic / total,
            w_escalation=self.w_escalation / total,
            w_downstream=self.w_downstream / total,
        )


@dataclass
class ScoringExplanation:
    """Explainable attribution and breakdown of a node's utility score."""

    node_id: str
    total_score: float
    component_scores: dict[str, float]
    rationale: str


@runtime_checkable
class NodeUtilityScorer(Protocol):
    """Protocol for pluggable context node utility scoring engines."""

    def score(
        self,
        node: ContextNode,
        current_step: int,
        context_graph: Any | None = None,
    ) -> float:
        """Compute utility score in range [0.0, 1.0]."""
        ...

    def explain(
        self,
        node: ContextNode,
        current_step: int,
        context_graph: Any | None = None,
    ) -> ScoringExplanation:
        """Provide explainable breakdown of utility score."""
        ...


class CalibratedKernelScorer:
    """Calibrated kernel scoring node utility across 7 dimensions.

    Guarantees deterministic scoring when provided identical inputs and steps.
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        decay_half_life_steps: float = 20.0,
        decay_curve: str = "exponential",  # "exponential" | "linear" | "sigmoid"
    ) -> None:
        self.weights = (weights or ScoringWeights()).normalized()
        self.decay_half_life_steps = max(1.0, decay_half_life_steps)
        self.decay_curve = decay_curve

    def _compute_recency_score(self, node: ContextNode, current_step: int) -> float:
        step_delta = max(0, current_step - node.last_accessed_step)

        if self.decay_curve == "linear":
            max_horizon = self.decay_half_life_steps * 2.0
            return max(0.0, 1.0 - (step_delta / max_horizon))
        elif self.decay_curve == "sigmoid":
            # Sigmoid centered around half_life
            k = 0.2
            return 1.0 / (1.0 + math.exp(k * (step_delta - self.decay_half_life_steps)))
        else:
            # Exponential decay: e^(-lambda * delta)
            decay_lambda = math.log(2.0) / self.decay_half_life_steps
            return math.exp(-decay_lambda * step_delta)

    def _compute_frequency_score(self, node: ContextNode) -> float:
        # Log-damped frequency saturation
        return min(1.0, math.log1p(node.access_count) / math.log1p(20.0))

    def _compute_downstream_score(self, node: ContextNode) -> float:
        return min(1.0, math.log1p(node.downstream_references) / math.log1p(10.0))

    def score(
        self,
        node: ContextNode,
        current_step: int,
        context_graph: Any | None = None,
    ) -> float:
        recency = self._compute_recency_score(node, current_step)
        frequency = self._compute_frequency_score(node)
        centrality = min(1.0, max(0.0, node.graph_centrality))
        task = min(1.0, max(0.0, node.task_relevance))
        semantic = min(1.0, max(0.0, node.semantic_relevance))
        escalation = 1.0 if node.in_escalation else 0.0
        downstream = self._compute_downstream_score(node)

        total = (
            self.weights.w_recency * recency
            + self.weights.w_frequency * frequency
            + self.weights.w_centrality * centrality
            + self.weights.w_task * task
            + self.weights.w_semantic * semantic
            + self.weights.w_escalation * escalation
            + self.weights.w_downstream * downstream
        )
        return round(max(0.0, min(1.0, total)), 4)

    def explain(
        self,
        node: ContextNode,
        current_step: int,
        context_graph: Any | None = None,
    ) -> ScoringExplanation:
        recency = self._compute_recency_score(node, current_step)
        frequency = self._compute_frequency_score(node)
        centrality = min(1.0, max(0.0, node.graph_centrality))
        task = min(1.0, max(0.0, node.task_relevance))
        semantic = min(1.0, max(0.0, node.semantic_relevance))
        escalation = 1.0 if node.in_escalation else 0.0
        downstream = self._compute_downstream_score(node)

        components = {
            "recency": round(recency, 4),
            "frequency": round(frequency, 4),
            "centrality": round(centrality, 4),
            "task_relevance": round(task, 4),
            "semantic_relevance": round(semantic, 4),
            "escalation_boost": round(escalation, 4),
            "downstream_references": round(downstream, 4),
        }
        total = self.score(node, current_step, context_graph)

        highest_contributor = max(
            components.items(),
            key=lambda item: item[1] * getattr(self.weights, f"w_{item[0].split('_')[0]}", 0.1),
        )
        rationale = (
            f"Node {node.node_id} utility {total:.4f} (primary driver: {highest_contributor[0]} "
            f"at {highest_contributor[1]:.4f})"
        )

        return ScoringExplanation(
            node_id=node.node_id,
            total_score=total,
            component_scores=components,
            rationale=rationale,
        )


__all__ = [
    "ContextNode",
    "ScoringWeights",
    "ScoringExplanation",
    "NodeUtilityScorer",
    "CalibratedKernelScorer",
]
