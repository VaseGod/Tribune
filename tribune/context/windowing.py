"""Sliding Window Management and Continuous Utility Ranking Engine.

Maintains sliding-window active context buffers with calibrated utility ranking,
continuous node eviction, token budget compliance, and non-blocking summarization hooks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .scoring import CalibratedKernelScorer, ContextNode, NodeUtilityScorer, ScoringExplanation

logger = logging.getLogger(__name__)


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window context management."""

    window_size: int = 2048  # Maximum active tokens in primary window
    token_budget: int = 32000  # Total session token budget ceiling
    eviction_threshold: float = 0.25  # Utility score below which nodes are eligible for eviction
    decay_curve: str = "exponential"  # "exponential" | "linear" | "sigmoid"
    decay_half_life_steps: float = 20.0
    summarization_hook_enabled: bool = True
    compression_enabled: bool = True


@dataclass
class EvictionRecord:
    """Documenting an eviction event with explainability rationale."""

    node_id: str
    token_count: int
    utility_score: float
    explanation: ScoringExplanation
    evicted_at_step: int
    summarized: bool = False
    summary_text: str = ""


class SlidingWindowEngine:
    """Engine executing continuous node ranking and calibrated eviction across sliding windows."""

    def __init__(
        self,
        config: SlidingWindowConfig | None = None,
        scorer: NodeUtilityScorer | None = None,
        summarize_hook: Callable[[list[ContextNode]], dict[str, str]] | None = None,
    ) -> None:
        self.config = config or SlidingWindowConfig()
        self.scorer = scorer or CalibratedKernelScorer(
            decay_curve=self.config.decay_curve,
            decay_half_life_steps=self.config.decay_half_life_steps,
        )
        self.summarize_hook = summarize_hook

        self._nodes: dict[str, ContextNode] = {}
        self._current_step: int = 0
        self._eviction_history: list[EvictionRecord] = []
        self._total_evicted_tokens: int = 0

    @property
    def current_step(self) -> int:
        return self._current_step

    def advance_step(self) -> int:
        self._current_step += 1
        return self._current_step

    def register_node(self, node: ContextNode) -> None:
        """Register or update a context node in the active sliding window."""
        self._nodes[node.node_id] = node

    def touch_node(self, node_id: str) -> None:
        """Record an access/reference to a node, resetting recency and boosting frequency."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.last_accessed_step = self._current_step
            node.access_count += 1

    def active_nodes(self) -> list[ContextNode]:
        """Return non-evicted nodes currently in active context."""
        return [n for n in self._nodes.values() if not n.is_evicted]

    def current_token_count(self) -> int:
        """Calculate total tokens consumed by active nodes."""
        return sum(n.token_count for n in self.active_nodes())

    def rank_nodes(self) -> list[tuple[ContextNode, float, ScoringExplanation]]:
        """Rank active nodes by calibrated utility score ascending (lowest utility first)."""
        scored: list[tuple[ContextNode, float, ScoringExplanation]] = []
        for node in self.active_nodes():
            score = self.scorer.score(node, self._current_step)
            explanation = self.scorer.explain(node, self._current_step)
            scored.append((node, score, explanation))

        # Sort ascending by score: lowest utility candidates appear first
        scored.sort(key=lambda x: (x[1], -x[0].last_accessed_step, x[0].node_id))
        return scored

    def evict_to_budget(
        self,
        target_token_budget: int | None = None,
    ) -> list[EvictionRecord]:
        """Evict lowest-utility nodes until active context falls within the token budget.

        Eviction decisions are explainable, logged, and deterministic.
        """
        budget = target_token_budget if target_token_budget is not None else self.config.window_size
        evicted_records: list[EvictionRecord] = []

        active_tokens = self.current_token_count()
        if active_tokens <= budget:
            return evicted_records

        ranked = self.rank_nodes()
        nodes_to_evict: list[tuple[ContextNode, float, ScoringExplanation]] = []

        for node, score, explanation in ranked:
            if active_tokens <= budget:
                break
            nodes_to_evict.append((node, score, explanation))
            active_tokens -= node.token_count

        # Mark nodes as evicted and prepare records
        for node, score, explanation in nodes_to_evict:
            node.is_evicted = True
            rec = EvictionRecord(
                node_id=node.node_id,
                token_count=node.token_count,
                utility_score=score,
                explanation=explanation,
                evicted_at_step=self._current_step,
            )
            self._eviction_history.append(rec)
            self._total_evicted_tokens += node.token_count
            evicted_records.append(rec)

            logger.info(
                f"[SlidingWindowEngine] Evicted node {node.node_id} ({node.token_count} tokens) "
                f"with utility score {score:.4f}: {explanation.rationale}"
            )

        # Non-blocking summarization hook
        if self.config.summarization_hook_enabled and self.summarize_hook and nodes_to_evict:
            try:
                summaries = self.summarize_hook([item[0] for item in nodes_to_evict])
                for rec in evicted_records:
                    if rec.node_id in summaries:
                        rec.summarized = True
                        rec.summary_text = summaries[rec.node_id]
                        self._nodes[rec.node_id].summary = rec.summary_text
            except Exception as exc:
                logger.warning(f"[SlidingWindowEngine] Summarization hook failed (non-blocking): {exc}")

        return evicted_records

    def stats(self) -> dict[str, Any]:
        active = self.active_nodes()
        tokens = sum(n.token_count for n in active)
        mean_utility = (
            sum(self.scorer.score(n, self._current_step) for n in active) / len(active)
            if active
            else 0.0
        )
        return {
            "current_step": self._current_step,
            "active_node_count": len(active),
            "active_token_count": tokens,
            "evicted_node_count": len(self._eviction_history),
            "total_evicted_tokens": self._total_evicted_tokens,
            "average_node_utility_score": round(mean_utility, 4),
            "token_budget": self.config.window_size,
        }


__all__ = [
    "SlidingWindowConfig",
    "EvictionRecord",
    "SlidingWindowEngine",
]
