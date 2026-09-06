"""Entropy-Directed Speculative Scheduler.

Replaces uniform trajectory branching with entropy-directed partial rollouts.
Monitors token-level entropy at runtime; upon encountering high-entropy decision nodes
(evidentiary pruning, cross-examination pivots), captures lightweight state checkpoints,
branches speculative micro-rollouts specifically from that fork, and prunes low-scoring
branches early via verification heuristics.
"""

from __future__ import annotations

import copy
import math
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntropyDecisionNode:
    """A detected decision point where entropy exceeds the speculative threshold."""

    node_id: str
    decision_point: str
    entropy_value: float
    threshold: float
    is_high_entropy: bool
    checkpoint_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeculativeBranch:
    """A speculative micro-rollout branching from a high-entropy decision node."""

    branch_id: str
    parent_node_id: str
    checkpoint_id: str
    candidate_action: str
    arguments: dict[str, Any]
    heuristic_score: float = 0.0
    pruned: bool = False
    execution_result: Any | None = None
    created_at: float = field(default_factory=time.time)
    latency_ms: float = 0.0


class EntropyDirectedScheduler:
    """Entropy-directed scheduler coordinating partial micro-rollouts and early branch pruning."""

    def __init__(
        self,
        entropy_threshold: float = 1.2,
        heuristic_prune_threshold: float = 0.6,
        max_micro_branches: int = 4,
    ) -> None:
        self.entropy_threshold = entropy_threshold
        self.heuristic_prune_threshold = heuristic_prune_threshold
        self.max_micro_branches = max_micro_branches

        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, EntropyDecisionNode] = {}
        self._branches: dict[str, list[SpeculativeBranch]] = {}
        self._lock = threading.RLock()

        # Telemetry
        self.nodes_evaluated = 0
        self.high_entropy_nodes_count = 0
        self.checkpoints_captured = 0
        self.micro_rollouts_dispatched = 0
        self.branches_pruned = 0
        self.estimated_latency_saved_ms = 0.0

    @staticmethod
    def calculate_entropy(distribution: list[float] | dict[str, float] | list[str]) -> float:
        """Calculate Shannon entropy H(X) = -sum(p * log2(p))."""
        if not distribution:
            return 0.0

        if isinstance(distribution, list) and distribution and isinstance(distribution[0], str):
            # Token occurrences list
            counts = Counter(distribution)
            total = len(distribution)
            probs = [c / total for c in counts.values()]
        elif isinstance(distribution, dict):
            # Key -> probability map
            vals = list(distribution.values())
            total = sum(vals) if sum(vals) > 0 else 1.0
            probs = [v / total for v in vals]
        else:
            # Probability list
            vals = list(distribution)  # type: ignore
            total = sum(vals) if sum(vals) > 0 else 1.0
            probs = [v / total for v in vals]

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def evaluate_decision_point(
        self,
        decision_point: str,
        distribution: list[float] | dict[str, float] | list[str],
        state_snapshot: dict[str, Any] | None = None,
        custom_threshold: float | None = None,
    ) -> EntropyDecisionNode:
        """Evaluate token-level / decision entropy at runtime and capture a checkpoint if high."""
        with self._lock:
            self.nodes_evaluated += 1
            threshold = custom_threshold if custom_threshold is not None else self.entropy_threshold
            entropy_val = self.calculate_entropy(distribution)
            is_high = entropy_val >= threshold

            node_id = f"node_{secrets.token_hex(6)}_{decision_point}"
            checkpoint_id = None

            if is_high:
                self.high_entropy_nodes_count += 1
                if state_snapshot is not None:
                    checkpoint_id = self.capture_checkpoint(state_snapshot)

            node = EntropyDecisionNode(
                node_id=node_id,
                decision_point=decision_point,
                entropy_value=entropy_val,
                threshold=threshold,
                is_high_entropy=is_high,
                checkpoint_id=checkpoint_id,
            )
            self._nodes[node_id] = node
            return node

    def capture_checkpoint(self, state_snapshot: dict[str, Any]) -> str:
        """Capture a lightweight state checkpoint specifically at the decision fork."""
        with self._lock:
            checkpoint_id = f"chk_{secrets.token_hex(8)}"
            self._checkpoints[checkpoint_id] = copy.deepcopy(state_snapshot)
            self.checkpoints_captured += 1
            return checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._checkpoints.get(checkpoint_id))

    def branch_micro_rollouts(
        self,
        node: EntropyDecisionNode,
        candidate_actions: list[dict[str, Any]],
        rollout_executor: Callable[[str, dict[str, Any], dict[str, Any]], Any],
    ) -> list[SpeculativeBranch]:
        """Dispatch speculative micro-rollouts specifically from the captured fork."""
        with self._lock:
            if not node.checkpoint_id or node.checkpoint_id not in self._checkpoints:
                raise ValueError(f"No valid state checkpoint found for node {node.node_id}")

            base_state = self._checkpoints[node.checkpoint_id]
            branches: list[SpeculativeBranch] = []

            for i, cand in enumerate(candidate_actions[: self.max_micro_branches]):
                action_name = cand.get("action", f"action_{i}")
                args = cand.get("arguments", {})
                branch_id = f"br_{node.node_id}_{i}_{action_name}"

                start_t = time.perf_counter()
                # Run speculative micro-rollout from the fork
                try:
                    fork_state = copy.deepcopy(base_state)
                    exec_result = rollout_executor(action_name, args, fork_state)
                except Exception as exc:
                    exec_result = {"error": str(exc)}
                lat_ms = (time.perf_counter() - start_t) * 1000.0

                branch = SpeculativeBranch(
                    branch_id=branch_id,
                    parent_node_id=node.node_id,
                    checkpoint_id=node.checkpoint_id,
                    candidate_action=action_name,
                    arguments=args,
                    execution_result=exec_result,
                    latency_ms=lat_ms,
                )
                branches.append(branch)
                self.micro_rollouts_dispatched += 1

            self._branches[node.node_id] = branches
            return branches

    def prune_branches(
        self,
        node_id: str,
        heuristic_scorer: Callable[[SpeculativeBranch], float],
        prune_threshold: float | None = None,
    ) -> list[SpeculativeBranch]:
        """Prune low-scoring branches early based on verification heuristics."""
        with self._lock:
            threshold = (
                prune_threshold if prune_threshold is not None else self.heuristic_prune_threshold
            )
            branches = self._branches.get(node_id, [])

            surviving: list[SpeculativeBranch] = []
            for b in branches:
                score = heuristic_scorer(b)
                b.heuristic_score = score
                if score < threshold:
                    b.pruned = True
                    self.branches_pruned += 1
                else:
                    b.pruned = False
                    surviving.append(b)

            # Sort surviving branches descending by heuristic verification score
            surviving.sort(key=lambda x: x.heuristic_score, reverse=True)
            self.estimated_latency_saved_ms += len(branches) * 15.0  # Savings estimation
            return surviving

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "nodes_evaluated": self.nodes_evaluated,
                "high_entropy_nodes": self.high_entropy_nodes_count,
                "checkpoints_captured": self.checkpoints_captured,
                "micro_rollouts_dispatched": self.micro_rollouts_dispatched,
                "branches_pruned": self.branches_pruned,
                "estimated_latency_saved_ms": self.estimated_latency_saved_ms,
            }


__all__ = [
    "EntropyDecisionNode",
    "SpeculativeBranch",
    "EntropyDirectedScheduler",
]
