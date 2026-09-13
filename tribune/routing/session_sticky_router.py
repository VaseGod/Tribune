"""Session-Sticky Attention Routing & Inference Optimization.

Replaces queue-depth dispatching with prefix-hash-aware affinity routing
to maximize warm KV-cache hits and eliminate redundant prompt prefilling.
Tracks WorkerNodeState across distributed workers and provides thread-safe
concurrency lifecycle management.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerNodeState:
    """State tracking for an inference worker node in the execution cluster."""

    node_id: str
    total_vram_gb: float = 24.0
    cached_prefix_hashes: set[str] = field(default_factory=set)
    active_concurrency: int = 0

    def has_prefix(self, prefix_hash: str) -> bool:
        """Check if node's warm KV-cache contains the specified prefix hash."""
        return prefix_hash in self.cached_prefix_hashes

    def register_prefix(self, prefix_hash: str) -> None:
        """Register a prefix hash in the node's warm KV-cache."""
        self.cached_prefix_hashes.add(prefix_hash)

    def evict_prefix(self, prefix_hash: str) -> None:
        """Evict a prefix hash from the node's warm cache."""
        self.cached_prefix_hashes.discard(prefix_hash)


@dataclass(frozen=True)
class RoutingDecision:
    """Result of a session-sticky routing dispatch decision."""

    node_id: str
    prefix_hash: str
    is_cache_hit: bool
    active_concurrency: int
    estimated_prefill_latency_ms: float
    reason: str = ""


class CapacityExceededError(RuntimeError):
    """Raised when all worker nodes have reached their maximum concurrency limit."""

    pass


class SessionStickyRouter:
    """Prefix-hash-aware affinity router for multi-turn inference optimization.

    Computes SHA-256 hashes over prompt prefixes and prioritizes nodes with
    warm KV cache hits. If cache misses occur or warm nodes are saturated,
    balances load to the least-concurrency available node and registers the prefix.
    """

    def __init__(
        self,
        nodes: list[WorkerNodeState] | None = None,
        max_concurrency_per_node: int = 4,
        prefix_window: int = 1024,
        cold_prefill_ms: float = 120.0,
        warm_hit_ms: float = 15.0,
    ) -> None:
        self.max_concurrency_per_node = max_concurrency_per_node
        self.prefix_window = prefix_window
        self.cold_prefill_ms = cold_prefill_ms
        self.warm_hit_ms = warm_hit_ms

        self._nodes: dict[str, WorkerNodeState] = {}
        self._lock = threading.RLock()

        # Telemetry stats
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: WorkerNodeState) -> None:
        """Add or update a worker node in the routing table."""
        with self._lock:
            self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> WorkerNodeState | None:
        """Remove a worker node from the routing table."""
        with self._lock:
            return self._nodes.pop(node_id, None)

    def get_node(self, node_id: str) -> WorkerNodeState | None:
        """Fetch worker node state by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def compute_prefix_hash(self, prompt: str | list[str]) -> str:
        """Compute SHA-256 hash over prompt prefix standardized to first N characters or tokens."""
        if isinstance(prompt, list):
            # Take tokens up to prefix_window
            prefix_text = " ".join(prompt[: self.prefix_window])
        else:
            prefix_text = prompt[: self.prefix_window]

        return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

    def route(
        self,
        prompt: str | list[str],
        prefix: str | list[str] | None = None,
    ) -> RoutingDecision:
        """Route an incoming request to an optimal worker node.

        1. Priority matching: routes to nodes with matching prefix cache hits
           provided active_concurrency < max_concurrency_per_node.
        2. Fallback balancing: on cache misses or warm saturation, routes to the
           least-concurrency available node, registers the prefix, and increments concurrency.
        """
        with self._lock:
            if not self._nodes:
                raise CapacityExceededError("No worker nodes registered in router.")

            self._total_requests += 1
            prefix_to_hash = prefix if prefix is not None else prompt
            prefix_hash = self.compute_prefix_hash(prefix_to_hash)

            # 1. Priority matching: Warm cache hits with available capacity
            warm_candidates = [
                node
                for node in self._nodes.values()
                if node.has_prefix(prefix_hash)
                and node.active_concurrency < self.max_concurrency_per_node
            ]

            if warm_candidates:
                # Select least-loaded warm node
                chosen_node = min(warm_candidates, key=lambda n: n.active_concurrency)
                chosen_node.active_concurrency += 1
                self._cache_hits += 1

                return RoutingDecision(
                    node_id=chosen_node.node_id,
                    prefix_hash=prefix_hash,
                    is_cache_hit=True,
                    active_concurrency=chosen_node.active_concurrency,
                    estimated_prefill_latency_ms=self.warm_hit_ms,
                    reason="WARM_PREFIX_HIT",
                )

            # 2. Fallback balancing: Least concurrency available node
            available_nodes = [
                node
                for node in self._nodes.values()
                if node.active_concurrency < self.max_concurrency_per_node
            ]

            if not available_nodes:
                self._cache_misses += 1
                raise CapacityExceededError(
                    f"All {len(self._nodes)} worker nodes are saturated at "
                    f"max concurrency {self.max_concurrency_per_node}."
                )

            chosen_node = min(available_nodes, key=lambda n: n.active_concurrency)
            chosen_node.register_prefix(prefix_hash)
            chosen_node.active_concurrency += 1
            self._cache_misses += 1

            return RoutingDecision(
                node_id=chosen_node.node_id,
                prefix_hash=prefix_hash,
                is_cache_hit=False,
                active_concurrency=chosen_node.active_concurrency,
                estimated_prefill_latency_ms=self.cold_prefill_ms,
                reason="COLD_FALLBACK_BALANCING",
            )

    def release_turn(self, node_id: str) -> None:
        """Thread-safe resource release: decrements active concurrency on turn completion."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is not None:
                node.active_concurrency = max(0, node.active_concurrency - 1)

    def get_stats(self) -> dict[str, Any]:
        """Return routing telemetry metrics."""
        with self._lock:
            hit_ratio = (
                self._cache_hits / self._total_requests if self._total_requests > 0 else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_ratio": round(hit_ratio, 4),
                "total_nodes": len(self._nodes),
                "active_concurrency_total": sum(n.active_concurrency for n in self._nodes.values()),
            }

    def reset_stats(self) -> None:
        """Reset routing counters."""
        with self._lock:
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
