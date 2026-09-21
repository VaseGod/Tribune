"""Comprehensive Metrics and Telemetry Layer for Tribune DeepSeek Roadmap.

Tracks:
1. Token accounting (total, cached, uncached, output).
2. Cache hit rate (with safe division).
3. Cost accounting (estimated vs actual).
4. Preparer metrics (prefix hash, prefix stability, ingestion duration).
5. Navigator metrics (step count, trajectory tokens, horizon breaches, verbosity violations).
6. Verifier metrics (failure turns, delta rewrites, full-trajectory regenerations).
7. ToolGrad assertions (pass/fail counts).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .providers.deepseek import DeepSeekCostCalculator

logger = logging.getLogger(__name__)


@dataclass
class TribuneRoadmapMetrics:
    """Consolidated telemetry and validation metrics."""

    total_input_tokens: int = 0
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_rate: float = 0.0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    provider_name: str = "deepseek:deepseek-flash"
    model_name: str = "deepseek-flash"
    task_type: str = "ingestion"
    ingestion_duration_ms: float = 0.0
    preparer_prefix_hash: str = ""
    prefix_stability_flag: bool = True
    navigator_step_count: int = 0
    navigator_trajectory_token_count: int = 0
    horizon_breach_count: int = 0
    verbosity_violation_count: int = 0
    verifier_failure_turn_count: int = 0
    verifier_delta_rewrite_count: int = 0
    full_trajectory_regeneration_count: int = 0
    tool_assertion_pass_count: int = 0
    tool_assertion_fail_count: int = 0

    # Roadmap Modernization Subsystems Telemetry
    edge_classification_latency_ms: float = 0.0
    edge_decision_confidence: float = 0.0
    edge_escalation_count: int = 0
    edge_malformed_count: int = 0
    memory_node_count: int = 0
    memory_token_estimate: int = 0
    memory_eviction_count: int = 0
    average_node_utility_score: float = 0.0
    audio_turn_latency_ms: float = 0.0
    audio_flush_latency_ms: float = 0.0
    audio_partial_count: int = 0
    audio_tool_call_count: int = 0
    adapter_active_expert_count: int = 0
    adapter_routing_latency_ms: float = 0.0
    adapter_estimated_kv_cache_mb: float = 0.0
    adapter_cross_modal_interference: float = 0.0
    sandbox_shell_tokens: int = 0
    sandbox_exploit_traps: int = 0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class RoadmapMetricsCollector:
    """Thread-safe collector aggregating system-wide metrics across all roadmap components."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.metrics = TribuneRoadmapMetrics()
        self.events: list[dict[str, Any]] = []

    def record_provider_call(
        self,
        provider_name: str,
        model_name: str,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        estimated_cost: float | None = None,
        actual_cost: float | None = None,
    ) -> None:
        self.metrics.provider_name = provider_name
        self.metrics.model_name = model_name
        self.metrics.task_type = task_type
        self.metrics.total_input_tokens += input_tokens
        self.metrics.cached_input_tokens += cached_tokens
        self.metrics.uncached_input_tokens += max(0, input_tokens - cached_tokens)
        self.metrics.output_tokens += output_tokens

        # Safe division for cache hit rate
        self.metrics.cache_hit_rate = DeepSeekCostCalculator.compute_cache_hit_rate(
            self.metrics.cached_input_tokens, self.metrics.total_input_tokens
        )

        cost = (
            estimated_cost
            if estimated_cost is not None
            else DeepSeekCostCalculator.calculate_cost(
                uncached_input_tokens=self.metrics.uncached_input_tokens,
                cached_input_tokens=self.metrics.cached_input_tokens,
                output_tokens=self.metrics.output_tokens,
            )
        )
        self.metrics.estimated_cost_usd = round(cost, 8)
        if actual_cost is not None:
            self.metrics.actual_cost_usd = round(actual_cost, 8)

        event = {
            "type": "provider_call",
            "provider": provider_name,
            "model": model_name,
            "task_type": task_type,
            "in": input_tokens,
            "out": output_tokens,
            "cached": cached_tokens,
            "cost": self.metrics.estimated_cost_usd,
        }
        self.events.append(event)
        logger.info(f"[ROADMAP_METRICS] {event}")

    def record_preparer_ingestion(
        self,
        prefix_hash: str,
        duration_ms: float,
        is_stable: bool = True,
    ) -> None:
        self.metrics.preparer_prefix_hash = prefix_hash
        self.metrics.ingestion_duration_ms = round(duration_ms, 2)
        self.metrics.prefix_stability_flag = is_stable

        event = {
            "type": "preparer_ingestion",
            "prefix_hash": prefix_hash,
            "duration_ms": duration_ms,
            "is_stable": is_stable,
        }
        self.events.append(event)
        logger.info(f"[ROADMAP_METRICS] {event}")

    def record_navigator_step(
        self,
        step_count: int,
        trajectory_tokens: int,
        horizon_breached: bool = False,
        verbosity_violation: bool = False,
    ) -> None:
        self.metrics.navigator_step_count = step_count
        self.metrics.navigator_trajectory_token_count = trajectory_tokens
        if horizon_breached:
            self.metrics.horizon_breach_count += 1
        if verbosity_violation:
            self.metrics.verbosity_violation_count += 1

        event = {
            "type": "navigator_step",
            "step_count": step_count,
            "trajectory_tokens": trajectory_tokens,
            "horizon_breached": horizon_breached,
            "verbosity_violation": verbosity_violation,
        }
        self.events.append(event)

    def record_verifier_action(
        self,
        failure_turn_detected: bool = False,
        delta_rewrite_applied: bool = False,
        full_trajectory_regenerated: bool = False,
    ) -> None:
        if failure_turn_detected:
            self.metrics.verifier_failure_turn_count += 1
        if delta_rewrite_applied:
            self.metrics.verifier_delta_rewrite_count += 1
        if full_trajectory_regenerated:
            self.metrics.full_trajectory_regeneration_count += 1

        event = {
            "type": "verifier_action",
            "failure_turn_detected": failure_turn_detected,
            "delta_rewrite_applied": delta_rewrite_applied,
            "full_trajectory_regenerated": full_trajectory_regenerated,
        }
        self.events.append(event)

    def record_tool_assertion(self, passed: bool) -> None:
        if passed:
            self.metrics.tool_assertion_pass_count += 1
        else:
            self.metrics.tool_assertion_fail_count += 1

        event = {"type": "tool_assertion", "passed": passed}
        self.events.append(event)

    def record_context_edge_decision(
        self,
        latency_ms: float,
        confidence: float,
        escalated: bool = False,
        malformed: bool = False,
        edge_class: str = "",
        correlation_id: str = "",
    ) -> None:
        self.metrics.edge_classification_latency_ms = round(latency_ms, 3)
        self.metrics.edge_decision_confidence = round(confidence, 4)
        if escalated:
            self.metrics.edge_escalation_count += 1
        if malformed:
            self.metrics.edge_malformed_count += 1

        event = {
            "type": "context_edge_decision",
            "latency_ms": latency_ms,
            "confidence": confidence,
            "escalated": escalated,
            "edge_class": edge_class,
            "correlation_id": correlation_id,
        }
        self.events.append(event)

    def record_memory_eviction(
        self,
        node_count: int,
        token_estimate: int,
        eviction_count: int,
        average_utility: float,
        correlation_id: str = "",
    ) -> None:
        self.metrics.memory_node_count = node_count
        self.metrics.memory_token_estimate = token_estimate
        self.metrics.memory_eviction_count = eviction_count
        self.metrics.average_node_utility_score = round(average_utility, 4)

        event = {
            "type": "memory_eviction",
            "node_count": node_count,
            "token_estimate": token_estimate,
            "eviction_count": eviction_count,
            "average_utility": average_utility,
            "correlation_id": correlation_id,
        }
        self.events.append(event)

    def record_acoustic_turn(
        self,
        turn_latency_ms: float,
        flush_latency_ms: float = 0.0,
        partial_count: int = 0,
        tool_call_count: int = 0,
        correlation_id: str = "",
    ) -> None:
        self.metrics.audio_turn_latency_ms = round(turn_latency_ms, 2)
        self.metrics.audio_flush_latency_ms = round(flush_latency_ms, 2)
        self.metrics.audio_partial_count += partial_count
        self.metrics.audio_tool_call_count += tool_call_count

        event = {
            "type": "acoustic_turn",
            "turn_latency_ms": turn_latency_ms,
            "flush_latency_ms": flush_latency_ms,
            "correlation_id": correlation_id,
        }
        self.events.append(event)

    def record_adapter_routing(
        self,
        active_experts: int,
        routing_latency_ms: float,
        estimated_kv_cache_mb: float,
        cross_modal_interference: float = 0.0,
        correlation_id: str = "",
    ) -> None:
        self.metrics.adapter_active_expert_count = active_experts
        self.metrics.adapter_routing_latency_ms = round(routing_latency_ms, 3)
        self.metrics.adapter_estimated_kv_cache_mb = round(estimated_kv_cache_mb, 2)
        self.metrics.adapter_cross_modal_interference = round(cross_modal_interference, 4)

        event = {
            "type": "adapter_routing",
            "active_experts": active_experts,
            "routing_latency_ms": routing_latency_ms,
            "kv_cache_mb": estimated_kv_cache_mb,
            "interference": cross_modal_interference,
            "correlation_id": correlation_id,
        }
        self.events.append(event)

    def record_sandbox_execution(
        self,
        tokens_consumed: int,
        exploit_trapped: bool = False,
        correlation_id: str = "",
    ) -> None:
        self.metrics.sandbox_shell_tokens += tokens_consumed
        if exploit_trapped:
            self.metrics.sandbox_exploit_traps += 1

        event = {
            "type": "sandbox_execution",
            "tokens_consumed": tokens_consumed,
            "exploit_trapped": exploit_trapped,
            "correlation_id": correlation_id,
        }
        self.events.append(event)

    def get_snapshot(self) -> TribuneRoadmapMetrics:
        return self.metrics


_GLOBAL_COLLECTOR = RoadmapMetricsCollector()


def get_roadmap_metrics() -> RoadmapMetricsCollector:
    return _GLOBAL_COLLECTOR


def reset_roadmap_metrics() -> None:
    _GLOBAL_COLLECTOR.reset()


__all__ = [
    "TribuneRoadmapMetrics",
    "RoadmapMetricsCollector",
    "get_roadmap_metrics",
    "reset_roadmap_metrics",
]
