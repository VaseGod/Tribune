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
