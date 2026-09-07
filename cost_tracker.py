"""Asymmetric Prompt Caching Telemetry & Dynamic Cost Accounting.

Records granular token buckets and calculates exact pricing with asymmetric
rates for prompt caching and fallback model routing:
- cache_read_input_tokens: $0.25 / 1M tokens
- cache_creation_input_tokens: $12.50 / 1M tokens
- uncached_input_tokens: $10.00 / 1M tokens
- output_tokens: $15.00 / 1M tokens (configurable)

Formula:
Cost = (Tokens_read * $0.25/M) + (Tokens_write * $12.50/M) + (Tokens_uncached * $10.00/M) + (Tokens_output * Rate_out)
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any

# Standard base rates per 1,000,000 tokens
DEFAULT_CACHE_READ_PER_M: float = 0.25
DEFAULT_CACHE_WRITE_PER_M: float = 12.50
DEFAULT_UNCACHED_INPUT_PER_M: float = 10.00
DEFAULT_OUTPUT_PER_M: float = 15.00

# Legacy Opus fallback rates per 1,000,000 tokens
OPUS_FALLBACK_CACHE_READ_PER_M: float = 1.50
OPUS_FALLBACK_CACHE_WRITE_PER_M: float = 18.75
OPUS_FALLBACK_UNCACHED_INPUT_PER_M: float = 15.00
OPUS_FALLBACK_OUTPUT_PER_M: float = 75.00


@dataclass
class PricingSchedule:
    """Configurable rate schedule per million tokens."""

    cache_read_per_m: float = DEFAULT_CACHE_READ_PER_M
    cache_write_per_m: float = DEFAULT_CACHE_WRITE_PER_M
    uncached_input_per_m: float = DEFAULT_UNCACHED_INPUT_PER_M
    output_per_m: float = DEFAULT_OUTPUT_PER_M
    active_model: str = "claude-3-5-sonnet"
    is_fallback: bool = False

    def compute_cost(
        self,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        uncached_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate total USD cost according to the pricing arithmetic."""
        cost = (
            (cache_read_tokens * (self.cache_read_per_m / 1_000_000.0))
            + (cache_creation_tokens * (self.cache_write_per_m / 1_000_000.0))
            + (uncached_tokens * (self.uncached_input_per_m / 1_000_000.0))
            + (output_tokens * (self.output_per_m / 1_000_000.0))
        )
        return round(cost, 8)


@dataclass
class GranularTokenUsage:
    """Granular token buckets recorded per call or turn."""

    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return (
            self.cache_read_input_tokens
            + self.cache_creation_input_tokens
            + self.uncached_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def add(self, other: GranularTokenUsage) -> None:
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.uncached_input_tokens += other.uncached_input_tokens
        self.output_tokens += other.output_tokens


class CostTracker:
    """Thread-safe dynamic cost accounting and granular token tracker."""

    def __init__(self, pricing: PricingSchedule | None = None) -> None:
        self.pricing = pricing or PricingSchedule()
        self.usage = GranularTokenUsage()
        self.call_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def record_usage(
        self,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        uncached_input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Record a model round-trip token usage and return the incurred USD cost."""
        with self._lock:
            call_usage = GranularTokenUsage(
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                uncached_input_tokens=uncached_input_tokens,
                output_tokens=output_tokens,
            )
            cost = self.pricing.compute_cost(
                cache_read_tokens=cache_read_input_tokens,
                cache_creation_tokens=cache_creation_input_tokens,
                uncached_tokens=uncached_input_tokens,
                output_tokens=output_tokens,
            )
            self.usage.add(call_usage)
            self.call_history.append({
                "usage": copy.deepcopy(call_usage),
                "cost_usd": cost,
                "pricing": copy.deepcopy(self.pricing),
                "metadata": metadata or {},
            })
            return cost

    def set_fallback_pricing(
        self,
        fallback_model: str = "claude-3-opus",
        cache_read_per_m: float = OPUS_FALLBACK_CACHE_READ_PER_M,
        cache_write_per_m: float = OPUS_FALLBACK_CACHE_WRITE_PER_M,
        uncached_input_per_m: float = OPUS_FALLBACK_UNCACHED_INPUT_PER_M,
        output_per_m: float = OPUS_FALLBACK_OUTPUT_PER_M,
    ) -> None:
        """Dynamically adjust pricing arithmetic when upstream model fallback is detected."""
        with self._lock:
            self.pricing.active_model = fallback_model
            self.pricing.is_fallback = True
            self.pricing.cache_read_per_m = cache_read_per_m
            self.pricing.cache_write_per_m = cache_write_per_m
            self.pricing.uncached_input_per_m = uncached_input_per_m
            self.pricing.output_per_m = output_per_m

    def adjust_pricing(
        self,
        cache_read_per_m: float | None = None,
        cache_write_per_m: float | None = None,
        uncached_input_per_m: float | None = None,
        output_per_m: float | None = None,
    ) -> None:
        """Update individual unit rates."""
        with self._lock:
            if cache_read_per_m is not None:
                self.pricing.cache_read_per_m = cache_read_per_m
            if cache_write_per_m is not None:
                self.pricing.cache_write_per_m = cache_write_per_m
            if uncached_input_per_m is not None:
                self.pricing.uncached_input_per_m = uncached_input_per_m
            if output_per_m is not None:
                self.pricing.output_per_m = output_per_m

    def total_cost(self) -> float:
        """Calculate total USD cost accumulated across all recorded turns."""
        with self._lock:
            return self.pricing.compute_cost(
                cache_read_tokens=self.usage.cache_read_input_tokens,
                cache_creation_tokens=self.usage.cache_creation_input_tokens,
                uncached_tokens=self.usage.uncached_input_tokens,
                output_tokens=self.usage.output_tokens,
            )

    def summary(self) -> dict[str, Any]:
        """Produce a comprehensive summary of token usage and dynamic cost breakdown."""
        with self._lock:
            return {
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "uncached_input_tokens": self.usage.uncached_input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_input_tokens": self.usage.total_input_tokens,
                "total_tokens": self.usage.total_tokens,
                "total_cost_usd": self.total_cost(),
                "active_model": self.pricing.active_model,
                "is_fallback": self.pricing.is_fallback,
                "rates": {
                    "cache_read_per_m": self.pricing.cache_read_per_m,
                    "cache_write_per_m": self.pricing.cache_write_per_m,
                    "uncached_input_per_m": self.pricing.uncached_input_per_m,
                    "output_per_m": self.pricing.output_per_m,
                },
                "total_calls": len(self.call_history),
            }

    def reset(self) -> None:
        """Reset usage and history."""
        with self._lock:
            self.usage = GranularTokenUsage()
            self.call_history.clear()


# Global default tracker
_DEFAULT_COST_TRACKER = CostTracker()


def get_default_cost_tracker() -> CostTracker:
    return _DEFAULT_COST_TRACKER


__all__ = [
    "DEFAULT_CACHE_READ_PER_M",
    "DEFAULT_CACHE_WRITE_PER_M",
    "DEFAULT_UNCACHED_INPUT_PER_M",
    "DEFAULT_OUTPUT_PER_M",
    "OPUS_FALLBACK_CACHE_READ_PER_M",
    "OPUS_FALLBACK_CACHE_WRITE_PER_M",
    "OPUS_FALLBACK_UNCACHED_INPUT_PER_M",
    "OPUS_FALLBACK_OUTPUT_PER_M",
    "PricingSchedule",
    "GranularTokenUsage",
    "CostTracker",
    "get_default_cost_tracker",
]
