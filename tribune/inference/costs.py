"""Cost Accounting & Token Budgeting Helpers.

Implements standard pricing models for:
- Input uncached tokens
- Input prompt-cached tokens
- Output completion tokens
- Model-specific rate schedules and dynamic budget tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import ProviderUsage

logger = logging.getLogger(__name__)

# Default standard baseline rate card ($/1M tokens)
# Reflects DeepSeek-V4.1-Flash economics and frontier reference rates
DEFAULT_RATE_CARD: dict[str, dict[str, float]] = {
    # Worker tier / Ingestion (DeepSeek-V4.1-Flash)
    "deepseek-flash": {
        "input_uncached": 0.30,
        "input_cached": 0.006,
        "output": 1.20,
    },
    "deepseek-chat": {
        "input_uncached": 0.30,
        "input_cached": 0.006,
        "output": 1.20,
    },
    "deepseek-v4.1-flash": {
        "input_uncached": 0.30,
        "input_cached": 0.006,
        "output": 1.20,
    },
    "swift-qwen3.8-27b": {
        "input_uncached": 0.20,
        "input_cached": 0.02,
        "output": 0.80,
    },
    # Open-weight local serving (effective amortized compute)
    "vllm": {
        "input_uncached": 0.15,
        "input_cached": 0.015,
        "output": 0.60,
    },
    "local_rules": {
        "input_uncached": 0.0,
        "input_cached": 0.0,
        "output": 0.0,
    },
    # Frontier / Lead tier (Astra / Claude Sonnet / GPT-4o)
    "frontier": {
        "input_uncached": 3.00,
        "input_cached": 0.75,
        "output": 15.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input_uncached": 3.00,
        "input_cached": 0.30,
        "output": 15.00,
    },
    "gpt-4o": {
        "input_uncached": 2.50,
        "input_cached": 1.25,
        "output": 10.00,
    },
    "default": {
        "input_uncached": 1.00,
        "input_cached": 0.25,
        "output": 4.00,
    },
}


@dataclass
class CostEstimate:
    """Detailed cost breakdown for an inference call or run."""

    uncached_input_usd: float
    cached_input_usd: float
    output_usd: float
    total_cost_usd: float
    cache_savings_usd: float
    model_rate_key: str


def calculate_token_cost(
    model: str,
    usage: ProviderUsage,
    rate_card: dict[str, dict[str, float]] | None = None,
) -> CostEstimate:
    """Calculate exact USD cost from ProviderUsage and model rate card."""
    rates = rate_card or DEFAULT_RATE_CARD
    normalized = model.lower()

    # Find closest matching rate
    matched_rates = rates.get("default", DEFAULT_RATE_CARD["default"])
    matched_key = "default"

    for key, r in rates.items():
        if key in normalized:
            matched_rates = r
            matched_key = key
            break

    rate_uncached = matched_rates.get("input_uncached", 1.0) / 1_000_000.0
    rate_cached = matched_rates.get("input_cached", rate_uncached) / 1_000_000.0
    rate_out = matched_rates.get("output", 4.0) / 1_000_000.0

    uncached_input_tokens = max(0, usage.prompt_tokens - usage.cached_tokens)
    cached_input_tokens = usage.cached_tokens
    output_tokens = usage.completion_tokens

    cost_uncached = uncached_input_tokens * rate_uncached
    cost_cached = cached_input_tokens * rate_cached
    cost_output = output_tokens * rate_out
    total = cost_uncached + cost_cached + cost_output

    # Calculate savings if cached tokens had been charged uncached
    baseline_uncached_cost = (uncached_input_tokens + cached_input_tokens) * rate_uncached + cost_output
    savings = max(0.0, baseline_uncached_cost - total)

    return CostEstimate(
        uncached_input_usd=round(cost_uncached, 6),
        cached_input_usd=round(cost_cached, 6),
        output_usd=round(cost_output, 6),
        total_cost_usd=round(total, 6),
        cache_savings_usd=round(savings, 6),
        model_rate_key=matched_key,
    )


class ReasoningBudgetTracker:
    """Enforces per-task or per-run computational and financial budget caps."""

    def __init__(
        self,
        max_cost_usd: float = 0.50,
        max_total_tokens: int = 200_000,
        fail_closed: bool = True,
    ) -> None:
        self.max_cost_usd = max_cost_usd
        self.max_total_tokens = max_total_tokens
        self.fail_closed = fail_closed
        self.accumulated_cost_usd: float = 0.0
        self.accumulated_tokens: int = 0

    def record_usage(self, usage: ProviderUsage, cost_usd: float) -> None:
        self.accumulated_cost_usd += cost_usd
        self.accumulated_tokens += usage.total_tokens

    def is_exceeded(self) -> tuple[bool, str]:
        if self.accumulated_cost_usd > self.max_cost_usd:
            return True, f"Cost cap exceeded: ${self.accumulated_cost_usd:.4f} > ${self.max_cost_usd:.4f}"
        if self.accumulated_tokens > self.max_total_tokens:
            return True, f"Token cap exceeded: {self.accumulated_tokens} > {self.max_total_tokens}"
        return False, ""
