"""Real-Time Token & Cost Telemetry Middleware and Logger.

Intercepts every LLM completion response, calculates exact USD cost based on model
pricing tables via CostModel, tracks input/output/cached tokens, latency, and exposes
operational metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..eval.costmodel import CostModel
from ..providers.llm_client import LLMCompletionRequest, LLMCompletionResponse, LLMProvider
from ..types import ModelCallUsage
from . import tracing

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    timestamp: str
    provider_name: str
    model: str
    role: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    latency_ms: float
    cost_usd: float
    backend_id: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryMetricsStore:
    """In-memory telemetry store accumulating query records and cost totals."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel.load()
        self.records: list[TelemetryRecord] = []

    def record_call(
        self,
        provider_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
        role: str = "general",
        accounting_date: date | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TelemetryRecord:
        on_date = accounting_date or date.today()

        call_usage = ModelCallUsage(
            role=role,
            model=model,
            tokenizer_id=model,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cache_read_tokens=cached_tokens,
        )

        cost_usd, backend_id = self.cost_model.cost_of_call(call_usage, on_date)

        rec = TelemetryRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider_name=provider_name,
            model=model,
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            backend_id=backend_id,
            extra_metadata=extra_metadata or {},
        )
        self.records.append(rec)

        # Log via structured tracing
        tracing.log(
            "llm_telemetry_call",
            provider=provider_name,
            model=model,
            role=role,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            tokens_cached=cached_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost_usd,
            backend_id=backend_id,
        )

        logger.info(
            f"[TELEMETRY] call={provider_name}:{model} role={role} "
            f"in={input_tokens} out={output_tokens} cached={cached_tokens} "
            f"lat={latency_ms:.1f}ms cost=${cost_usd:.6f}"
        )
        return rec

    def get_summary(self) -> dict[str, Any]:
        total_calls = len(self.records)
        total_input_tokens = sum(r.input_tokens for r in self.records)
        total_output_tokens = sum(r.output_tokens for r in self.records)
        total_cached_tokens = sum(r.cached_tokens for r in self.records)
        total_cost_usd = sum(r.cost_usd for r in self.records)
        avg_latency_ms = (
            sum(r.latency_ms for r in self.records) / total_calls if total_calls > 0 else 0.0
        )

        by_provider: dict[str, dict[str, Any]] = {}
        for r in self.records:
            if r.provider_name not in by_provider:
                by_provider[r.provider_name] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            by_provider[r.provider_name]["calls"] += 1
            by_provider[r.provider_name]["cost_usd"] += r.cost_usd
            by_provider[r.provider_name]["tokens"] += r.input_tokens + r.output_tokens

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_tokens": total_cached_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "by_provider": by_provider,
        }

    def clear(self) -> None:
        self.records.clear()


# Global default telemetry store instance
GLOBAL_TELEMETRY_STORE = TelemetryMetricsStore()


class CostTelemetryMiddleware:
    """Middleware wrapper that wraps an LLMProvider and records real-time cost telemetry."""

    def __init__(
        self,
        provider: LLMProvider,
        store: TelemetryMetricsStore | None = None,
        role: str = "general",
    ) -> None:
        self.provider = provider
        self.name = provider.name
        self.store = store or GLOBAL_TELEMETRY_STORE
        self.role = role

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        resp = self.provider.complete(request)
        self.store.record_call(
            provider_name=resp.provider_name,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cached_tokens=resp.cached_tokens,
            latency_ms=resp.latency_ms,
            role=self.role,
        )
        return resp
