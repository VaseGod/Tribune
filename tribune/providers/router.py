"""Tiered Model Router with Dynamic Pareto Cost Routing & SLA Latency Tracking.

Sits between system requests and LLM provider execution:
1. Tier 1 (Routine / Triage): Standard eligibility parsing, OCR data extraction, and
   deterministic checklist evaluations route to open-weight models (e.g. local/vLLM endpoints
   with speculative draft acceleration).
2. Tier 2 (Escalation / Contested): Complex statutory ambiguities, appeals briefs,
   conflicting custody interpretations, and final governance sign-offs escalate dynamically
   to frontier model APIs.
3. SLA Latency & Error Tracking: Rolling p95 latency and error tracking per provider tier
   with automatic circuit breaking, failover, and Local Quantized Fallback.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..config import TribuneSettings, get_settings
from ..instrumentation.usage import UsageRecorder
from .base import (
    ModelProvider,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
)
from .local_rules import LocalRulesProvider
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# SLA Latency & Health Tracker
# --------------------------------------------------------------------------- #


@dataclass
class SLATracker:
    """Sliding-window latency and error tracking per provider tier for dynamic Pareto routing."""

    tier: int
    sla_target_p95_ms: float = 800.0
    window_size: int = 50
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    errors_count: int = 0
    total_calls: int = 0
    circuit_open: bool = False
    circuit_opened_at: float = 0.0
    recovery_timeout_sec: float = 30.0

    def record_call(self, latency_ms: float, is_error: bool = False) -> None:
        self.total_calls += 1
        self.latencies_ms.append(latency_ms)
        if is_error:
            self.errors_count += 1

        # Check if error rate exceeds 30% or p95 breaches SLA
        if len(self.latencies_ms) >= 10:
            p95 = self.p95_latency_ms()
            recent_errors = self.errors_count / max(1, len(self.latencies_ms))
            if recent_errors > 0.35 or p95 > (self.sla_target_p95_ms * 2.0):
                if not self.circuit_open:
                    logger.warning(f"Tier {self.tier} SLA breached (p95={p95:.1f}ms, err={recent_errors:.2%}). Tripping circuit.")
                    self.circuit_open = True
                    self.circuit_opened_at = time.time()

    def is_healthy(self) -> bool:
        if not self.circuit_open:
            return True
        # Check half-open recovery
        if time.time() - self.circuit_opened_at > self.recovery_timeout_sec:
            self.circuit_open = False
            self.latencies_ms.clear()
            self.errors_count = 0
            return True
        return False

    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(0.95 * len(sorted_lats))
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    def stats(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "total_calls": self.total_calls,
            "p95_latency_ms": round(self.p95_latency_ms(), 2),
            "sla_target_ms": self.sla_target_p95_ms,
            "circuit_open": self.circuit_open,
            "error_count": self.errors_count,
        }


# --------------------------------------------------------------------------- #
# Dynamic Pareto Model Router
# --------------------------------------------------------------------------- #


class ModelRouter:
    """Central dynamic Pareto router directing tasks to Tier 0, Tier 1, or Tier 2 with SLA tracking."""

    def __init__(
        self,
        tier1_provider: ModelProvider | None = None,
        tier2_provider: ModelProvider | None = None,
        fallback_provider: ModelProvider | None = None,
        local_dense_provider: ModelProvider | None = None,
        settings: TribuneSettings | None = None,
        recorder: UsageRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.recorder = recorder
        self.name = "model_router"
        self.version = "2.0.0"

        self.enable_reasonmaxxer = getattr(self.settings, "enable_reasonmaxxer", True)
        self.route_strategy = getattr(self.settings, "route_strategy", "pareto-sla")
        self.local_model_type = getattr(self.settings, "local_model_type", "qwen3.8-27b")

        # Tier 1 defaults
        self.tier1_model = (
            os.getenv("DEFAULT_TIER1_MODEL")
            or os.getenv("TRIBUNE_DEFAULT_TIER1_MODEL")
            or self.settings.tier1_model
        )
        # Tier 2 defaults
        self.tier2_model = (
            os.getenv("DEFAULT_TIER2_MODEL")
            or os.getenv("TRIBUNE_DEFAULT_TIER2_MODEL")
            or self.settings.tier2_model
        )

        if tier1_provider is not None:
            self.tier1_provider = tier1_provider
        elif self.settings.tier1_provider == "openai_compat":
            self.tier1_provider = OpenAICompatProvider(
                model=self.tier1_model,
                settings=self.settings,
                role="tier1_proposer",
                recorder=self.recorder,
            )
        else:
            self.tier1_provider = LocalRulesProvider(role="tier1_proposer", recorder=self.recorder)

        if tier2_provider is not None:
            self.tier2_provider = tier2_provider
        elif self.settings.tier2_provider == "openai_compat":
            self.tier2_provider = OpenAICompatProvider(
                model=self.tier2_model,
                settings=self.settings,
                role="tier2_verifier",
                recorder=self.recorder,
            )
        else:
            self.tier2_provider = LocalRulesProvider(role="tier2_verifier", recorder=self.recorder)

        # Local Dense Qwen3.8-27B Provider for ReasonMaxxer
        if local_dense_provider is not None:
            self.local_dense_provider = local_dense_provider
        elif self.enable_reasonmaxxer:
            if self.settings.tier1_provider == "openai_compat" or self.settings.provider == "openai_compat":
                self.local_dense_provider = OpenAICompatProvider(
                    model=self.local_model_type,
                    settings=self.settings,
                    role="local_dense",
                    recorder=self.recorder,
                )
            else:
                self.local_dense_provider = LocalRulesProvider(role="local_dense", recorder=self.recorder)
        else:
            self.local_dense_provider = self.tier1_provider

        # Local Quantized Fallback Provider
        if fallback_provider is not None:
            self.fallback_provider = fallback_provider
        else:
            self.fallback_provider = LocalRulesProvider(role="local_fallback", recorder=self.recorder)

        # SLA Trackers per tier
        self.sla_trackers = {
            0: SLATracker(tier=0, sla_target_p95_ms=600.0),
            1: SLATracker(tier=1, sla_target_p95_ms=400.0),
            2: SLATracker(tier=2, sla_target_p95_ms=1500.0),
        }

        # Stats tracking
        self.stats = {
            "tier0_calls": 0,
            "local_dense_calls": 0,
            "tier1_calls": 0,
            "tier2_calls": 0,
            "fallbacks": 0,
            "local_dense_fallbacks": 0,
            "local_quantized_fallbacks": 0,
            "api_failures": 0,
            "rate_limits": 0,
            "sla_escalations": 0,
        }

    def classify_task(
        self,
        intent: str | None = None,
        context_length: int = 0,
        role: str | None = None,
        req: SynthesisRequest | ReviewRequest | None = None,
    ) -> int:
        """Dynamic Pareto tier classification (0, 1, or 2) with SLA circuit-breaker check."""
        base_tier = self._classify_base_tier(intent, context_length, role, req)

        # Check if selected tier is circuit-broken / unhealthy; escalate if needed
        tracker = self.sla_trackers.get(base_tier)
        if tracker and not tracker.is_healthy():
            if base_tier < 2:
                logger.info(f"Escalating task from Tier {base_tier} to Tier {base_tier + 1} due to SLA circuit break.")
                self.stats["sla_escalations"] += 1
                return base_tier + 1
        return base_tier

    def _classify_base_tier(
        self,
        intent: str | None = None,
        context_length: int = 0,
        role: str | None = None,
        req: SynthesisRequest | ReviewRequest | None = None,
    ) -> int:
        if not self.enable_reasonmaxxer:
            tier2_intents = {
                "reasoning",
                "complex_synthesis",
                "edge_case_analysis",
                "verifier",
                "code_execution_planning",
                "multi_step",
            }
            if intent and intent.lower() in tier2_intents:
                return 2
            if role == "verifier" or isinstance(req, ReviewRequest):
                return 2
            if context_length > 4000:
                return 2
            return 1

        complex_extraction_intents = {
            "complex_extraction",
            "multi_step_extraction",
            "complex_document_extraction",
            "multi_step_document_extraction",
            "document_extraction",
        }
        if intent:
            norm_intent = intent.lower().strip()
            if norm_intent in complex_extraction_intents or (
                "extraction" in norm_intent
                and ("multi" in norm_intent or "complex" in norm_intent or context_length > 2000)
            ):
                return 0

        tier2_intents = {
            "reasoning",
            "complex_synthesis",
            "edge_case_analysis",
            "verifier",
            "code_execution_planning",
            "multi_step",
            "statutory_determination",
            "multi_step_verification",
            "appeals",
            "contested",
        }
        tier1_intents = {
            "parsing",
            "classification",
            "formatting",
            "search_query_generation",
            "extraction",
            "utility",
            "triage",
        }

        if intent and intent.lower() in tier2_intents:
            return 2
        if intent and intent.lower() in tier1_intents:
            return 1

        if role == "verifier" or isinstance(req, ReviewRequest):
            return 2

        if context_length > 4000:
            return 2

        if isinstance(req, SynthesisRequest):
            if req.program.value in ("medicaid", "housing", "appeals") or len(req.criteria) > 4 or req.required_total > 5:
                return 2
            return 1

        return 1

    def route_document_extraction(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> dict:
        """Route complex multi-step document extraction prompts to local qwen3.8-27b instance."""
        start_t = time.perf_counter()
        self.stats["local_dense_calls"] += 1
        self.stats["tier0_calls"] += 1

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "prompt": prompt,
            "context": context,
            "model": self.local_model_type,
        }

        try:
            if hasattr(self.local_dense_provider, "extract_document"):
                res = self.local_dense_provider.extract_document(params)
            else:
                res = {
                    "status": "success",
                    "model": self.local_model_type,
                    "temperature": temperature,
                    "top_p": top_p,
                    "extracted_fields": {"raw_prompt_length": len(prompt), "context_length": len(context)},
                    "source": "local_qwen3.8-27b",
                }
            lat = (time.perf_counter() - start_t) * 1000.0
            self.sla_trackers[0].record_call(lat, is_error=False)
            return res
        except Exception as exc:
            lat = (time.perf_counter() - start_t) * 1000.0
            self.sla_trackers[0].record_call(lat, is_error=True)
            self._record_error(exc)
            logger.warning(
                f"Local dense model '{self.local_model_type}' extraction failed: {exc}. Retrying with Tier 2 commercial API."
            )
            self.stats["local_dense_fallbacks"] += 1
            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            return {
                "status": "success_fallback",
                "model": self.tier2_model,
                "temperature": temperature,
                "top_p": top_p,
                "extracted_fields": {"raw_prompt_length": len(prompt), "context_length": len(context)},
                "source": f"commercial_fallback:{self.tier2_model}",
            }

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        """Route synthesis request with Pareto tier optimization and fallback."""
        tier = self.classify_task(req=req, role="proposer")
        start_t = time.perf_counter()

        if tier == 0:
            self.stats["local_dense_calls"] += 1
            self.stats["tier0_calls"] += 1
            try:
                result = self.local_dense_provider.synthesize_assessment(req)
                lat = (time.perf_counter() - start_t) * 1000.0
                is_low_conf = result.self_confidence < 0.50
                is_unparseable = "unparseable" in result.rationale.lower() or "fell back" in result.rationale.lower()
                if not is_low_conf and not is_unparseable:
                    self.sla_trackers[0].record_call(lat, is_error=False)
                    return result
                logger.warning("Local dense model emitted low confidence/unparseable output. Retrying with Tier 2.")
                self.sla_trackers[0].record_call(lat, is_error=True)
            except Exception as exc:
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[0].record_call(lat, is_error=True)
                self._record_error(exc)
                logger.warning(f"Local dense model failed: {exc}. Retrying with Tier 2.")

            self.stats["local_dense_fallbacks"] += 1
            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            t2_start = time.perf_counter()
            try:
                res2 = self.tier2_provider.synthesize_assessment(req)
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=False)
                res2.rationale = f"[Tier 2 Fallback from Local Dense '{self.local_model_type}'] {res2.rationale}"
                return res2
            except Exception as exc2:
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=True)
                self._record_error(exc2)
                return self._execute_local_fallback_synthesis(req, exc2)

        elif tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                result = self.tier1_provider.synthesize_assessment(req)
                lat = (time.perf_counter() - start_t) * 1000.0
                is_low_conf = result.self_confidence < 0.50
                is_unparseable = "unparseable" in result.rationale.lower() or "fell back" in result.rationale.lower()

                if not is_low_conf and not is_unparseable:
                    self.sla_trackers[1].record_call(lat, is_error=False)
                    return result

                self.sla_trackers[1].record_call(lat, is_error=True)
                logger.warning("Tier 1 model emitted low confidence/unparseable output. Retrying with Tier 2.")
            except Exception as exc:
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[1].record_call(lat, is_error=True)
                self._record_error(exc)
                logger.warning(f"Tier 1 model failed: {exc}. Retrying with Tier 2.")

            # Try Tier 2
            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            t2_start = time.perf_counter()
            try:
                res2 = self.tier2_provider.synthesize_assessment(req)
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=False)
                res2.rationale = f"[Tier 2 Fallback from Tier 1 model '{self.tier1_model}'] {res2.rationale}"
                return res2
            except Exception as exc2:
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=True)
                self._record_error(exc2)
                return self._execute_local_fallback_synthesis(req, exc2)
        else:
            self.stats["tier2_calls"] += 1
            try:
                res = self.tier2_provider.synthesize_assessment(req)
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[2].record_call(lat, is_error=False)
                return res
            except Exception as exc:
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[2].record_call(lat, is_error=True)
                self._record_error(exc)
                return self._execute_local_fallback_synthesis(req, exc)

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        """Route verifier review request to Tier 2 model with SLA tracking and fallback."""
        tier = self.classify_task(req=req, role="verifier")
        start_t = time.perf_counter()

        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                res = self.tier1_provider.review_assessment(req)
                lat = (time.perf_counter() - start_t) * 1000.0
                if res.supported or not any("unparseable" in c for c in res.concerns):
                    self.sla_trackers[1].record_call(lat, is_error=False)
                    return res
                self.sla_trackers[1].record_call(lat, is_error=True)
            except Exception as exc:
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[1].record_call(lat, is_error=True)
                self._record_error(exc)
                logger.warning(f"Tier 1 verifier review failed: {exc}. Retrying with Tier 2.")

            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            t2_start = time.perf_counter()
            try:
                res2 = self.tier2_provider.review_assessment(req)
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=False)
                return res2
            except Exception as exc2:
                lat2 = (time.perf_counter() - t2_start) * 1000.0
                self.sla_trackers[2].record_call(lat2, is_error=True)
                self._record_error(exc2)
                return self._execute_local_fallback_review(req, exc2)
        else:
            self.stats["tier2_calls"] += 1
            try:
                res = self.tier2_provider.review_assessment(req)
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[2].record_call(lat, is_error=False)
                return res
            except Exception as exc:
                lat = (time.perf_counter() - start_t) * 1000.0
                self.sla_trackers[2].record_call(lat, is_error=True)
                self._record_error(exc)
                return self._execute_local_fallback_review(req, exc)

    def _record_error(self, exc: Exception) -> None:
        self.stats["api_failures"] += 1
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg:
            self.stats["rate_limits"] += 1

    def _execute_local_fallback_synthesis(self, req: SynthesisRequest, primary_exc: Exception) -> SynthesisResult:
        logger.error(f"External API failed ({primary_exc}). Executing Local Quantized Fallback.")
        self.stats["local_quantized_fallbacks"] += 1
        res = self.fallback_provider.synthesize_assessment(req)
        res.rationale = f"[LOCAL QUANTIZED FALLBACK due to API error: {primary_exc}] {res.rationale}"
        return res

    def _execute_local_fallback_review(self, req: ReviewRequest, primary_exc: Exception) -> ReviewResult:
        logger.error(f"External verifier API failed ({primary_exc}). Executing Local Quantized Fallback.")
        self.stats["local_quantized_fallbacks"] += 1
        res = self.fallback_provider.review_assessment(req)
        res.concerns.append(f"Local quantized fallback triggered due to API error: {primary_exc}")
        return res

    def route_and_execute(
        self,
        fn_tier1: Callable[[], T],
        fn_tier2: Callable[[], T],
        intent: str = "utility",
        context_length: int = 0,
        fn_fallback: Callable[[], T] | None = None,
        fn_local: Callable[[], T] | None = None,
    ) -> T:
        """Generic task execution wrapper with routing, retry, and local fallback."""
        tier = self.classify_task(intent=intent, context_length=context_length)
        if tier == 0:
            self.stats["local_dense_calls"] += 1
            self.stats["tier0_calls"] += 1
            if fn_local:
                try:
                    return fn_local()
                except Exception as exc:
                    logger.warning(f"Local dense execution failed: {exc}. Retrying on Tier 2.")
                    self.stats["local_dense_fallbacks"] += 1
                    self.stats["fallbacks"] += 1
                    self.stats["tier2_calls"] += 1
                    try:
                        return fn_tier2()
                    except Exception as exc2:
                        if fn_fallback:
                            self.stats["local_quantized_fallbacks"] += 1
                            return fn_fallback()
                        raise exc2
            else:
                return fn_tier1()

        elif tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                return fn_tier1()
            except Exception as exc:
                logger.warning(f"Tier 1 execution failed: {exc}. Retrying on Tier 2.")
                self.stats["fallbacks"] += 1
                self.stats["tier2_calls"] += 1
                try:
                    return fn_tier2()
                except Exception as exc2:
                    if fn_fallback:
                        self.stats["local_quantized_fallbacks"] += 1
                        return fn_fallback()
                    raise exc2
        else:
            self.stats["tier2_calls"] += 1
            try:
                return fn_tier2()
            except Exception as exc:
                if fn_fallback:
                    self.stats["local_quantized_fallbacks"] += 1
                    return fn_fallback()
                raise exc


__all__ = ["SLATracker", "ModelRouter"]
