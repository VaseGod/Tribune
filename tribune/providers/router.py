"""Tiered Model Router interface with Local Quantized Fallback.

Sits between system requests and LLM provider execution. Intelligently routes routine
tasks (parsing, classification, search generation) to lightweight Tier 1 models, and
reserves frontier reasoning models (Tier 2) for multi-step reasoning, verifier duties,
and complex synthesis.

Includes automatic failover:
1. Fallback from Tier 1 to Tier 2 on low confidence or error.
2. Local Quantized Fallback: If external proprietary APIs return errors or rate limits (429),
   gracefully fails over to configured local or self-hosted quantized endpoints.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

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


class ModelRouter:
    """Central router directing tasks to Tier 1 or Tier 2, with Local Quantized Fallback."""

    def __init__(
        self,
        tier1_provider: ModelProvider | None = None,
        tier2_provider: ModelProvider | None = None,
        fallback_provider: ModelProvider | None = None,
        settings: TribuneSettings | None = None,
        recorder: UsageRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.recorder = recorder
        self.name = "model_router"
        self.version = "1.0.0"

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

        # Local Quantized Fallback Provider
        if fallback_provider is not None:
            self.fallback_provider = fallback_provider
        elif getattr(self.settings, "enable_local_fallback", True):
            self.fallback_provider = LocalRulesProvider(role="local_fallback", recorder=self.recorder)
        else:
            self.fallback_provider = LocalRulesProvider(role="local_fallback", recorder=self.recorder)

        # Stats tracking
        self.stats = {
            "tier1_calls": 0,
            "tier2_calls": 0,
            "fallbacks": 0,
            "local_quantized_fallbacks": 0,
            "api_failures": 0,
            "rate_limits": 0,
        }

    def classify_task(
        self,
        intent: str | None = None,
        context_length: int = 0,
        role: str | None = None,
        req: SynthesisRequest | ReviewRequest | None = None,
    ) -> int:
        """Lightweight heuristic classifier assigning task tier (1 or 2)."""
        tier2_intents = {
            "reasoning",
            "complex_synthesis",
            "edge_case_analysis",
            "verifier",
            "code_execution_planning",
            "multi_step",
        }
        tier1_intents = {
            "parsing",
            "classification",
            "formatting",
            "search_query_generation",
            "extraction",
            "utility",
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
            if req.program.value in ("medicaid", "housing") or len(req.criteria) > 4 or req.required_total > 5:
                return 2
            return 1

        return 1

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        """Route synthesis request with automatic Tier 1->Tier 2 and Local Quantized Fallback."""
        tier = self.classify_task(req=req, role="proposer")

        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                result = self.tier1_provider.synthesize_assessment(req)
                is_low_conf = result.self_confidence < 0.50
                is_unparseable = "unparseable" in result.rationale.lower() or "fell back" in result.rationale.lower()

                if not is_low_conf and not is_unparseable:
                    return result

                logger.warning("Tier 1 model emitted low confidence/unparseable output. Retrying with Tier 2.")
            except Exception as exc:
                self._record_error(exc)
                logger.warning(f"Tier 1 model failed: {exc}. Retrying with Tier 2.")

            # Try Tier 2
            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            try:
                res2 = self.tier2_provider.synthesize_assessment(req)
                res2.rationale = f"[Tier 2 Fallback from Tier 1 model '{self.tier1_model}'] {res2.rationale}"
                return res2
            except Exception as exc2:
                self._record_error(exc2)
                return self._execute_local_fallback_synthesis(req, exc2)
        else:
            self.stats["tier2_calls"] += 1
            try:
                return self.tier2_provider.synthesize_assessment(req)
            except Exception as exc:
                self._record_error(exc)
                return self._execute_local_fallback_synthesis(req, exc)

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        """Route verifier review request to Tier 2 model with Local Quantized Fallback guarantee."""
        tier = self.classify_task(req=req, role="verifier")

        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                res = self.tier1_provider.review_assessment(req)
                if res.supported or not any("unparseable" in c for c in res.concerns):
                    return res
            except Exception as exc:
                self._record_error(exc)
                logger.warning(f"Tier 1 verifier review failed: {exc}. Retrying with Tier 2.")

            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            try:
                return self.tier2_provider.review_assessment(req)
            except Exception as exc2:
                self._record_error(exc2)
                return self._execute_local_fallback_review(req, exc2)
        else:
            self.stats["tier2_calls"] += 1
            try:
                return self.tier2_provider.review_assessment(req)
            except Exception as exc:
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
    ) -> T:
        """Generic task execution wrapper with routing, retry, and local fallback."""
        tier = self.classify_task(intent=intent, context_length=context_length)
        if tier == 1:
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
