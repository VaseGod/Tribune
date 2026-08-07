"""Tiered Model Router interface.

Sits between system requests and LLM provider execution. Intelligently routes routine
tasks (parsing, classification, search generation) to lightweight Tier 1 models, and
reserves frontier reasoning models (Tier 2) for multi-step reasoning, verifier duties,
and complex synthesis. Includes an automatic fallback from Tier 1 to Tier 2 when Tier 1
encounters errors or low confidence.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

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
    """Central router directing tasks to Tier 1 (fast/cheap) or Tier 2 (deep reasoning)."""

    def __init__(
        self,
        tier1_provider: ModelProvider | None = None,
        tier2_provider: ModelProvider | None = None,
        settings: TribuneSettings | None = None,
        recorder: UsageRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.recorder = recorder
        self.name = "model_router"
        self.version = "1.0.0"

        # Tier 1 defaults: env vars DEFAULT_TIER1_MODEL / TRIBUNE_DEFAULT_TIER1_MODEL / settings.tier1_model
        self.tier1_model = (
            os.getenv("DEFAULT_TIER1_MODEL")
            or os.getenv("TRIBUNE_DEFAULT_TIER1_MODEL")
            or self.settings.tier1_model
        )
        # Tier 2 defaults: env vars DEFAULT_TIER2_MODEL / TRIBUNE_DEFAULT_TIER2_MODEL / settings.tier2_model
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

        # Accounting and metrics
        self.stats = {
            "tier1_calls": 0,
            "tier2_calls": 0,
            "fallbacks": 0,
        }

    def classify_task(
        self,
        intent: str | None = None,
        context_length: int = 0,
        role: str | None = None,
        req: SynthesisRequest | ReviewRequest | None = None,
    ) -> int:
        """Lightweight heuristic classifier assigning task tier (1 or 2)."""
        # Tier 2 forced intents
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

        # Verifier role or review requests require Tier 2 reasoning
        if role == "verifier" or isinstance(req, ReviewRequest):
            return 2

        # Context length heuristic: inputs > 4000 tokens / 12000 chars require Tier 2
        if context_length > 4000:
            return 2

        # Synthesis Request heuristics
        if isinstance(req, SynthesisRequest):
            # Complex programs or large criteria sets route to Tier 2
            if req.program.value in ("medicaid", "housing") or len(req.criteria) > 4 or req.required_total > 5:
                return 2
            return 1

        return 1

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        """Route synthesis request with automatic fallback from Tier 1 to Tier 2."""
        tier = self.classify_task(req=req, role="proposer")

        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                result = self.tier1_provider.synthesize_assessment(req)
                # Check for low confidence or parse failure triggering fallback
                is_low_conf = result.self_confidence < 0.50
                is_unparseable = "unparseable" in result.rationale.lower() or "fell back" in result.rationale.lower()

                if not is_low_conf and not is_unparseable:
                    return result

                logger.warning("Tier 1 model emitted low confidence/unparseable output. Retrying with Tier 2.")
            except Exception as exc:
                logger.warning(f"Tier 1 model failed with exception: {exc}. Retrying with Tier 2.")

            # Fallback to Tier 2
            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            res2 = self.tier2_provider.synthesize_assessment(req)
            res2.rationale = f"[Tier 2 Fallback from Tier 1 model '{self.tier1_model}'] {res2.rationale}"
            return res2
        else:
            self.stats["tier2_calls"] += 1
            return self.tier2_provider.synthesize_assessment(req)

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        """Route verifier review request to Tier 2 model with fallback guarantee."""
        tier = self.classify_task(req=req, role="verifier")

        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                res = self.tier1_provider.review_assessment(req)
                if res.supported or not any("unparseable" in c for c in res.concerns):
                    return res
            except Exception as exc:
                logger.warning(f"Tier 1 verifier review failed: {exc}. Retrying with Tier 2.")

            self.stats["fallbacks"] += 1
            self.stats["tier2_calls"] += 1
            return self.tier2_provider.review_assessment(req)
        else:
            self.stats["tier2_calls"] += 1
            return self.tier2_provider.review_assessment(req)

    def route_and_execute(
        self,
        fn_tier1: Callable[[], T],
        fn_tier2: Callable[[], T],
        intent: str = "utility",
        context_length: int = 0,
    ) -> T:
        """Generic task execution wrapper with routing and fallback."""
        tier = self.classify_task(intent=intent, context_length=context_length)
        if tier == 1:
            self.stats["tier1_calls"] += 1
            try:
                return fn_tier1()
            except Exception as exc:
                logger.warning(f"Tier 1 execution failed: {exc}. Retrying on Tier 2.")
                self.stats["fallbacks"] += 1
                self.stats["tier2_calls"] += 1
                return fn_tier2()
        else:
            self.stats["tier2_calls"] += 1
            return fn_tier2()
