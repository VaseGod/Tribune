"""Coarse-to-Fine Dynamic Adapter Gating Router (MoVA Integration).

Orchestrates two-stage routing:
- Stage 1: Lightweight coarse router detects input modality. Pure text bypasses adapters entirely.
- Stage 2: Multimodal inputs dynamically activate task-relevant experts subject to KV-cache constraints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .experts import ExpertType
from .gating import CoarseGateDecision, CoarseGatingRouter, InputContextFeatures, InputModality
from .mova import FineGatingDecision, MoVAdapterPipeline

logger = logging.getLogger(__name__)


@dataclass
class AdapterRoutingDecision:
    """Comprehensive routing decision across both coarse and fine stages."""

    modality: InputModality
    bypass_mova: bool
    active_experts: list[ExpertType]
    estimated_kv_cache_mb: float
    cross_modal_interference_proxy: float
    total_routing_latency_ms: float
    coarse_decision: CoarseGateDecision
    fine_decision: FineGatingDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterRouterTelemetry:
    """Telemetry tracking dynamic adapter activations and memory footprint."""

    total_queries: int = 0
    active_expert_count: int = 0
    routing_latency_ms: float = 0.0
    estimated_kv_cache_mb: float = 0.0
    cross_modal_interference_proxy: float = 0.0
    modality_activation_distribution: dict[str, int] = field(
        default_factory=lambda: {m.value: 0 for m in InputModality}
    )
    expert_activation_distribution: dict[str, int] = field(
        default_factory=lambda: {e.value: 0 for e in ExpertType}
    )

    def record_decision(self, decision: AdapterRoutingDecision) -> None:
        self.total_queries += 1
        self.active_expert_count = len(decision.active_experts)
        self.routing_latency_ms = decision.total_routing_latency_ms
        self.estimated_kv_cache_mb = decision.estimated_kv_cache_mb
        self.cross_modal_interference_proxy = decision.cross_modal_interference_proxy

        mod_val = decision.modality.value
        self.modality_activation_distribution[mod_val] = (
            self.modality_activation_distribution.get(mod_val, 0) + 1
        )
        for exp in decision.active_experts:
            exp_val = exp.value
            self.expert_activation_distribution[exp_val] = (
                self.expert_activation_distribution.get(exp_val, 0) + 1
            )


class DynamicMoVARouter:
    """Coarse-to-fine Dynamic Adapter Router.

    Prevents static adapter overhead and cross-modal interference by routing pure text
    directly to the LLM trunk and activating only task-relevant multimodal experts.
    """

    def __init__(
        self,
        max_active_experts: int = 2,
        expert_timeout: float = 5.0,
        minimum_activation_confidence: float = 0.70,
        cache_budget_mb: float = 1024.0,
        enable_dynamic_routing: bool = True,
    ) -> None:
        self.max_active_experts = max_active_experts
        self.expert_timeout = expert_timeout
        self.minimum_activation_confidence = minimum_activation_confidence
        self.cache_budget_mb = cache_budget_mb
        self.enable_dynamic_routing = enable_dynamic_routing

        self.coarse_router = CoarseGatingRouter()
        self.mova_pipeline = MoVAdapterPipeline(
            max_active_experts=max_active_experts,
            minimum_activation_confidence=minimum_activation_confidence,
            cache_budget_mb=cache_budget_mb,
            enable_dynamic_routing=enable_dynamic_routing,
        )
        self.telemetry = AdapterRouterTelemetry()

    def route(
        self,
        context: InputContextFeatures,
        force_static: bool = False,
    ) -> AdapterRoutingDecision:
        """Route input through Stage 1 coarse gate and Stage 2 fine expert gating."""
        start_t = time.perf_counter()

        if force_static:
            # Force static mode: bypass coarse gate and activate static set
            coarse = self.coarse_router.classify_modality(context)
            fine = self.mova_pipeline.select_experts(context, force_static=True)
            tot_latency = (time.perf_counter() - start_t) * 1000.0
            decision = AdapterRoutingDecision(
                modality=coarse.modality,
                bypass_mova=False,
                active_experts=fine.active_experts,
                estimated_kv_cache_mb=fine.estimated_kv_cache_mb,
                cross_modal_interference_proxy=fine.cross_modal_interference_proxy,
                total_routing_latency_ms=round(tot_latency, 3),
                coarse_decision=coarse,
                fine_decision=fine,
            )
            self.telemetry.record_decision(decision)
            return decision

        # Stage 1: Coarse Gating Router
        coarse = self.coarse_router.classify_modality(context)

        if coarse.bypass_mova or coarse.modality == InputModality.UNIMODAL_TEXT:
            # Pure text input -> Bypass all adapters directly to LLM trunk
            tot_latency = (time.perf_counter() - start_t) * 1000.0
            decision = AdapterRoutingDecision(
                modality=InputModality.UNIMODAL_TEXT,
                bypass_mova=True,
                active_experts=[],
                estimated_kv_cache_mb=0.0,
                cross_modal_interference_proxy=0.0,
                total_routing_latency_ms=round(tot_latency, 3),
                coarse_decision=coarse,
                fine_decision=None,
            )
            self.telemetry.record_decision(decision)
            logger.debug("[DynamicMoVARouter] Pure text routed directly to LLM trunk (0 adapters, 0MB KV-cache)")
            return decision

        # Stage 2: Fine Gating MoVA Pipeline
        fine = self.mova_pipeline.select_experts(context, force_static=False)
        tot_latency = (time.perf_counter() - start_t) * 1000.0

        decision = AdapterRoutingDecision(
            modality=coarse.modality,
            bypass_mova=False,
            active_experts=fine.active_experts,
            estimated_kv_cache_mb=fine.estimated_kv_cache_mb,
            cross_modal_interference_proxy=fine.cross_modal_interference_proxy,
            total_routing_latency_ms=round(tot_latency, 3),
            coarse_decision=coarse,
            fine_decision=fine,
        )
        self.telemetry.record_decision(decision)
        return decision

    def get_telemetry(self) -> dict[str, Any]:
        return {
            "active_expert_count": self.telemetry.active_expert_count,
            "routing_latency_ms": self.telemetry.routing_latency_ms,
            "estimated_kv_cache_mb": self.telemetry.estimated_kv_cache_mb,
            "cross_modal_interference_proxy": self.telemetry.cross_modal_interference_proxy,
            "modality_activation_distribution": self.telemetry.modality_activation_distribution,
            "expert_activation_distribution": self.telemetry.expert_activation_distribution,
        }


__all__ = [
    "AdapterRoutingDecision",
    "AdapterRouterTelemetry",
    "DynamicMoVARouter",
]
