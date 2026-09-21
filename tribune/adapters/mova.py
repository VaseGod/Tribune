"""Stage 2 MoV-Adapter (Mixture-of-Value Attention) Pipeline.

Dynamically activates only task-relevant modal experts, enforcing KV-cache-aware routing
constraints and measuring cross-modal interference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .experts import (
    AcousticStreamExpert,
    ExpertAdapter,
    ExpertType,
    GraphStructuralExpert,
    VisionCLIPExpert,
    VisionDINOv2Expert,
)
from .gating import InputContextFeatures

logger = logging.getLogger(__name__)


@dataclass
class FineGatingDecision:
    """Outcome of Stage 2 dynamic expert selection."""

    active_experts: list[ExpertType]
    activation_scores: dict[str, float]
    estimated_kv_cache_mb: float
    cross_modal_interference_proxy: float
    decision_latency_ms: float
    static_fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class MoVAdapterPipeline:
    """Stage 2 Mixture-of-Value Attention dynamic expert selection engine."""

    def __init__(
        self,
        max_active_experts: int = 2,
        minimum_activation_confidence: float = 0.70,
        cache_budget_mb: float = 1024.0,
        enable_dynamic_routing: bool = True,
    ) -> None:
        self.max_active_experts = max(1, max_active_experts)
        self.minimum_activation_confidence = minimum_activation_confidence
        self.cache_budget_mb = cache_budget_mb
        self.enable_dynamic_routing = enable_dynamic_routing

        # Pre-registered expert adapters
        self.experts: dict[ExpertType, ExpertAdapter] = {
            ExpertType.VISION_DINOV2: VisionDINOv2Expert(),
            ExpertType.VISION_CLIP: VisionCLIPExpert(),
            ExpertType.ACOUSTIC_STREAM: AcousticStreamExpert(),
            ExpertType.GRAPH_STRUCTURAL: GraphStructuralExpert(),
        }

    def compute_expert_affinities(self, context: InputContextFeatures) -> dict[ExpertType, float]:
        """Compute task-relevant affinity/activation scores for each candidate expert."""
        affinities: dict[ExpertType, float] = {}

        has_audio = context.has_audio or context.audio_duration_s > 0
        has_image = context.has_image or context.image_count > 0
        has_graph = context.has_structured_graph

        # Acoustic expert affinity
        if has_audio:
            dur_weight = min(0.30, context.audio_duration_s / 60.0 * 0.30)
            affinities[ExpertType.ACOUSTIC_STREAM] = round(0.70 + dur_weight, 4)
        else:
            affinities[ExpertType.ACOUSTIC_STREAM] = 0.05

        # Vision experts affinity
        if has_image:
            # If document layout / bounding boxes mentioned, boost DINOv2
            is_doc_layout = "layout" in context.text.lower() or "table" in context.text.lower() or "form" in context.text.lower()
            if is_doc_layout:
                affinities[ExpertType.VISION_DINOV2] = 0.95
                affinities[ExpertType.VISION_CLIP] = 0.72
            else:
                affinities[ExpertType.VISION_DINOV2] = 0.75
                affinities[ExpertType.VISION_CLIP] = 0.90
        else:
            affinities[ExpertType.VISION_DINOV2] = 0.05
            affinities[ExpertType.VISION_CLIP] = 0.05

        # Graph structural expert affinity
        if has_graph or "citation" in context.text.lower() or "dependency" in context.text.lower():
            affinities[ExpertType.GRAPH_STRUCTURAL] = 0.82
        else:
            affinities[ExpertType.GRAPH_STRUCTURAL] = 0.10

        return affinities

    def select_experts(
        self,
        context: InputContextFeatures,
        force_static: bool = False,
    ) -> FineGatingDecision:
        """Dynamically select active experts observing KV-cache and confidence constraints."""
        start_t = time.perf_counter()

        if force_static or not self.enable_dynamic_routing:
            # Static fallback: activate all vision and acoustic experts regardless of task
            static_experts = [ExpertType.VISION_DINOV2, ExpertType.VISION_CLIP, ExpertType.ACOUSTIC_STREAM]
            total_cache = sum(self.experts[e].spec.kv_cache_size_mb for e in static_experts)
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            return FineGatingDecision(
                active_experts=static_experts,
                activation_scores={e.value: 1.0 for e in static_experts},
                estimated_kv_cache_mb=total_cache,
                cross_modal_interference_proxy=0.65,  # High interference in static mode
                decision_latency_ms=round(dur_ms, 3),
                static_fallback_used=True,
            )

        affinities = self.compute_expert_affinities(context)

        # Filter by minimum confidence
        viable_candidates = [
            (exp, score) for exp, score in affinities.items()
            if score >= self.minimum_activation_confidence
        ]
        # Sort descending by activation score
        viable_candidates.sort(key=lambda x: x[1], reverse=True)

        # Enforce max_active_experts
        selected_candidates = viable_candidates[: self.max_active_experts]

        # Enforce KV-cache budget constraints
        active: list[ExpertType] = []
        accumulated_cache_mb = 0.0
        for exp, _score in selected_candidates:
            exp_cache = self.experts[exp].spec.kv_cache_size_mb
            if (accumulated_cache_mb + exp_cache) <= self.cache_budget_mb:
                active.append(exp)
                accumulated_cache_mb += exp_cache
            else:
                logger.warning(
                    f"[MoVAdapterPipeline] Pruned expert {exp.value} (requires {exp_cache}MB, "
                    f"would exceed budget {self.cache_budget_mb}MB)"
                )

        # Cross-modal interference proxy: 0.0 for 0-1 experts, low for well-aligned 2 experts
        if len(active) <= 1:
            interference_proxy = 0.0
        elif len(active) == 2:
            interference_proxy = 0.12
        else:
            interference_proxy = 0.40

        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return FineGatingDecision(
            active_experts=active,
            activation_scores={exp.value: round(score, 4) for exp, score in affinities.items()},
            estimated_kv_cache_mb=round(accumulated_cache_mb, 2),
            cross_modal_interference_proxy=round(interference_proxy, 4),
            decision_latency_ms=round(dur_ms, 3),
            static_fallback_used=False,
        )


__all__ = [
    "FineGatingDecision",
    "MoVAdapterPipeline",
]
