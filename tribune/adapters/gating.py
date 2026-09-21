"""Stage 1 Coarse Gating Router for Modality Classification.

Classifies input queries into unimodal text vs multimodal variants (acoustic-only, vision-only,
mixed multimodal). Routes pure text queries directly to the LLM trunk to bypass all adapters,
eliminating KV-cache allocation and cross-modal interference.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class InputModality(str, enum.Enum):
    """Input modality classification taxonomy."""

    UNIMODAL_TEXT = "unimodal_text"
    ACOUSTIC_ONLY = "acoustic_only"
    VISION_ONLY = "vision_only"
    MULTIMODAL_MIXED = "multimodal_mixed"


@dataclass
class InputContextFeatures:
    """Features extracted from the input payload for coarse gating."""

    text: str = ""
    has_audio: bool = False
    audio_duration_s: float = 0.0
    has_image: bool = False
    image_count: int = 0
    has_structured_graph: bool = False
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoarseGateDecision:
    """Outcome of Stage 1 coarse modality gating."""

    modality: InputModality
    confidence: float
    bypass_mova: bool  # True for pure text
    routing_latency_ms: float
    decision_trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CoarseGatingRouter:
    """Stage 1 Lightweight Calibrated Modality Classifier."""

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self.confidence_threshold = confidence_threshold

    def classify_modality(self, context: InputContextFeatures) -> CoarseGateDecision:
        """Classify input modality. Fast deterministic evaluation (<1ms)."""
        start_t = time.perf_counter()

        has_audio = context.has_audio or context.audio_duration_s > 0
        has_vision = context.has_image or context.image_count > 0
        has_graph = context.has_structured_graph

        if not has_audio and not has_vision and not has_graph:
            # Pure text input -> Direct LLM trunk route
            modality = InputModality.UNIMODAL_TEXT
            conf = 0.99
            bypass = True
        elif has_audio and not has_vision and not has_graph and len(context.text.strip()) == 0:
            modality = InputModality.ACOUSTIC_ONLY
            conf = 0.95
            bypass = False
        elif has_vision and not has_audio and not has_graph and len(context.text.strip()) == 0:
            modality = InputModality.VISION_ONLY
            conf = 0.95
            bypass = False
        else:
            modality = InputModality.MULTIMODAL_MIXED
            conf = 0.98
            bypass = False

        dur_ms = (time.perf_counter() - start_t) * 1000.0

        return CoarseGateDecision(
            modality=modality,
            confidence=conf,
            bypass_mova=bypass,
            routing_latency_ms=round(dur_ms, 3),
        )


__all__ = [
    "InputModality",
    "InputContextFeatures",
    "CoarseGateDecision",
    "CoarseGatingRouter",
]
