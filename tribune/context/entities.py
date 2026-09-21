"""Entities, Data Models, and Strict Edge Taxonomy for Modernized Context Graph.

Defines strict categorical edge taxonomy, entity mentions, calibrated edge decisions,
and graph transaction models for non-autoregressive decision routing.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class EdgeClass(str, enum.Enum):
    """Strict categorical edge taxonomy."""

    Contradicts = "Contradicts"
    Extends = "Extends"
    TemporalFollowup = "TemporalFollowup"
    Irrelevant = "Irrelevant"


@dataclass
class EntityMention:
    """An unstructured entity mention extracted from evidence or dialogue."""

    mention_id: str = field(default_factory=lambda: f"mention_{uuid.uuid4().hex[:8]}")
    text: str = ""
    entity_type: str = "general"
    start_char: int = 0
    end_char: int = 0
    source_doc_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class EntityResolutionResult:
    """Outcome of entity resolution and canonical identification."""

    mention_id: str
    resolved_id: str | None = None
    canonical_name: str = ""
    confidence: float = 1.0
    is_new_entity: bool = False
    normalized_attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeFeatures:
    """Contextual and semantic features for edge decision classification."""

    source_entity_id: str
    target_entity_id: str
    source_text: str = ""
    target_text: str = ""
    source_type: str = "general"
    target_type: str = "general"
    temporal_distance_s: float | None = None
    semantic_similarity: float = 0.5
    negation_detected: bool = False
    subsumption_detected: bool = False
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationMetadata:
    """Calibration metadata documenting probability confidence calibration."""

    model_version: str = "rlcd-decision-v1.0"
    calibration_method: str = "temperature_platt_scaling"
    confidence_score: float = 0.0
    threshold: float = 0.85
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    decision_trace_id: str = field(
        default_factory=lambda: f"trace_{uuid.uuid4().hex[:10]}"
    )


@dataclass
class EdgeCandidate:
    """Candidate relationship between two entities prior to commit or escalation."""

    source_id: str
    target_id: str
    candidate_class: EdgeClass | None = None
    raw_score: float = 0.0
    features: EdgeFeatures = field(default_factory=lambda: EdgeFeatures("", ""))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeDecision:
    """Calibrated decision regarding whether and how to link two entities."""

    source_id: str
    target_id: str
    edge_class: EdgeClass
    confidence: float
    calibration: CalibrationMetadata
    features: EdgeFeatures | None = None
    decision_latency_ms: float = 0.0
    is_malformed: bool = False

    @property
    def is_committed_eligible(self) -> bool:
        return (
            not self.is_malformed
            and self.edge_class != EdgeClass.Irrelevant
            and self.confidence >= self.calibration.threshold
        )


@dataclass
class EscalationRecord:
    """Record placed on the escalation queue when confidence falls below the threshold."""

    record_id: str = field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:8]}")
    candidate: EdgeCandidate = field(
        default_factory=lambda: EdgeCandidate("", "")
    )
    decision: EdgeDecision = field(
        default_factory=lambda: EdgeDecision(
            "", "", EdgeClass.Irrelevant, 0.0, CalibrationMetadata()
        )
    )
    reason: str = "Confidence below 0.85 threshold"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"  # "pending" | "resolved" | "dismissed"
    resolution_notes: str = ""


@dataclass
class GraphTransactionResult:
    """Result of an atomic graph update transaction."""

    transaction_id: str = field(default_factory=lambda: f"tx_{uuid.uuid4().hex[:8]}")
    committed: bool = False
    edge_decision: EdgeDecision | None = None
    escalated: bool = False
    escalation_record: EscalationRecord | None = None
    latency_ms: float = 0.0
    error_message: str | None = None


__all__ = [
    "EdgeClass",
    "EntityMention",
    "EntityResolutionResult",
    "EdgeFeatures",
    "CalibrationMetadata",
    "EdgeCandidate",
    "EdgeDecision",
    "EscalationRecord",
    "GraphTransactionResult",
]
