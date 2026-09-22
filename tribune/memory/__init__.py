"""Typed, per-case, access-controlled memory partitions, episodic memory, and Tri-Memory architecture."""

from .hdm import (
    DocumentaryUnit,
    HDMTier,
    HierarchicalDocumentaryMemory,
    ProvenanceGatedMemory,
    ReasoningBudgetTracker,
    SpeculativeEmbeddingCache,
    majority_rule_bundle,
)
from .retrieval import (
    AbstentionSignal,
    CitationLockHarness,
    ProvenancedTuple,
    ProvenancePointer,
    RetentionAwareRetriever,
    UngroundedAssertionViolationError,
)
from .timeline import (
    MemoryEventsTimeline,
    RetentionPolicy,
    StateDelta,
    TemporalImmutabilityError,
    TimelineEvent,
    TimelineEventPayload,
)

__all__ = [
    "AbstentionSignal",
    "CitationLockHarness",
    "DocumentaryUnit",
    "HDMTier",
    "HierarchicalDocumentaryMemory",
    "MemoryEventsTimeline",
    "ProvenanceGatedMemory",
    "ProvenancePointer",
    "ProvenancedTuple",
    "ReasoningBudgetTracker",
    "RetentionAwareRetriever",
    "RetentionPolicy",
    "SpeculativeEmbeddingCache",
    "StateDelta",
    "TemporalImmutabilityError",
    "TimelineEvent",
    "TimelineEventPayload",
    "UngroundedAssertionViolationError",
    "majority_rule_bundle",
]
