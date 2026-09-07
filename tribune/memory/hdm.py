"""Hierarchical Documentary Memory (HDM).

Three discrete abstraction tiers for repository documentation and contracts:
- L0 Tier: Global architectural invariants and repository-wide constraints.
- L1 Tier: Module contracts, service interfaces, and protocol definitions.
- L2 Tier: Concrete file schemas, implementation specifications, and local code blocks.

Every committed documentary unit is encapsulated as a provenanced tuple: τ = (x, π(x)).
Supports hierarchical drill-down, rollup, and scoped contextual queries.
"""

from __future__ import annotations

import collections
import enum
import threading
from dataclasses import dataclass, field
from typing import Any

from .retrieval import ProvenancedTuple, ProvenancePointer


class HDMTier(str, enum.Enum):
    L0_GLOBAL = "L0"  # Global architectural invariants & repository-wide constraints
    L1_MODULE = "L1"  # Module contracts, service interfaces, protocol definitions
    L2_CONCRETE = "L2"  # Concrete file schemas, implementation specifications, local blocks


@dataclass
class DocumentaryUnit:
    """A documentary memory unit encapsulated with provenanced tuple τ = (x, π(x))."""

    doc_id: str
    tier: HDMTier
    title: str
    content: str
    provenanced_tuple: ProvenancedTuple[dict[str, Any]]
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def provenance(self) -> ProvenancePointer:
        return self.provenanced_tuple.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tier": self.tier.value,
            "title": self.title,
            "content": self.content,
            "parent_id": self.parent_id,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
        }


class HierarchicalDocumentaryMemory:
    """Multi-tier hierarchical documentary memory store with provenanced tuples."""

    def __init__(self) -> None:
        self._units: dict[str, DocumentaryUnit] = {}
        self._by_tier: dict[HDMTier, list[str]] = {
            HDMTier.L0_GLOBAL: [],
            HDMTier.L1_MODULE: [],
            HDMTier.L2_CONCRETE: [],
        }
        self._children: dict[str, list[str]] = collections.defaultdict(list)
        self._lock = threading.RLock()

    def store_document(
        self,
        doc_id: str,
        tier: HDMTier | str,
        title: str,
        content: str,
        source_uri: str,
        commit_hash: str = "HEAD",
        line_start: int = 1,
        line_end: int = 1,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentaryUnit:
        """Store a documentary unit encapsulated as a provenanced tuple τ = (x, π(x))."""
        with self._lock:
            h_tier = HDMTier(tier) if isinstance(tier, str) else tier
            ptr = ProvenancePointer.create(
                source_uri=source_uri,
                commit_hash=commit_hash,
                line_start=line_start,
                line_end=line_end,
            )
            payload = {
                "doc_id": doc_id,
                "tier": h_tier.value,
                "title": title,
                "content": content,
                "parent_id": parent_id,
                "tags": list(tags or []),
                "metadata": metadata or {},
            }
            p_tuple = ProvenancedTuple(data=payload, provenance=ptr)
            unit = DocumentaryUnit(
                doc_id=doc_id,
                tier=h_tier,
                title=title,
                content=content,
                provenanced_tuple=p_tuple,
                parent_id=parent_id,
                tags=list(tags or []),
                metadata=metadata or {},
            )

            self._units[doc_id] = unit
            self._by_tier[h_tier].append(doc_id)
            if parent_id:
                self._children[parent_id].append(doc_id)

            return unit

    def get_unit(self, doc_id: str) -> DocumentaryUnit | None:
        with self._lock:
            return self._units.get(doc_id)

    def query_tier(self, tier: HDMTier | str) -> list[DocumentaryUnit]:
        """Fetch all documents belonging to an abstraction tier."""
        with self._lock:
            h_tier = HDMTier(tier) if isinstance(tier, str) else tier
            doc_ids = self._by_tier.get(h_tier, [])
            return [self._units[did] for did in doc_ids if did in self._units]

    def drill_down(self, doc_id: str) -> list[DocumentaryUnit]:
        """Hierarchical drill-down: fetch all direct children belonging to a lower tier."""
        with self._lock:
            child_ids = self._children.get(doc_id, [])
            return [self._units[cid] for cid in child_ids if cid in self._units]

    def rollup(self, doc_id: str) -> DocumentaryUnit | None:
        """Hierarchical rollup: fetch parent document in the higher tier."""
        with self._lock:
            unit = self._units.get(doc_id)
            if unit and unit.parent_id:
                return self._units.get(unit.parent_id)
            return None

    def get_provenance_pointers(self, doc_ids: list[str] | None = None) -> list[ProvenancePointer]:
        """Extract provenance pointers for citation locking."""
        with self._lock:
            selected = (
                [self._units[did] for did in doc_ids if did in self._units]
                if doc_ids is not None
                else list(self._units.values())
            )
            return [u.provenance for u in selected]

    def to_context_string(self, tier: HDMTier | str | None = None) -> str:
        """Format tier documents into structured context for LLM prompts."""
        with self._lock:
            tiers = [HDMTier(tier)] if tier else [HDMTier.L0_GLOBAL, HDMTier.L1_MODULE, HDMTier.L2_CONCRETE]
            lines = ["=== HIERARCHICAL DOCUMENTARY MEMORY (HDM) ==="]
            for t in tiers:
                units = self.query_tier(t)
                if not units:
                    continue
                tier_label = {
                    HDMTier.L0_GLOBAL: "L0: GLOBAL ARCHITECTURAL INVARIANTS",
                    HDMTier.L1_MODULE: "L1: MODULE CONTRACTS & PROTOCOLS",
                    HDMTier.L2_CONCRETE: "L2: CONCRETE SCHEMAS & SPECIFICATIONS",
                }[t]
                lines.append(f"\n--- {tier_label} ---")
                for u in units:
                    lines.append(f"• [{u.doc_id}] {u.title} (Pointer: {u.provenance.pointer_id})")
                    lines.append(f"  {u.content.strip()}")
            return "\n".join(lines)


__all__ = [
    "HDMTier",
    "DocumentaryUnit",
    "HierarchicalDocumentaryMemory",
]
