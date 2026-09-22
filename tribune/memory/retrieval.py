"""Citation-Locked Retrieval Harness & Provenanced Tuples.

Enforces the mathematical inclusion property for grounded agent reasoning:
    Valid(a) <=> C ⊆ O and C != ∅ for actionable assertions
where:
    O = {π_1, π_2, ..., π_k} is the set of open, validated evidence pointers loaded into context.
    C = cited evidence pointers extracted from candidate agent output.
    a = ⊥ (AbstentionSignal) returned with UNGROUNDED_ASSERTION_VIOLATION if C ⊈ O.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..security.audit import SecurityEventType, record_security_event
from .consolidation import ConsolidatedMemoryTrace, ProvenanceNode

T = TypeVar("T")


@dataclass(frozen=True)
class ProvenancePointer:
    """Immutable evidence pointer π(x) anchoring data to physical repository artifacts."""

    source_uri: str
    commit_hash: str
    timestamp: float
    line_start: int
    line_end: int

    @property
    def pointer_id(self) -> str:
        """Canonical pointer identifier."""
        short_hash = self.commit_hash[:8] if len(self.commit_hash) >= 8 else self.commit_hash
        return f"{self.source_uri}@{short_hash}:{self.line_start}-{self.line_end}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "pointer_id": self.pointer_id,
        }

    @classmethod
    def create(
        cls,
        source_uri: str,
        commit_hash: str = "HEAD",
        line_start: int = 1,
        line_end: int = 1,
        timestamp: float | None = None,
    ) -> ProvenancePointer:
        return cls(
            source_uri=source_uri,
            commit_hash=commit_hash,
            timestamp=timestamp if timestamp is not None else time.time(),
            line_start=line_start,
            line_end=line_end,
        )


@dataclass
class ProvenancedTuple(Generic[T]):
    """Encapsulated memory unit τ = (x, π(x))."""

    data: T
    provenance: ProvenancePointer

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data if isinstance(self.data, dict | list | str | int | float | bool) else str(self.data),
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class AbstentionSignal:
    """Abstention signal (a = ⊥) emitted when citation inclusion invariants are violated."""

    is_abstention: bool = True
    diagnostic_error: str = "UNGROUNDED_ASSERTION_VIOLATION"
    reason: str = ""
    unauthorized_citations: list[str] = field(default_factory=list)
    open_citations_count: int = 0
    candidate_output: Any = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_abstention": True,
            "diagnostic_error": self.diagnostic_error,
            "reason": self.reason,
            "unauthorized_citations": list(self.unauthorized_citations),
            "open_citations_count": self.open_citations_count,
            "timestamp": self.timestamp,
        }


class UngroundedAssertionViolationError(RuntimeError):
    """Raised when an ungrounded factual assertion or unauthorized code modification is detected."""
    pass


class CitationLockHarness:
    """Enforces citation-locked retrieval verification across memory stores.

    Guarantees:
        Valid(a) <=> C ⊆ O and (C != ∅ for actionable assertions)
    """

    _CITATION_REGEX = [
        # Explicit citation tag: [cite: target] or [ref: target]
        re.compile(r"\[(?:cite|provenance|ref):\s*([^\]]+)\]", re.IGNORECASE),
        # Markdown file range link pattern: [ref](uri#Lstart-Lend) or (uri:start-end)
        re.compile(r"(?:\[(?:[^\]]*)\]\(([^)#\s]+)(?:#L?(\d+)-L?(\d+))?\))"),
        # Plain citation reference pattern: @source_uri:line_start-line_end
        re.compile(r"(?:^|[\s(])@([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+):(\d+)-(\d+)"),
    ]

    def __init__(self, case_id: str | None = None) -> None:
        self.case_id = case_id
        self._open_pointers: dict[str, ProvenancePointer] = {}
        self._lock = threading.RLock()

    def register_evidence(
        self,
        pointer_or_tuple: (
            ProvenancePointer
            | ProvenancedTuple[Any]
            | ConsolidatedMemoryTrace
            | ProvenanceNode
            | list[Any]
        ),
    ) -> None:
        """Load open validated evidence pointers O into the agent context."""
        with self._lock:
            items = pointer_or_tuple if isinstance(pointer_or_tuple, list) else [pointer_or_tuple]
            for item in items:
                if isinstance(item, ProvenancedTuple):
                    ptr = item.provenance
                    self._open_pointers[ptr.pointer_id] = ptr
                    self._open_pointers[ptr.source_uri] = ptr
                elif isinstance(item, ProvenancePointer):
                    self._open_pointers[item.pointer_id] = item
                    self._open_pointers[item.source_uri] = item
                elif isinstance(item, ProvenanceNode):
                    ptr = ProvenancePointer(
                        source_uri=item.source_id,
                        commit_hash=item.content_hash,
                        timestamp=time.time(),
                        line_start=item.chunk_index,
                        line_end=item.chunk_index,
                    )
                    self._open_pointers[ptr.pointer_id] = ptr
                    self._open_pointers[item.source_id] = ptr
                    self._open_pointers[item.content_hash] = ptr
                    self._open_pointers[item.content_hash[:16]] = ptr
                    self._open_pointers[f"{item.source_id}:{item.chunk_index}"] = ptr
                    if item.node_id:
                        self._open_pointers[item.node_id] = ptr
                elif isinstance(item, ConsolidatedMemoryTrace):
                    # Register root assertion pointer
                    trace_ptr = ProvenancePointer(
                        source_uri=f"trace:{item.trace_id[:16]}",
                        commit_hash=item.trace_id,
                        timestamp=time.time(),
                        line_start=1,
                        line_end=1,
                    )
                    self._open_pointers[item.trace_id] = trace_ptr
                    self._open_pointers[item.trace_id[:16]] = trace_ptr
                    self._open_pointers[f"trace:{item.trace_id[:16]}"] = trace_ptr

                    # Recursively register all underlying leaf ProvenanceNodes
                    for leaf in item.resolve_all_leaf_citations():
                        l_ptr = ProvenancePointer(
                            source_uri=leaf.source_id,
                            commit_hash=leaf.content_hash,
                            timestamp=time.time(),
                            line_start=leaf.chunk_index,
                            line_end=leaf.chunk_index,
                        )
                        self._open_pointers[l_ptr.pointer_id] = l_ptr
                        self._open_pointers[leaf.source_id] = l_ptr
                        self._open_pointers[leaf.content_hash] = l_ptr
                        self._open_pointers[leaf.content_hash[:16]] = l_ptr
                        self._open_pointers[f"{leaf.source_id}:{leaf.chunk_index}"] = l_ptr
                        if leaf.node_id:
                            self._open_pointers[leaf.node_id] = l_ptr

    def resolve_trace_provenance(self, trace: ConsolidatedMemoryTrace) -> list[ProvenanceNode]:
        """Navigate from root assertion down to underlying leaf ProvenanceNode citations."""
        return trace.resolve_all_leaf_citations()

    def get_open_pointers(self) -> list[ProvenancePointer]:
        """Return list of distinct open evidence pointers."""
        with self._lock:
            # Filter distinct by pointer_id
            seen = set()
            distinct = []
            for ptr in self._open_pointers.values():
                if ptr.pointer_id not in seen:
                    seen.add(ptr.pointer_id)
                    distinct.append(ptr)
            return distinct

    def clear(self) -> None:
        """Clear open evidence pointers."""
        with self._lock:
            self._open_pointers.clear()

    def extract_citations(self, output: str | dict[str, Any]) -> set[str]:
        """Parse candidate agent output and extract set of cited pointers C."""
        cited: set[str] = set()

        if isinstance(output, dict):
            # Check explicit citation fields
            if "citations" in output and isinstance(output["citations"], list):
                for c in output["citations"]:
                    if isinstance(c, str):
                        cited.add(c.strip())
                    elif isinstance(c, dict):
                        if "pointer_id" in c:
                            cited.add(c["pointer_id"])
                        elif "source_uri" in c:
                            cited.add(c["source_uri"])
            if "provenance_pointers" in output and isinstance(output["provenance_pointers"], list):
                for p in output["provenance_pointers"]:
                    if isinstance(p, str):
                        cited.add(p.strip())
                    elif isinstance(p, dict) and "pointer_id" in p:
                        cited.add(p["pointer_id"])

            text_to_scan = json.dumps(output)
        else:
            text_to_scan = str(output)

        # Regex scan text
        # 1. [cite: URI@hash:start-end]
        for m in self._CITATION_REGEX[0].finditer(text_to_scan):
            cited.add(m.group(1).strip())

        # 2. [text](uri#Lstart-Lend)
        for m in self._CITATION_REGEX[1].finditer(text_to_scan):
            uri = m.group(1).strip()
            if uri.startswith("http") or uri.endswith((".py", ".md", ".json", ".yaml", ".txt", ".sql")):
                start = m.group(2)
                end = m.group(3)
                if start and end:
                    cited.add(f"{uri}:{start}-{end}")
                cited.add(uri)

        # 3. @uri:start-end
        for m in self._CITATION_REGEX[2].finditer(text_to_scan):
            uri = m.group(1).strip()
            start = m.group(2)
            end = m.group(3)
            cited.add(f"{uri}:{start}-{end}")
            cited.add(uri)

        if not cited:
            for k in self._open_pointers:
                if len(k) > 4 and k in text_to_scan:
                    cited.add(k)

        return cited

    def validate_assertion(
        self,
        candidate_output: str | dict[str, Any],
        is_actionable: bool = True,
        raise_on_violation: bool = False,
    ) -> tuple[bool, Any]:
        """Enforce inclusion property: Valid(a) <=> C ⊆ O and (C != ∅ for actionable).

        Returns:
            (True, candidate_output) if assertion is valid.
            (False, AbstentionSignal) if ungrounded assertion is detected.
        """
        with self._lock:
            cited = self.extract_citations(candidate_output)
            open_keys = set(self._open_pointers.keys())

            # Check 1: Non-empty cited pointers for actionable assertions
            if is_actionable and not cited:
                reason = "Actionable assertion made with empty citations (C = ∅). Grounded evidence required."
                return self._handle_violation(candidate_output, reason, list(cited), raise_on_violation)

            # Check 2: Inclusion property C ⊆ O
            unauthorized: list[str] = []
            for c in cited:
                # Match against pointer_id or source_uri
                matched = any(
                    c == k or c in k or k in c
                    for k in open_keys
                )
                if not matched:
                    unauthorized.append(c)

            if unauthorized:
                reason = (
                    f"Ungrounded assertion violation: Cited evidence C is not a subset of open evidence O (C ⊈ O). "
                    f"Unauthorized citations: {unauthorized}"
                )
                return self._handle_violation(candidate_output, reason, unauthorized, raise_on_violation)

            return True, candidate_output

    def _handle_violation(
        self,
        candidate_output: Any,
        reason: str,
        unauthorized: list[str],
        raise_on_violation: bool,
    ) -> tuple[bool, AbstentionSignal]:
        """Dispatch audit event and construct AbstentionSignal (a = ⊥)."""
        # Dispatch structured security event
        record_security_event(
            event_type=SecurityEventType.UNGROUNDED_ASSERTION_VIOLATION,
            source="tribune.memory.retrieval.CitationLockHarness",
            message=reason,
            severity="HIGH",
            details={
                "unauthorized_citations": unauthorized,
                "open_evidence_count": len(self._open_pointers),
                "open_keys": list(self._open_pointers.keys())[:10],
            },
            case_id=self.case_id,
        )

        signal = AbstentionSignal(
            is_abstention=True,
            diagnostic_error="UNGROUNDED_ASSERTION_VIOLATION",
            reason=reason,
            unauthorized_citations=unauthorized,
            open_citations_count=len(self._open_pointers),
            candidate_output=candidate_output,
        )

        if raise_on_violation:
            raise UngroundedAssertionViolationError(reason)

        return False, signal


class HierarchicalTraceRetriever:
    """Hierarchical trace resolver navigating from root assertions down to leaf provenance citations."""

    def __init__(self, harness: CitationLockHarness | None = None) -> None:
        self.harness = harness or CitationLockHarness()

    def resolve_leaf_sources(self, trace: ConsolidatedMemoryTrace) -> list[ProvenanceNode]:
        """Resolve all leaf ProvenanceNode sources for a given consolidated trace."""
        return trace.resolve_all_leaf_citations()

    def navigate_trace_hierarchy(self, trace: ConsolidatedMemoryTrace) -> dict[str, Any]:
        """Recursively navigate trace hierarchy and return structured tree with leaf citations."""
        return {
            "trace_id": trace.trace_id,
            "compaction_level": trace.compaction_level,
            "root_assertion": trace.root_assertion,
            "confidence_score": trace.confidence_score,
            "sub_traces": [self.navigate_trace_hierarchy(st) for st in trace.sub_traces],
            "leaf_citations": [
                {
                    "source_id": node.source_id,
                    "chunk_index": node.chunk_index,
                    "content_hash": node.content_hash,
                    "metadata": node.metadata,
                }
                for node in trace.resolve_all_leaf_citations()
            ],
        }


__all__ = [
    "ProvenancePointer",
    "ProvenancedTuple",
    "AbstentionSignal",
    "UngroundedAssertionViolationError",
    "CitationLockHarness",
    "HierarchicalTraceRetriever",
    "StateDeltaNode",
    "RetentionAwareRetriever",
]


# --------------------------------------------------------------------------- #
# Protocol-Aware Retention: associative search restricted to StateDelta nodes
# --------------------------------------------------------------------------- #


@dataclass
class StateDeltaNode:
    """Retrievable unit for default associative search (never raw observations)."""

    node_id: str
    text: str  # compact StateDelta rendering (dense factual update)
    timestamp: float = field(default_factory=time.time)
    causal_refs: list[str] = field(default_factory=list)
    verified_provenance: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    is_ephemeral: bool = False  # always False for indexed nodes; True blocks indexing


class RetentionAwareRetriever:
    """Default associative retrieval over StateDelta nodes only.

    - EphemeralObservation content is never indexed here.
    - Raw observations require explicit provenance lookup (see
      ``MemoryEventsTimeline.fetch_cold_observation``) or a privileged debug
      query with ``include_ephemeral=True`` + ``privileged=True``.
    - Ranking favors StateDelta recency, causal relevance, verified provenance.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, StateDeltaNode] = {}
        self._quarantined_raw_attempts = 0
        self._lock = threading.RLock()

    def index_state_delta(
        self,
        node_id: str,
        delta_text: str,
        timestamp: float | None = None,
        causal_refs: list[str] | None = None,
        verified_provenance: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> StateDeltaNode:
        with self._lock:
            node = StateDeltaNode(
                node_id=node_id,
                text=delta_text,
                timestamp=timestamp if timestamp is not None else time.time(),
                causal_refs=list(causal_refs or []),
                verified_provenance=bool(verified_provenance),
                metadata=dict(metadata or {}),
                is_ephemeral=False,
            )
            self._nodes[node_id] = node
            return node

    def index_timeline_delta(self, delta: Any) -> StateDeltaNode:
        """Index a StateDelta dataclass (duck-typed) from timeline retention."""
        to_dict = delta.to_dict() if hasattr(delta, "to_dict") else dict(delta)
        return self.index_state_delta(
            node_id=str(to_dict.get("event_id", f"node_{len(self._nodes)}")),
            delta_text=json.dumps(to_dict, sort_keys=True, default=str),
            timestamp=float(to_dict.get("timestamp", time.time())),
            causal_refs=list(to_dict.get("causal_refs", []) or []),
            verified_provenance=bool(to_dict.get("observation_digest")),
            metadata={"kind": "StateDelta", "tool": to_dict.get("tool_name", "")},
        )

    def try_index_raw_observation(self, node_id: str, raw_text: str) -> bool:
        """Refused by policy: raw observations are excluded from default search."""
        with self._lock:
            self._quarantined_raw_attempts += 1
        record_security_event(
            event_type=SecurityEventType.UNGROUNDED_ASSERTION_VIOLATION,
            source="tribune.memory.retrieval.RetentionAwareRetriever",
            message=f"Blocked raw EphemeralObservation indexing attempt for '{node_id}'.",
            severity="MEDIUM",
            details={"node_id": node_id},
        )
        return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_ephemeral: bool = False,
        privileged: bool = False,
        now: float | None = None,
    ) -> list[tuple[StateDeltaNode, float]]:
        """Lexical associative search over StateDelta nodes.

        ``include_ephemeral=True`` requires ``privileged=True``; otherwise raw
        content is never returned (defense: noisy logs must not dilute recall).
        """
        with self._lock:
            if include_ephemeral and not privileged:
                raise PermissionError(
                    "Raw observation retrieval requires privileged debug query."
                )
            now_ts = now if now is not None else time.time()
            q_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", query)}
            scored: list[tuple[StateDeltaNode, float]] = []
            for node in self._nodes.values():
                n_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", node.text)}
                overlap = len(q_terms & n_terms)
                base = overlap / max(1, len(q_terms))
                # recency boost (30-day half-life approx)
                age_s = max(0.0, now_ts - node.timestamp)
                recency = 1.0 / (1.0 + age_s / 2_592_000.0)
                causal = 1.0 + 0.1 * min(5, len(node.causal_refs))
                provenance = 1.25 if node.verified_provenance else 1.0
                score = base * (0.5 + 0.5 * recency) * causal * provenance
                scored.append((node, round(score, 6)))
            scored.sort(key=lambda kv: (kv[1], kv[0].timestamp), reverse=True)
            return scored[:top_k]

    def inject_context(self, query: str, top_k: int = 3) -> str:
        """Dense factual context injection (StateDelta summaries, not raw logs)."""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "=== RETENTION-AWARE CONTEXT (no StateDelta matches) ==="
        lines = ["=== RETENTION-AWARE CONTEXT (StateDelta only) ==="]
        for node, score in hits:
            lines.append(f"• [{node.node_id}] (score={score:.3f}) {node.text[:400]}")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "indexed_state_deltas": len(self._nodes),
                "blocked_raw_attempts": self._quarantined_raw_attempts,
            }
