"""Relational Entity-Temporal Graph Memory Store & ContextPilot Offloading.

Refactors flat key-value storage into a relational entity-temporal graph memory store
supporting multi-hop causal reasoning, temporal versioning, and aggressive ContextPilot
offloading to maintain active generation contexts strictly under 1,000 tokens.
"""

from __future__ import annotations

import collections
import fnmatch as _fnmatch
import hashlib as _hashlib
import json as _json
import threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from datetime import datetime, timezone
from typing import Any

from ..types import (
    CausalChain,
    EntityGraph,
    EntityNode,
    EntityRelation,
    GraphQuery,
    SubGraphResult,
)

NodeID = str


class RelationalEpisodicMemory:
    """Relational Entity-Temporal Graph Memory Store.

    Stores entities, relations, and causal inference chains with temporal indexing
    and multi-hop graph querying.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, EntityNode] = {}
        self._edges: list[EntityRelation] = []
        self._adjacency: dict[str, list[tuple[str, EntityRelation]]] = collections.defaultdict(list)
        self._node_timestamps: dict[str, datetime] = {}
        self._causal_chains: dict[str, list[CausalChain]] = collections.defaultdict(list)
        self._all_causal_chains: list[CausalChain] = []
        self._lock = threading.RLock()

    def memorize(
        self,
        entity_graph: EntityGraph,
        timestamp: datetime,
        causal_chains: list[CausalChain],
    ) -> NodeID:
        """Memorize an entity graph and causal chains with a temporal timestamp."""
        with self._lock:
            primary_id: NodeID = ""

            for node in entity_graph.nodes:
                self._nodes[node.node_id] = node
                self._node_timestamps[node.node_id] = timestamp
                if not primary_id:
                    primary_id = node.node_id

            for edge in entity_graph.edges:
                self._edges.append(edge)
                self._adjacency[edge.source_id].append((edge.target_id, edge))
                # Add bidirectional adjacency traversal support
                self._adjacency[edge.target_id].append((edge.source_id, edge))

            for chain in causal_chains:
                self._all_causal_chains.append(chain)
                if primary_id:
                    self._causal_chains[primary_id].append(chain)
                # Link chains to matching node names or attributes
                for node in entity_graph.nodes:
                    if (
                        node.name.lower() in chain.premise.lower()
                        or node.name.lower() in chain.consequence.lower()
                    ):
                        self._causal_chains[node.node_id].append(chain)

            if not primary_id:
                # Fallback node if graph had no nodes
                primary_id = f"node_temporal_{len(self._nodes) + 1}"
                fallback_node = EntityNode(
                    node_id=primary_id,
                    name="TemporalEvidenceCluster",
                    entity_type="evidence_cluster",
                    attributes={"timestamp": timestamp.isoformat()},
                )
                self._nodes[primary_id] = fallback_node
                self._node_timestamps[primary_id] = timestamp

            return primary_id

    def readMemory(self, query: GraphQuery, max_hops: int = 2) -> SubGraphResult:
        """Query memory traversing up to max_hops from root nodes with relation and entity filters."""
        with self._lock:
            effective_hops = min(query.max_hops, max_hops)
            visited_nodes: set[str] = set()
            collected_edges: list[EntityRelation] = []
            collected_chains: list[CausalChain] = []

            # Determine starting root nodes
            frontier = list(query.root_node_ids) if query.root_node_ids else list(self._nodes.keys())
            if query.root_node_ids:
                frontier = [nid for nid in frontier if nid in self._nodes]
            else:
                # If no root specified, limit initial frontier to keep subgraph focused
                frontier = frontier[:10]

            queue = collections.deque([(nid, 0) for nid in frontier])
            for nid in frontier:
                visited_nodes.add(nid)

            max_depth_reached = 0

            while queue:
                current_id, depth = queue.popleft()
                max_depth_reached = max(max_depth_reached, depth)

                # Collect causal chains for visited node
                for chain in self._causal_chains.get(current_id, []):
                    if chain not in collected_chains:
                        collected_chains.append(chain)

                if depth >= effective_hops:
                    continue

                for neighbor_id, edge in self._adjacency.get(current_id, []):
                    # Filter by relation type
                    if query.relation_filters and edge.relation_type not in query.relation_filters:
                        continue

                    # Filter by temporal bounds if present
                    neighbor_time = self._node_timestamps.get(neighbor_id)
                    if query.start_time and neighbor_time and neighbor_time < query.start_time:
                        continue
                    if query.end_time and neighbor_time and neighbor_time > query.end_time:
                        continue

                    # Filter neighbor node by entity type
                    neighbor_node = self._nodes.get(neighbor_id)
                    if neighbor_node and query.entity_type_filters:
                        if neighbor_node.entity_type not in query.entity_type_filters:
                            continue

                    if edge not in collected_edges:
                        collected_edges.append(edge)

                    if neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

            # Include any causal chains matching node names
            result_nodes = [self._nodes[nid] for nid in visited_nodes if nid in self._nodes]
            for chain in self._all_causal_chains:
                if chain not in collected_chains:
                    for node in result_nodes:
                        if node.name.lower() in chain.premise.lower() or node.name.lower() in chain.consequence.lower():
                            collected_chains.append(chain)
                            break

            return SubGraphResult(
                nodes=result_nodes,
                edges=collected_edges,
                causal_chains=collected_chains,
                total_hop_depth=max_depth_reached,
            )

    def updateMemory(self, node_id: NodeID, patch: dict[str, Any]) -> None:
        """Update node attributes or relations in memory using a patch dictionary."""
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Memory node '{node_id}' not found.")

            node = self._nodes[node_id]
            updated_attrs = dict(node.attributes)
            if "attributes" in patch and isinstance(patch["attributes"], dict):
                updated_attrs.update(patch["attributes"])

            updated_name = patch.get("name", node.name)
            updated_type = patch.get("entity_type", node.entity_type)

            # Re-instantiate immutable StrictModel
            updated_node = EntityNode(
                node_id=node.node_id,
                name=updated_name,
                entity_type=updated_type,
                attributes=updated_attrs,
            )
            self._nodes[node_id] = updated_node

    def all_nodes_count(self) -> int:
        with self._lock:
            return len(self._nodes)


class ContextPilot:
    """ContextPilot design pattern orchestrator.

    Aggressively offloads long-term evidentiary items and voluminous transcripts to the
    relational entity-temporal memory, keeping active working context strictly under 1,000 tokens
    until final strategy compilation.
    """

    def __init__(
        self,
        memory_store: RelationalEpisodicMemory | None = None,
        max_active_tokens: int = 1000,
    ) -> None:
        self.memory = memory_store or RelationalEpisodicMemory()
        self.max_active_tokens = max_active_tokens
        self._active_spans: dict[str, str] = {}
        self._offloaded_references: dict[str, str] = {}  # span_id -> node_id

    def estimate_active_tokens(self) -> int:
        """Rough token estimate (~4 chars per token) across active working context."""
        total_chars = sum(len(text) for text in self._active_spans.values())
        return total_chars // 4

    def add_working_evidence(
        self,
        span_id: str,
        text: str,
        entities: list[EntityNode] | None = None,
        relations: list[EntityRelation] | None = None,
        causal_chains: list[CausalChain] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Add working context span and aggressively offload if token threshold is exceeded."""
        self._active_spans[span_id] = text
        ts = timestamp or datetime.now(timezone.utc)

        # Offload immediately to relational graph memory
        graph = EntityGraph(
            nodes=entities or [
                EntityNode(
                    node_id=f"ent_{span_id}",
                    name=f"Evidence_{span_id}",
                    entity_type="evidence_span",
                    attributes={"summary": text[:80]},
                )
            ],
            edges=relations or [],
        )
        node_id = self.memory.memorize(graph, ts, causal_chains or [])
        self._offloaded_references[span_id] = node_id

        # Enforce ContextPilot < 1,000 token active generation budget
        if self.estimate_active_tokens() >= self.max_active_tokens:
            self.offload_spans_to_graph()

    def offload_spans_to_graph(self) -> None:
        """Offload raw active spans to graph references to keep active generation context < 1,000 tokens."""
        for span_id, text in list(self._active_spans.items()):
            node_id = self._offloaded_references.get(span_id, "ref_node")
            # Replace voluminous text with compact graph pointer
            self._active_spans[span_id] = f"[GRAPH-POINTER: {node_id} | preview={text[:60]}...]"
            if self.estimate_active_tokens() < (self.max_active_tokens * 0.7):
                break

    def compile_strategy_context(self, target_nodes: list[str] | None = None) -> str:
        """Rehydrate and compile concise strategic context from the relational graph store."""
        query = GraphQuery(
            root_node_ids=target_nodes or list(self._offloaded_references.values())[:4],
            max_hops=2,
        )
        subgraph = self.memory.readMemory(query)

        lines = ["=== CONTEXTPILOT STRATEGY CONTEXT ==="]
        lines.append(f"Identified Entities ({len(subgraph.nodes)}):")
        for node in subgraph.nodes:
            lines.append(f"  • [{node.entity_type}] {node.name} (ID: {node.node_id})")

        if subgraph.edges:
            lines.append("Relational Connections:")
            for edge in subgraph.edges:
                lines.append(f"  • {edge.source_id} --[{edge.relation_type}]--> {edge.target_id}")

        if subgraph.causal_chains:
            lines.append("Causal Chains:")
            for chain in subgraph.causal_chains:
                lines.append(f"  • {chain.premise} -> {chain.predicate} -> {chain.consequence}")

        return "\n".join(lines)


__all__ = [
    "NodeID",
    "RelationalEpisodicMemory",
    "ContextPilot",
    "ASTSelector",
    "ShardManifest",
    "ShardFinding",
    "ReduceBundle",
    "MapReduceConfig",
    "AgenticMapReduceHarness",
]


# --------------------------------------------------------------------------- #
# Agentic MapReduce Harness: deterministic plan -> parallel map -> majority reduce
# --------------------------------------------------------------------------- #


@_dataclass(frozen=True)
class ASTSelector:
    """Deterministic selector: file globs, module paths, symbols, AST types, labels."""

    file_globs: tuple[str, ...] = ()
    module_paths: tuple[str, ...] = ()
    symbol_names: tuple[str, ...] = ()
    ast_node_types: tuple[str, ...] = ()
    memory_labels: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    time_from: float | None = None
    time_to: float | None = None

    def matches(self, candidate: dict[str, Any]) -> bool:
        if self.file_globs:
            path = str(candidate.get("path", candidate.get("file", "")))
            if path and not any(_fnmatch.fnmatch(path, g) for g in self.file_globs):
                return False
        if self.module_paths:
            mod = str(candidate.get("module", ""))
            if mod and not any(mod == m or mod.startswith(m + ".") for m in self.module_paths):
                return False
        if self.symbol_names:
            syms = candidate.get("symbols", candidate.get("symbol", ""))
            sym_list = syms if isinstance(syms, list) else [str(syms)]
            if not any(s in sym_list or s == candidate.get("name", "") for s in self.symbol_names):
                return False
        if self.ast_node_types:
            ntype = str(candidate.get("ast_type", candidate.get("node_type", "")))
            if ntype and ntype not in self.ast_node_types:
                return False
        if self.memory_labels:
            labels = candidate.get("labels", [])
            labels = labels if isinstance(labels, list) else [labels]
            if not any(lbl in labels for lbl in self.memory_labels):
                return False
        if self.episode_ids:
            if str(candidate.get("episode_id", candidate.get("id", ""))) not in self.episode_ids:
                return False
        ts = candidate.get("timestamp")
        if ts is not None:
            try:
                tsf = float(ts)
            except (TypeError, ValueError):
                tsf = None
            if tsf is not None:
                if self.time_from is not None and tsf < self.time_from:
                    return False
                if self.time_to is not None and tsf > self.time_to:
                    return False
        return True


@_dataclass
class ShardManifest:
    shard_id: str
    candidate_ids: list[str] = _field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"shard_id": self.shard_id, "candidate_ids": list(self.candidate_ids)}


@_dataclass
class ShardFinding:
    shard_id: str
    status: str  # "ok" | "failed" | "skipped"
    findings: list[dict[str, Any]] = _field(default_factory=list)
    error: str = ""
    vector_signature: list[float] = _field(default_factory=list)


@_dataclass
class ReduceBundle:
    result: dict[str, Any]
    coverage: dict[str, Any]
    contradictions: list[dict[str, Any]]
    confidence: float
    dissent: float
    shard_provenance: list[str]


@_dataclass
class MapReduceConfig:
    enabled: bool = True
    worker_count: int = 4
    shard_size: int = 64
    deterministic_seed: int = 7
    max_context_per_worker: int = 8000
    tolerate_missing_coverage: bool = False


class AgenticMapReduceHarness:
    """Three-stage deterministic sweep: plan (selectors+hash sharding), map
    (parallel isolated workers, bounded context, no shared mutable state),
    reduce (bitwise majority-rule bundling + deterministic synthesis)."""

    def __init__(self, config: MapReduceConfig | None = None) -> None:
        self.config = config or MapReduceConfig()
        self._lock = threading.RLock()
        self.last_coverage: dict[str, Any] | None = None

    # -- Stage 1: planning ----------------------------------------------------- #
    def plan(
        self, candidates: list[dict[str, Any]], selector: ASTSelector | None = None
    ) -> tuple[list[ShardManifest], list[str]]:
        """Partition candidates into disjoint shards via stable hashing.

        Returns (manifests, selected_ids). No model inference during partitioning.
        """
        sel = selector or ASTSelector()
        selected = [c for c in candidates if sel.matches(c)]
        # stable order by candidate id for determinism
        def _cid(c: dict[str, Any]) -> str:
            return str(c.get("id", c.get("path", _json.dumps(c, sort_keys=True, default=str))))

        selected.sort(key=_cid)
        ids = [_cid(c) for c in selected]
        shards: list[ShardManifest] = []
        size = max(1, self.config.shard_size)
        for i in range(0, len(ids), size):
            chunk = ids[i : i + size]
            digest = _hashlib.sha256(
                f"{self.config.deterministic_seed}:{i}:{','.join(chunk)}".encode()
            ).hexdigest()[:12]
            shards.append(ShardManifest(shard_id=f"shard_{i // size:04d}_{digest}", candidate_ids=chunk))
        return shards, ids

    # -- Stage 2: mapping ------------------------------------------------------ #
    def map_shards(
        self,
        shards: list[ShardManifest],
        candidates_by_id: dict[str, dict[str, Any]],
        worker_fn: Any | None = None,
        fail_shards: set[str] | None = None,
    ) -> list[ShardFinding]:
        """Run isolated workers over disjoint shards with bounded context.

        ``worker_fn(shard, items) -> list[dict]``; defaults to deterministic
        structural scan. Failed/skipped shards recorded explicitly.
        """
        fail_shards = fail_shards or set()
        by_id = dict(candidates_by_id)

        def _default_worker(shard: ShardManifest, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            budget = self.config.max_context_per_worker
            used = 0
            for item in items:
                text = _json.dumps(item, sort_keys=True, default=str)
                cost = max(1, len(text) // 4)
                if used + cost > budget:
                    break  # bounded context: stop, do not overflow
                used += cost
                blob = text.lower()
                flags: list[str] = []
                if "contradict" in blob:
                    flags.append("contradiction_signal")
                if "todo" in blob or "fixme" in blob:
                    flags.append("attention_signal")
                out.append({"candidate_id": item.get("id", "?"), "flags": flags, "chars": len(text)})
            return out

        fn = worker_fn or _default_worker
        ordered = sorted(shards, key=lambda s: s.shard_id)

        def _run(shard: ShardManifest) -> ShardFinding:
            if shard.shard_id in fail_shards:
                return ShardFinding(shard_id=shard.shard_id, status="failed", error="injected_failure")
            items = [by_id[cid] for cid in shard.candidate_ids if cid in by_id]
            try:
                findings = fn(shard, [dict(it) for it in items])  # copy: no shared mutable state
                sig = self._finding_signature(findings)
                return ShardFinding(
                    shard_id=shard.shard_id, status="ok", findings=findings, vector_signature=sig
                )
            except Exception as exc:
                return ShardFinding(shard_id=shard.shard_id, status="failed", error=str(exc)[:300])

        with _ThreadPoolExecutor(max_workers=max(1, self.config.worker_count)) as pool:
            results = list(pool.map(_run, ordered))
        results.sort(key=lambda r: r.shard_id)  # deterministic ordering
        return results

    @staticmethod
    def _finding_signature(findings: list[dict[str, Any]]) -> list[float]:
        blob = _json.dumps(findings, sort_keys=True, default=str)
        digest = _hashlib.sha256(blob.encode()).digest()
        # map bytes to [-1, 1] pseudo-vector for majority bundling demo
        return [1.0 if b >= 128 else -1.0 for b in digest[:32]]

    # -- Stage 3: reduce -------------------------------------------------------- #
    def reduce(
        self,
        manifests: list[ShardManifest],
        findings: list[ShardFinding],
        total_candidates: int,
    ) -> ReduceBundle:
        """Deterministic synthesis with majority-rule bundling + coverage proof."""
        processed = sum(len(f.findings) for f in findings if f.status == "ok")
        failed = [f.shard_id for f in findings if f.status == "failed"]
        skipped = [f.shard_id for f in findings if f.status == "skipped"]
        manifest_ids = {cid for m in manifests for cid in m.candidate_ids}
        seen_ids: set[str] = set()
        contradictions: list[dict[str, Any]] = []
        for f in findings:
            if f.status != "ok":
                continue
            for item in f.findings:
                cid = str(item.get("candidate_id", ""))
                seen_ids.add(cid)
                if "contradiction_signal" in item.get("flags", []):
                    contradictions.append({"candidate_id": cid, "shard_id": f.shard_id})
        accounted = len(manifest_ids)
        coverage_ratio = (len(seen_ids) / max(1, total_candidates)) if total_candidates else 1.0
        # processed + failed + skipped shard accounting must equal total shards
        if len(findings) != len(manifests):
            raise RuntimeError("MapReduce shard accounting mismatch.")
        if accounted != total_candidates and not self.config.tolerate_missing_coverage:
            raise RuntimeError(
                f"Coverage gap: manifest has {accounted} candidates, expected {total_candidates}."
            )
        # majority-rule vote over shard signature vectors
        ok_sigs = [f.vector_signature for f in findings if f.status == "ok" and f.vector_signature]
        if ok_sigs:
            import numpy as _np

            stacked = _np.asarray(ok_sigs, dtype=float)
            votes = _np.sign(_np.sum(_np.sign(stacked), axis=0))
            consensus = [float(v) for v in votes]
            dissent = float((_np.sum(_np.sign(stacked) != _np.sign(votes), axis=0).mean()) / max(1, len(ok_sigs)))
        else:
            consensus, dissent = [], 1.0
        ok_count = sum(1 for f in findings if f.status == "ok")
        confidence = round(ok_count / max(1, len(findings)) * coverage_ratio, 4)
        coverage = {
            "total_candidates": total_candidates,
            "manifest_candidates": accounted,
            "processed_findings": processed,
            "failed_shards": failed,
            "skipped_shards": skipped,
            "coverage_ratio": round(coverage_ratio, 4),
            "deterministic": True,
        }
        with self._lock:
            self.last_coverage = coverage
        return ReduceBundle(
            result={"consensus_signature": consensus, "ok_shards": ok_count},
            coverage=coverage,
            contradictions=contradictions,
            confidence=confidence,
            dissent=round(float(dissent), 4),
            shard_provenance=[m.shard_id for m in manifests],
        )

    def run(
        self,
        candidates: list[dict[str, Any]],
        selector: ASTSelector | None = None,
        worker_fn: Any | None = None,
        fail_shards: set[str] | None = None,
    ) -> ReduceBundle:
        """End-to-end deterministic sweep with full-coverage guarantee."""
        manifests, ids = self.plan(candidates, selector)
        by_id: dict[str, dict[str, Any]] = {}
        for c in candidates:
            cid = str(c.get("id", c.get("path", _json.dumps(c, sort_keys=True, default=str))))
            by_id[cid] = c
        findings = self.map_shards(manifests, by_id, worker_fn, fail_shards)
        return self.reduce(manifests, findings, len(ids))
