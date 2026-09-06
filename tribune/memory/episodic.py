"""Relational Entity-Temporal Graph Memory Store & ContextPilot Offloading.

Refactors flat key-value storage into a relational entity-temporal graph memory store
supporting multi-hop causal reasoning, temporal versioning, and aggressive ContextPilot
offloading to maintain active generation contexts strictly under 1,000 tokens.
"""

from __future__ import annotations

import collections
import threading
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
]
