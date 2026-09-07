"""Associative Graph Subsystem: EntityEventGraph.

Tracks architectural entities and directional dependency relationships:
- Entity Kinds: module, class, method, microservice, database_schema
- Directional Edge Types: imports, calls, inherits, mutates

Provides multi-hop graph traversal (k-hop neighborhood queries) for structural context discovery.
All entities and relation edges are encapsulated as provenanced tuples: τ = (x, π(x)).
"""

from __future__ import annotations

import collections
import enum
import threading
from dataclasses import dataclass, field
from typing import Any

from ..memory.retrieval import ProvenancedTuple, ProvenancePointer


class EntityKind(str, enum.Enum):
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    MICROSERVICE = "microservice"
    DATABASE_SCHEMA = "database_schema"


class DependencyEdgeType(str, enum.Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    MUTATES = "mutates"


@dataclass
class EntityNode:
    """Architectural entity encapsulated as a provenanced tuple."""

    entity_id: str
    kind: EntityKind
    name: str
    provenanced_tuple: ProvenancedTuple[dict[str, Any]]
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def provenance(self) -> ProvenancePointer:
        return self.provenanced_tuple.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind.value,
            "name": self.name,
            "attributes": self.attributes,
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class DependencyEdge:
    """Directional dependency edge between entities encapsulated with provenance."""

    source_id: str
    target_id: str
    edge_type: DependencyEdgeType
    provenanced_tuple: ProvenancedTuple[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def provenance(self) -> ProvenancePointer:
        return self.provenanced_tuple.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
        }


class EntityEventGraph:
    """Associative graph tracking architectural entities and directional dependencies."""

    def __init__(self, graph_id: str = "repo_entity_graph") -> None:
        self.graph_id = graph_id
        self._nodes: dict[str, EntityNode] = {}
        self._adjacency_out: dict[str, list[DependencyEdge]] = collections.defaultdict(list)
        self._adjacency_in: dict[str, list[DependencyEdge]] = collections.defaultdict(list)
        self._lock = threading.RLock()

    def add_entity(
        self,
        entity_id: str,
        kind: EntityKind | str,
        name: str,
        source_uri: str,
        commit_hash: str = "HEAD",
        line_start: int = 1,
        line_end: int = 1,
        attributes: dict[str, Any] | None = None,
    ) -> EntityNode:
        """Register an entity node with provenanced tuple encapsulation."""
        with self._lock:
            ekind = EntityKind(kind) if isinstance(kind, str) else kind
            ptr = ProvenancePointer.create(
                source_uri=source_uri,
                commit_hash=commit_hash,
                line_start=line_start,
                line_end=line_end,
            )
            p_tuple = ProvenancedTuple(data=attributes or {}, provenance=ptr)
            node = EntityNode(
                entity_id=entity_id,
                kind=ekind,
                name=name,
                provenanced_tuple=p_tuple,
                attributes=attributes or {},
            )
            self._nodes[entity_id] = node
            return node

    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        edge_type: DependencyEdgeType | str,
        source_uri: str | None = None,
        commit_hash: str = "HEAD",
        line_start: int = 1,
        line_end: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> DependencyEdge:
        """Add a directional dependency edge: source -> target with provenance."""
        with self._lock:
            etype = DependencyEdgeType(edge_type) if isinstance(edge_type, str) else edge_type
            uri = source_uri or (self._nodes[source_id].provenance.source_uri if source_id in self._nodes else "graph://edge")
            ptr = ProvenancePointer.create(
                source_uri=uri,
                commit_hash=commit_hash,
                line_start=line_start,
                line_end=line_end,
            )
            p_tuple = ProvenancedTuple(data=metadata or {}, provenance=ptr)
            edge = DependencyEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=etype,
                provenanced_tuple=p_tuple,
                metadata=metadata or {},
            )
            self._adjacency_out[source_id].append(edge)
            self._adjacency_in[target_id].append(edge)
            return edge

    def get_entity(self, entity_id: str) -> EntityNode | None:
        with self._lock:
            return self._nodes.get(entity_id)

    def get_k_hop_neighborhood(
        self,
        start_entity_id: str,
        k: int = 1,
        direction: str = "out",  # "out" | "in" | "both"
        edge_types: list[DependencyEdgeType | str] | None = None,
    ) -> dict[str, Any]:
        """Multi-hop breadth-first graph traversal for structural context discovery.

        Returns:
            dict with:
                "nodes": list of EntityNode dicts
                "edges": list of DependencyEdge dicts
                "distance": dict mapping entity_id -> hop distance
        """
        with self._lock:
            if start_entity_id not in self._nodes:
                return {"nodes": [], "edges": [], "distance": {}}

            allowed_types = (
                {DependencyEdgeType(t) if isinstance(t, str) else t for t in edge_types}
                if edge_types is not None
                else None
            )

            visited_nodes: dict[str, int] = {start_entity_id: 0}
            collected_edges: list[DependencyEdge] = []
            queue = collections.deque([(start_entity_id, 0)])

            while queue:
                curr_id, dist = queue.popleft()
                if dist >= k:
                    continue

                # Gather outgoing edges
                candidate_edges: list[DependencyEdge] = []
                if direction in ("out", "both"):
                    candidate_edges.extend(self._adjacency_out.get(curr_id, []))
                if direction in ("in", "both"):
                    candidate_edges.extend(self._adjacency_in.get(curr_id, []))

                for edge in candidate_edges:
                    if allowed_types and edge.edge_type not in allowed_types:
                        continue

                    # Determine neighbor
                    neighbor_id = edge.target_id if edge.source_id == curr_id else edge.source_id
                    collected_edges.append(edge)

                    if neighbor_id not in visited_nodes:
                        visited_nodes[neighbor_id] = dist + 1
                        queue.append((neighbor_id, dist + 1))

            nodes_list = [self._nodes[nid].to_dict() for nid in visited_nodes if nid in self._nodes]
            # Deduplicate edges
            seen_edges = set()
            distinct_edges = []
            for e in collected_edges:
                key = (e.source_id, e.target_id, e.edge_type.value)
                if key not in seen_edges:
                    seen_edges.add(key)
                    distinct_edges.append(e.to_dict())

            return {
                "nodes": nodes_list,
                "edges": distinct_edges,
                "distance": visited_nodes,
            }

    def find_path(self, start_id: str, target_id: str) -> list[DependencyEdge] | None:
        """Find shortest directional path between two entities."""
        with self._lock:
            if start_id not in self._nodes or target_id not in self._nodes:
                return None

            visited: set[str] = {start_id}
            queue = collections.deque([(start_id, [])])

            while queue:
                curr_id, path = queue.popleft()
                if curr_id == target_id:
                    return path

                for edge in self._adjacency_out.get(curr_id, []):
                    nxt = edge.target_id
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [edge]))
            return None


__all__ = [
    "EntityKind",
    "DependencyEdgeType",
    "EntityNode",
    "DependencyEdge",
    "EntityEventGraph",
]
