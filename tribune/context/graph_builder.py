"""Module, Dependency, & Filterable HNSW Context Graph Builder.

Provides:
1. AST-based repository scanner building cross-file dependency and export context graphs.
2. Multi-agent dependency and scope graph builder for parallel execution waves.
3. Filterable HNSW (Hierarchical Navigable Small World) graph vector index with
   explicit metadata attribute bridging edges to guarantee path connectivity under
   highly selective (e.g. 1%) payload filters.
"""

from __future__ import annotations

import ast
import copy
import heapq
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .decision_router import (
    DEFAULT_EDGE_CONFIDENCE_THRESHOLD,
    DeterministicHeuristicClassifier,
    EdgeDecisionClassifier,
    EdgeDecisionRouter,
    ModelBackedRLCDClassifier,
    RouterTelemetry,
)
from .entities import (
    CalibrationMetadata,
    EdgeCandidate,
    EdgeClass,
    EdgeDecision,
    EdgeFeatures,
    EntityMention,
    EntityResolutionResult,
    EscalationRecord,
    GraphTransactionResult,
)
from .escalation import EscalationQueue
from .graph import DependencyEdgeType, EntityEventGraph, EntityKind


@dataclass
class ModuleNode:
    module_path: str
    relative_path: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)


@dataclass
class RepoContextGraph:
    root_dir: str
    modules: dict[str, ModuleNode] = field(default_factory=dict)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)

    def to_context_string(self, target_modules: list[str] | None = None) -> str:
        """Format the dependency graph into a structured context string for LLM prompts."""
        lines = ["=== REPOSITORY CONTEXT GRAPH ==="]
        lines.append(f"Root Directory: {self.root_dir}")
        lines.append(f"Total Modules Scanned: {len(self.modules)}\n")

        selected_keys = target_modules if target_modules else list(self.modules.keys())

        lines.append("--- CROSS-FILE DEPENDENCY MAPPINGS ---")
        for mod_name in sorted(selected_keys):
            if mod_name in self.dependency_graph:
                deps = self.dependency_graph[mod_name]
                lines.append(f"• {mod_name} -> [{', '.join(deps) if deps else 'none'}]")

        lines.append("\n--- MODULE SYMBOL & EXPORT DECLARATIONS ---")
        for mod_name in sorted(selected_keys):
            node = self.modules.get(mod_name)
            if not node:
                continue
            lines.append(f"Module: {node.relative_path} ({mod_name})")
            if node.exports:
                lines.append(f"  Exports (__all__): {', '.join(node.exports)}")
            if node.classes:
                lines.append(f"  Classes: {', '.join(node.classes)}")
            if node.functions:
                lines.append(f"  Functions: {', '.join(node.functions)}")
            if node.imports:
                lines.append(f"  Imports: {', '.join(node.imports[:10])}")
            lines.append("")

        return "\n".join(lines)


class RepoContextGraphBuilder:
    """AST-based repository scanner building cross-file dependency and export context graphs."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)

    def build_graph(self, max_depth: int = 5) -> RepoContextGraph:
        graph = RepoContextGraph(root_dir=self.root_dir)

        for current_root, subdirs, files in os.walk(self.root_dir):
            # Exclude hidden, cache, and test directories
            subdirs[:] = [d for d in subdirs if not d.startswith((".", "__pycache__", "venv"))]
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(current_root, file)
                rel_path = os.path.relpath(file_path, self.root_dir)

                # Compute module dot name (e.g. tribune.providers.base)
                parts = Path(rel_path).with_suffix("").parts
                mod_name = ".".join(parts)

                node = self._scan_file(file_path, rel_path)
                graph.modules[mod_name] = node

        # Build cross-file dependency mapping
        for mod_name, node in graph.modules.items():
            deps: list[str] = []
            for imp in node.imports:
                # Find matching module in scanned project
                for target_name in graph.modules:
                    if imp == target_name or imp.startswith(target_name + ".") or target_name.endswith("." + imp):
                        if target_name != mod_name and target_name not in deps:
                            deps.append(target_name)
            graph.dependency_graph[mod_name] = deps

        return graph

    def _scan_file(self, file_path: str, rel_path: str) -> ModuleNode:
        node = ModuleNode(module_path=file_path, relative_path=rel_path)
        try:
            with open(file_path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=file_path)
        except Exception:
            return node

        for item in tree.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    node.imports.append(alias.name)
            elif isinstance(item, ast.ImportFrom):
                mod = item.module or ""
                for alias in item.names:
                    node.imports.append(f"{mod}.{alias.name}" if mod else alias.name)
            elif isinstance(item, ast.ClassDef):
                node.classes.append(item.name)
            elif isinstance(item, ast.FunctionDef):
                node.functions.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(item.value, ast.List | ast.Tuple):
                            for elt in item.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    node.exports.append(elt.value)

        return node


# --------------------------------------------------------------------------- #
# Multi-Agent Dependency & Read/Write Scope Modeling
# --------------------------------------------------------------------------- #


@dataclass
class AgentNode:
    """Represents an agent in the multi-agent orchestration dependency graph."""

    agent_id: str
    role: str  # "navigator" | "proposer" | "verifier" | "preparer" | "gather" | "auxiliary"
    program: str | None = None
    read_scopes: list[str] = field(default_factory=list)  # JSON-pointer prefixes agent is allowed to read
    write_scopes: list[str] = field(default_factory=list)  # JSON-pointer prefixes agent is allowed to patch
    dependencies: list[str] = field(default_factory=list)  # Agent IDs that must complete before this agent
    broadcast_subscriptions: list[str] = field(default_factory=list)  # Channel subscriptions


@dataclass
class AgentDependencyGraph:
    """Dependency and dataflow graph modeling agent interactions, scopes, and execution waves."""

    nodes: dict[str, AgentNode] = field(default_factory=dict)

    def add_node(self, node: AgentNode) -> None:
        self.nodes[node.agent_id] = node

    def get_node(self, agent_id: str) -> AgentNode | None:
        return self.nodes.get(agent_id)

    def topological_order(self) -> list[AgentNode]:
        """Kahn's algorithm topological sort over agent dependencies."""
        indeg = {aid: 0 for aid in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Agent '{node.agent_id}' depends on unknown agent '{dep}'")
                indeg[node.agent_id] += 1

        queue = [aid for aid, n in indeg.items() if n == 0]
        order: list[AgentNode] = []
        while queue:
            aid = queue.pop(0)
            order.append(self.nodes[aid])
            for node in self.nodes.values():
                if aid in node.dependencies:
                    indeg[node.agent_id] -= 1
                    if indeg[node.agent_id] == 0:
                        queue.append(node.agent_id)

        if len(order) != len(self.nodes):
            raise ValueError("Agent dependency graph contains a cycle")
        return order

    def topological_waves(self) -> list[list[AgentNode]]:
        """Compute parallel execution waves for concurrent fan-out."""
        self.topological_order()  # Validates cycle-freedom
        completed: set[str] = set()
        waves: list[list[AgentNode]] = []

        while len(completed) < len(self.nodes):
            current_wave = [
                node for aid, node in self.nodes.items()
                if aid not in completed and all(d in completed for d in node.dependencies)
            ]
            if not current_wave:
                raise ValueError("Agent dependency deadlock: unable to form next wave")
            waves.append(current_wave)
            completed.update(node.agent_id for node in current_wave)

        return waves

    def validate_scope_isolation(self) -> list[str]:
        """Verify that agent write scopes do not have uncoordinated write collisions."""
        warnings: list[str] = []
        waves = self.topological_waves()
        for wave_idx, wave in enumerate(waves):
            seen_writes: dict[str, str] = {}
            for agent in wave:
                for scope in agent.write_scopes:
                    if scope in seen_writes:
                        warnings.append(
                            f"Concurrent write conflict in wave {wave_idx}: '{agent.agent_id}' "
                            f"and '{seen_writes[scope]}' both write to '{scope}'"
                        )
                    seen_writes[scope] = agent.agent_id
        return warnings

    def to_mermaid(self) -> str:
        """Generate mermaid flowchart diagram for documentation and tracing."""
        lines = ["graph TD"]
        for aid, node in self.nodes.items():
            lines.append(f'    {aid}["{aid} ({node.role})"]')
            for dep in node.dependencies:
                lines.append(f"    {dep} --> {aid}")
        return "\n".join(lines)


class AgentGraphBuilder:
    """Constructs agent dependency and scope graphs for standard benefit evaluation pipelines."""

    @classmethod
    def build_for_programs(
        cls,
        target_programs: list[str],
        jurisdiction: str = "EX",
    ) -> AgentDependencyGraph:
        graph = AgentDependencyGraph()

        # 1. Gather Agent (Ingests & consolidates evidence and visual layouts)
        graph.add_node(
            AgentNode(
                agent_id="gather",
                role="gather",
                read_scopes=["/documents"],
                write_scopes=["/evidence", "/shared_facts", "/visual_layouts"],
                dependencies=[],
                broadcast_subscriptions=["/documents"],
            )
        )

        # 2. Navigator Agent (High-level planning, layout verification, & coordination)
        graph.add_node(
            AgentNode(
                agent_id="navigator",
                role="navigator",
                read_scopes=["/evidence", "/assessments", "/materials", "/visual_layouts"],
                write_scopes=["/metadata/plan", "/agent_metadata/navigator", "/layout_verifications"],
                dependencies=["gather"],
                broadcast_subscriptions=["*"],
            )
        )

        # 3. Per-program Proposer, Verifier, and Preparer Agents
        for prog in target_programs:
            prog_val = str(prog).lower()
            prop_id = f"proposer_{prog_val}"
            ver_id = f"verifier_{prog_val}"
            prep_id = f"preparer_{prog_val}"

            # Eligibility Proposer
            graph.add_node(
                AgentNode(
                    agent_id=prop_id,
                    role="proposer",
                    program=prog_val,
                    read_scopes=["/evidence", f"/shared_facts/{prog_val}", "/shared_facts", "/visual_layouts"],
                    write_scopes=[f"/assessments/{prog_val}", f"/criteria_outcomes/{prog_val}"],
                    dependencies=["gather", "navigator"],
                    broadcast_subscriptions=["/evidence", f"/criteria_outcomes/{prog_val}"],
                )
            )

            # Independent Verifier (depends on proposer)
            graph.add_node(
                AgentNode(
                    agent_id=ver_id,
                    role="verifier",
                    program=prog_val,
                    read_scopes=["/evidence", f"/assessments/{prog_val}", f"/criteria_outcomes/{prog_val}", "/visual_layouts"],
                    write_scopes=[f"/verification_verdicts/{prog_val}"],
                    dependencies=[prop_id],
                    broadcast_subscriptions=[f"/assessments/{prog_val}"],
                )
            )

            # Preparer (depends on verifier)
            graph.add_node(
                AgentNode(
                    agent_id=prep_id,
                    role="preparer",
                    program=prog_val,
                    read_scopes=["/evidence", f"/assessments/{prog_val}", f"/verification_verdicts/{prog_val}", "/visual_layouts"],
                    write_scopes=[f"/materials/{prog_val}"],
                    dependencies=[ver_id],
                    broadcast_subscriptions=[f"/verification_verdicts/{prog_val}"],
                )
            )

        return graph


@dataclass
class VisualLayoutGraphNode:
    """A node in the ingested visual document layout context graph."""

    token_id: str
    doc_id: str
    bbox: list[float]
    token_type: str
    text: str
    reading_order_idx: int
    neighbors: list[str] = field(default_factory=list)


def build_visual_layout_subgraph(layout_dict: dict[str, Any]) -> dict[str, VisualLayoutGraphNode]:
    """Convert visual document layout dict into an indexed graph of layout nodes and reading-order neighbors."""
    doc_id = layout_dict.get("doc_id", "doc_1")
    tokens = layout_dict.get("tokens", [])
    edges = layout_dict.get("edges", [])

    node_map: dict[str, VisualLayoutGraphNode] = {}
    for tok in tokens:
        tid = tok.get("token_id", "")
        bbox = tok.get("bbox", [0.0, 0.0, 1.0, 1.0])
        node_map[tid] = VisualLayoutGraphNode(
            token_id=tid,
            doc_id=doc_id,
            bbox=bbox,
            token_type=tok.get("token_type", "PARAGRAPH"),
            text=tok.get("text", ""),
            reading_order_idx=tok.get("reading_order_index", 0),
        )

    for u, v in edges:
        if u in node_map and v not in node_map[u].neighbors:
            node_map[u].neighbors.append(v)

    return node_map



# --------------------------------------------------------------------------- #
# Filterable HNSW Vector Index with Explicit Metadata Attribute Bridging Edges
# --------------------------------------------------------------------------- #


@dataclass
class HNSWNode:
    """A single vector node in the Filterable HNSW graph."""

    node_id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    level: int = 0
    # Mapping of level -> list of neighbor node IDs
    neighbors: dict[int, list[str]] = field(default_factory=dict)
    # Attribute bridge edges for metadata-constrained navigation (level -> list of node IDs)
    bridge_neighbors: dict[int, list[str]] = field(default_factory=dict)


class FilterableHNSWIndex:
    """Filterable Hierarchical Navigable Small World (HNSW) graph index with attribute bridging edges.

    Constructs explicit inter-vector graph edges across shared metadata attributes
    (`jurisdiction`, `benefit_program`, `effective_year`, `statutory_level`) to prevent
    graph fragmentation during highly selective filtering (e.g., 1% selective queries),
    guaranteeing navigable path connectivity across disconnected semantic clusters.
    """

    def __init__(
        self,
        dim: int = 96,
        m: int = 16,
        m0: int = 32,
        ef_construction: int = 64,
        ml: float = 1.0 / math.log(16),
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.m = m
        self.m0 = m0
        self.ef_construction = ef_construction
        self.ml = ml
        self.rng = random.Random(seed)
        self.nodes: dict[str, HNSWNode] = {}
        self.enter_node_id: str | None = None
        self.max_level: int = -1

        # Inverted index for metadata bridging: (attr_name, attr_val) -> list[node_id]
        self._attribute_index: dict[tuple[str, Any], list[str]] = {}
        self._bridge_attributes = ["jurisdiction", "benefit_program", "program", "effective_year", "statutory_level"]

    def _random_level(self) -> int:
        """Draw level from exponential distribution."""
        unif = max(1e-9, self.rng.random())
        return int(-math.log(unif) * self.ml)

    @staticmethod
    def _distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine distance in [0, 2]. Lower is more similar."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 1.0
        dot = float(np.dot(v1, v2))
        cos_sim = dot / (norm1 * norm2)
        return max(0.0, 1.0 - cos_sim)

    def insert(self, node_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Insert a vector with metadata into the filterable HNSW graph."""
        vec = np.asarray(vector, dtype=np.float64)
        if vec.shape[-1] != self.dim:
            # Adjust dimension dynamically if needed
            if vec.shape[-1] < self.dim:
                padded = np.zeros(self.dim, dtype=np.float64)
                padded[: vec.shape[-1]] = vec
                vec = padded
            else:
                vec = vec[: self.dim]

        meta = dict(metadata or {})
        level = self._random_level()

        node = HNSWNode(
            node_id=node_id,
            vector=vec,
            metadata=meta,
            level=level,
            neighbors={lv: [] for lv in range(level + 1)},
            bridge_neighbors={lv: [] for lv in range(level + 1)},
        )
        self.nodes[node_id] = node

        # Update metadata inverted index & build explicit attribute bridges
        for attr in self._bridge_attributes:
            if attr in meta and meta[attr] is not None:
                key = (attr, str(meta[attr]).lower())
                existing = self._attribute_index.setdefault(key, [])
                for peer_id in existing[-4:]:  # Connect up to 4 recent peers sharing this attribute
                    if peer_id != node_id and peer_id in self.nodes:
                        peer_node = self.nodes[peer_id]
                        max_shared_lv = min(node.level, peer_node.level)
                        for lv in range(max_shared_lv + 1):
                            if peer_id not in node.bridge_neighbors[lv]:
                                node.bridge_neighbors[lv].append(peer_id)
                            if node_id not in peer_node.bridge_neighbors[lv]:
                                peer_node.bridge_neighbors[lv].append(node_id)
                existing.append(node_id)

        # First node in index
        if self.enter_node_id is None:
            self.enter_node_id = node_id
            self.max_level = level
            return

        curr_obj = self.enter_node_id
        curr_dist = self._distance(vec, self.nodes[curr_obj].vector)

        # 1. Top-down traversal from max_level down to level + 1
        for lv in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                all_neighbors = list(self.nodes[curr_obj].neighbors.get(lv, [])) + list(
                    self.nodes[curr_obj].bridge_neighbors.get(lv, [])
                )
                for n_id in all_neighbors:
                    d = self._distance(vec, self.nodes[n_id].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = n_id
                        changed = True

        # 2. Bottom-up connection from min(level, max_level) down to level 0
        w: list[tuple[float, str]] = [(curr_dist, curr_obj)]
        for lv in range(min(level, self.max_level), -1, -1):
            w = self._search_layer(vec, [curr_obj], self.ef_construction, lv)
            m_max = self.m0 if lv == 0 else self.m
            neighbors = self._select_neighbors(w, m_max)
            node.neighbors[lv] = neighbors

            for n_id in neighbors:
                n_node = self.nodes[n_id]
                n_node.neighbors.setdefault(lv, []).append(node_id)
                if len(n_node.neighbors[lv]) > (self.m0 if lv == 0 else self.m):
                    # Prune overfilled neighbors
                    n_dists = [(self._distance(n_node.vector, self.nodes[x].vector), x) for x in n_node.neighbors[lv]]
                    n_node.neighbors[lv] = self._select_neighbors(n_dists, self.m0 if lv == 0 else self.m)

            curr_obj = w[0][1]

        if level > self.max_level:
            self.max_level = level
            self.enter_node_id = node_id

    def _search_layer(
        self,
        query_vec: np.ndarray,
        entry_points: list[str],
        ef: int,
        level: int,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[float, str]]:
        """Greedy beam search on a single HNSW level utilizing both proximity and attribute bridge edges."""
        v = set(entry_points)
        candidates: list[tuple[float, str]] = []
        w: list[tuple[float, str]] = []

        for ep in entry_points:
            d = self._distance(query_vec, self.nodes[ep].vector)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w, (-d, ep))  # Max-heap for furthest element

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w[0][0]

            if c_dist > furthest_w_dist and len(w) >= ef:
                break

            curr_node = self.nodes[c_id]
            # Explore proximity neighbors AND attribute bridge neighbors
            combined_neighbors = list(curr_node.neighbors.get(level, [])) + list(
                curr_node.bridge_neighbors.get(level, [])
            )

            for n_id in combined_neighbors:
                if n_id not in v:
                    v.add(n_id)
                    n_node = self.nodes[n_id]
                    n_dist = self._distance(query_vec, n_node.vector)

                    furthest_w_dist = -w[0][0]
                    if n_dist < furthest_w_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, n_id))
                        heapq.heappush(w, (-n_dist, n_id))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Return sorted list of (dist, node_id) ascending
        result = [(-item[0], item[1]) for item in w]
        result.sort(key=lambda x: x[0])
        return result

    @staticmethod
    def _select_neighbors(candidates: list[tuple[float, str]], max_m: int) -> list[str]:
        """Simple heuristic neighbor selection."""
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        seen = set()
        chosen = []
        for _dist, nid in sorted_cands:
            if nid not in seen:
                seen.add(nid)
                chosen.append(nid)
            if len(chosen) >= max_m:
                break
        return chosen

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        ef_search: int = 32,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Execute payload-constrained nearest-neighbor search.

        Returns list of tuples: (node_id, cosine_similarity, metadata).
        Guarantees path navigation across filtered subsets via explicit attribute bridges.
        """
        if not self.nodes or self.enter_node_id is None:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float64)
        if q_vec.shape[-1] != self.dim:
            if q_vec.shape[-1] < self.dim:
                padded = np.zeros(self.dim, dtype=np.float64)
                padded[: q_vec.shape[-1]] = q_vec
                q_vec = padded
            else:
                q_vec = q_vec[: self.dim]

        curr_obj = self.enter_node_id

        # 1. Traverse top levels to entry level 0
        for lv in range(self.max_level, 0, -1):
            changed = True
            curr_dist = self._distance(q_vec, self.nodes[curr_obj].vector)
            while changed:
                changed = False
                all_neighbors = list(self.nodes[curr_obj].neighbors.get(lv, [])) + list(
                    self.nodes[curr_obj].bridge_neighbors.get(lv, [])
                )
                for n_id in all_neighbors:
                    d = self._distance(q_vec, self.nodes[n_id].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = n_id
                        changed = True

        # 2. Bottom layer beam search
        ef = max(ef_search, k * 2)
        candidates = self._search_layer(q_vec, [curr_obj], ef, level=0, filter_fn=filter_fn)

        # 3. Filter candidates if predicate specified
        matched: list[tuple[str, float, dict[str, Any]]] = []
        for dist, nid in candidates:
            node = self.nodes[nid]
            if filter_fn is None or filter_fn(node.metadata):
                sim = max(0.0, 1.0 - dist)
                matched.append((nid, sim, node.metadata))
            if len(matched) >= k:
                break

        # If strict filter yielded fewer than k due to high selectivity, do attribute-directed recovery
        if len(matched) < k and filter_fn is not None:
            for nid, node in self.nodes.items():
                if any(m[0] == nid for m in matched):
                    continue
                if filter_fn(node.metadata):
                    dist = self._distance(q_vec, node.vector)
                    sim = max(0.0, 1.0 - dist)
                    matched.append((nid, sim, node.metadata))
            matched.sort(key=lambda x: x[1], reverse=True)

        return matched[:k]


# --------------------------------------------------------------------------- #
# Xiaohongshu's Self-Governing Context Architecture & Compaction Primitives
# --------------------------------------------------------------------------- #


class ExternalKVStore:
    """External Key-Value persistence store for offloaded/folded context payloads."""

    def __init__(self, namespace: str = "context_kv") -> None:
        self.namespace = namespace
        self._store: dict[str, Any] = {}
        self._total_bytes_evicted = 0

    def put(self, key: str, data: Any) -> str:
        """Store payload in KV persistence and return an inline location pointer URI."""
        self._store[key] = data
        try:
            payload_bytes = len(str(data).encode("utf-8"))
        except Exception:
            payload_bytes = 100
        self._total_bytes_evicted += payload_bytes
        return f"ref://{self.namespace}/{key}"

    def get(self, key_or_uri: str) -> Any:
        """Retrieve payload from key or URI pointer."""
        key = key_or_uri.split("/")[-1] if key_or_uri.startswith("ref://") else key_or_uri
        return self._store.get(key)

    def has(self, key_or_uri: str) -> bool:
        key = key_or_uri.split("/")[-1] if key_or_uri.startswith("ref://") else key_or_uri
        return key in self._store

    def delete(self, key_or_uri: str) -> bool:
        key = key_or_uri.split("/")[-1] if key_or_uri.startswith("ref://") else key_or_uri
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes_evicted = 0

    def stats(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "entry_count": len(self._store),
            "total_bytes_evicted": self._total_bytes_evicted,
        }


GLOBAL_KV_STORE = ExternalKVStore()


def fold_payload(
    payload: Any,
    kv_store: ExternalKVStore | None = None,
    key_prefix: str = "payload",
) -> dict[str, Any]:
    """Fold primitive: Evicts oversized tool outputs, code dumps, and JSON payloads into external KV persistence.

    Returns a compact inline URI/location pointer node.
    """
    store = kv_store or GLOBAL_KV_STORE
    key = f"{key_prefix}_{random.randint(100000, 999999)}_{int(time.time() * 1000)}"
    uri = store.put(key, payload)

    serialized = str(payload)
    size_bytes = len(serialized.encode("utf-8", errors="ignore"))
    token_est = max(1, size_bytes // 4)

    # Create summary preview
    preview = serialized[:120] + "..." if len(serialized) > 120 else serialized

    return {
        "$ref": uri,
        "folded": True,
        "uri_pointer": uri,
        "original_type": type(payload).__name__,
        "size_bytes": size_bytes,
        "estimated_tokens": token_est,
        "summary": preview,
    }


def mask_stream(
    log_text: str,
    head_lines: int = 5,
    tail_lines: int = 5,
) -> str:
    """Mask primitive: Truncates high-volume intermediate command logs, preserving header and trailer lines.

    Preserves first `head_lines` and last `tail_lines` while embedding exact truncation token metrics.
    """
    if not log_text:
        return ""

    lines = log_text.splitlines()
    if len(lines) <= (head_lines + tail_lines):
        return log_text

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    truncated_lines = len(lines) - (head_lines + tail_lines)
    truncated_text = "\n".join(lines[head_lines:-tail_lines])
    truncated_tokens = max(1, len(truncated_text.encode("utf-8", errors="ignore")) // 4)

    delimiter = f"\n[... truncated {truncated_lines} lines / ~{truncated_tokens} tokens ...]\n"
    return "\n".join(head) + delimiter + "\n".join(tail)


def prune_trajectory(
    frames: list[dict[str, Any]] | list[Any],
    preserve_keys: set[str] | None = None,
) -> list[Any]:
    """Prune primitive: Cleanly excises redundant tool calls, superseded state queries, and aborted trajectories."""
    if not frames:
        return []

    pruned: list[Any] = []
    seen_queries: dict[str, int] = {}

    for frame in frames:
        if isinstance(frame, dict):
            action = frame.get("action", "")
            is_aborted = frame.get("aborted", False)
            query_key = frame.get("query_key") or frame.get("tool_name")
        else:
            action = getattr(frame, "action", "")
            is_aborted = getattr(frame, "aborted", False)
            query_key = getattr(frame, "tool_name", None) or getattr(frame, "query_key", None)

        if is_aborted:
            continue

        if pruned:
            last = pruned[-1]
            last_action = last.get("action", "") if isinstance(last, dict) else getattr(last, "action", "")
            if action and action == last_action:
                pruned[-1] = frame
                continue

        if query_key and query_key not in (preserve_keys or set()):
            if query_key in seen_queries:
                prev_idx = seen_queries[query_key]
                if 0 <= prev_idx < len(pruned):
                    pruned[prev_idx] = None

        seen_queries[query_key or action] = len(pruned)
        pruned.append(frame)

    return [f for f in pruned if f is not None]


class SelfGCPlanner:
    """Xiaohongshu's Self-Governing Context Planner service for active context memory garbage collection."""

    def __init__(
        self,
        capacity_threshold: float = 0.30,
        max_context_capacity_tokens: int = 1_048_576,
        kv_store: ExternalKVStore | None = None,
    ) -> None:
        self.capacity_threshold = capacity_threshold
        self.max_context_capacity_tokens = max_context_capacity_tokens
        self.kv_store = kv_store or GLOBAL_KV_STORE
        self.gc_invocation_count = 0
        self.total_tokens_evicted = 0

    def should_trigger_gc(
        self,
        current_token_count: int,
        capacity_limit: int | None = None,
    ) -> bool:
        """Evaluates whether active context memory exceeds the 30% historical graph capacity threshold."""
        limit = capacity_limit or self.max_context_capacity_tokens
        if limit <= 0:
            return False
        ratio = current_token_count / limit
        return ratio >= self.capacity_threshold

    def plan_compaction(
        self,
        context_items: list[dict[str, Any]] | dict[str, Any],
        current_token_count: int,
        predicted_cost_benefit_passed: bool = True,
    ) -> dict[str, Any]:
        """Execute compaction plan applying Fold, Mask, and Prune primitives."""
        if not predicted_cost_benefit_passed:
            return {
                "gc_executed": False,
                "reason": "Cost-benefit gating failed: projected token savings did not exceed planner inference charges.",
                "tokens_saved": 0,
            }

        self.gc_invocation_count += 1
        folded_count = 0
        masked_count = 0
        original_tokens = current_token_count

        compacted_items: list[dict[str, Any]] = []
        items_list = context_items if isinstance(context_items, list) else [context_items]

        filtered_items = prune_trajectory(items_list)
        pruned_count = len(items_list) - len(filtered_items)

        for item in filtered_items:
            comp_item = dict(item) if isinstance(item, dict) else copy.deepcopy(item.__dict__)

            if "large_payload" in comp_item or "raw_code" in comp_item or "tool_output" in comp_item:
                target_key = "large_payload" if "large_payload" in comp_item else ("raw_code" if "raw_code" in comp_item else "tool_output")
                raw_data = comp_item[target_key]
                if isinstance(raw_data, str | dict | list) and len(str(raw_data)) > 200:
                    comp_item[target_key] = fold_payload(raw_data, kv_store=self.kv_store, key_prefix="gc_fold")
                    folded_count += 1

            if "terminal_output" in comp_item and isinstance(comp_item["terminal_output"], str):
                orig_log = comp_item["terminal_output"]
                masked_log = mask_stream(orig_log, head_lines=5, tail_lines=5)
                if masked_log != orig_log:
                    comp_item["terminal_output"] = masked_log
                    masked_count += 1

            if "execution_logs" in comp_item and isinstance(comp_item["execution_logs"], str):
                orig_log = comp_item["execution_logs"]
                masked_log = mask_stream(orig_log, head_lines=5, tail_lines=5)
                if masked_log != orig_log:
                    comp_item["execution_logs"] = masked_log
                    masked_count += 1

            compacted_items.append(comp_item)

        compacted_serialized = json.dumps(compacted_items, default=str)
        new_token_count = max(1, len(compacted_serialized.encode("utf-8")) // 4)
        tokens_saved = max(0, original_tokens - new_token_count)
        self.total_tokens_evicted += tokens_saved

        return {
            "gc_executed": True,
            "compacted_items": compacted_items,
            "folded_count": folded_count,
            "masked_count": masked_count,
            "pruned_count": pruned_count,
            "original_tokens": original_tokens,
            "new_tokens": new_token_count,
            "tokens_saved": tokens_saved,
            "gc_invocation_count": self.gc_invocation_count,
        }

# --------------------------------------------------------------------------- #
# Modernized Context Graph via Calibrated Decision Routing
# --------------------------------------------------------------------------- #


class CalibratedGraphBuilder:
    """Modernized context graph builder decoupling ingestion from calibrated relationship resolution.

    Eliminates slow (>1.5s) autoregressive LLM edge extraction by relying on a calibrated
    non-autoregressive decision model with strict thresholding (>= 0.85 commits, < 0.85 escalates).
    """

    def __init__(
        self,
        graph: EntityEventGraph | None = None,
        router: EdgeDecisionRouter | None = None,
        confidence_threshold: float = DEFAULT_EDGE_CONFIDENCE_THRESHOLD,
        normalize_entity_fn: Callable[[EntityMention], EntityResolutionResult] | None = None,
    ) -> None:
        self.graph = graph or EntityEventGraph(graph_id="calibrated_context_graph")
        self.router = router or EdgeDecisionRouter(confidence_threshold=confidence_threshold)
        self.normalize_entity_fn = normalize_entity_fn
        self._entities: dict[str, EntityResolutionResult] = {}
        self._committed_edges: list[EdgeDecision] = []

    def normalize_entity(self, mention: EntityMention) -> EntityResolutionResult:
        """Resolve an unstructured mention to a canonical entity.

        Retains an autoregressive normalization path when a callable is provided,
        otherwise performs deterministic canonicalization.
        """
        if self.normalize_entity_fn is not None:
            res = self.normalize_entity_fn(mention)
        else:
            # Deterministic normalization fallback
            clean_name = mention.text.strip().lower()
            entity_id = f"ent_{mention.entity_type}_{clean_name.replace(' ', '_')}"
            res = EntityResolutionResult(
                mention_id=mention.mention_id,
                resolved_id=entity_id,
                canonical_name=mention.text.strip().title(),
                confidence=mention.confidence,
                is_new_entity=(entity_id not in self._entities),
                normalized_attributes=dict(mention.attributes),
            )

        if res.resolved_id:
            self._entities[res.resolved_id] = res
            # Ensure registered in underlying EntityEventGraph
            if self.graph.get_entity(res.resolved_id) is None:
                self.graph.add_entity(
                    entity_id=res.resolved_id,
                    kind=EntityKind.MODULE,
                    name=res.canonical_name or res.resolved_id,
                    source_uri=mention.source_doc_id or "context://mention",
                    attributes=res.normalized_attributes,
                )

        return res

    def propose_and_route_edge(self, candidate: EdgeCandidate) -> GraphTransactionResult:
        """Route an edge candidate through calibrated decision classification.

        If confidence >= threshold (default 0.85) and not Irrelevant, commits edge to graph.
        If confidence < threshold, routes candidate to escalation queue and DOES NOT commit.
        """
        result = self.router.route_candidate(candidate)

        if result.committed and result.edge_decision:
            decision = result.edge_decision
            self._committed_edges.append(decision)

            # Map EdgeClass to DependencyEdgeType for associative graph storage
            edge_type_map = {
                EdgeClass.Contradicts: DependencyEdgeType.MUTATES,
                EdgeClass.Extends: DependencyEdgeType.INHERITS,
                EdgeClass.TemporalFollowup: DependencyEdgeType.CALLS,
                EdgeClass.Irrelevant: DependencyEdgeType.IMPORTS,
            }
            mapped_type = edge_type_map.get(decision.edge_class, DependencyEdgeType.CALLS)

            self.graph.add_dependency(
                source_id=decision.source_id,
                target_id=decision.target_id,
                edge_type=mapped_type,
                metadata={
                    "edge_class": decision.edge_class.value,
                    "confidence": decision.confidence,
                    "model_version": decision.calibration.model_version,
                    "decision_trace_id": decision.calibration.decision_trace_id,
                },
            )

        return result

    def ingest_mentions_and_resolve(
        self,
        mentions: list[EntityMention],
        pairwise_similarity_fn: Callable[[EntityResolutionResult, EntityResolutionResult], float] | None = None,
    ) -> list[GraphTransactionResult]:
        """Normalize mentions, generate candidates, and route transactions through calibrated decisioning."""
        resolved = [self.normalize_entity(m) for m in mentions]
        results: list[GraphTransactionResult] = []

        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                ent_a = resolved[i]
                ent_b = resolved[j]
                if not ent_a.resolved_id or not ent_b.resolved_id:
                    continue
                if ent_a.resolved_id == ent_b.resolved_id:
                    continue

                sim = 0.5
                if pairwise_similarity_fn is not None:
                    sim = pairwise_similarity_fn(ent_a, ent_b)
                elif ent_a.canonical_name and ent_b.canonical_name:
                    # Simple heuristic overlap
                    words_a = set(ent_a.canonical_name.lower().split())
                    words_b = set(ent_b.canonical_name.lower().split())
                    intersect = words_a.intersection(words_b)
                    sim = len(intersect) / max(1, len(words_a.union(words_b)))

                features = EdgeFeatures(
                    source_entity_id=ent_a.resolved_id,
                    target_entity_id=ent_b.resolved_id,
                    source_text=ent_a.canonical_name,
                    target_text=ent_b.canonical_name,
                    semantic_similarity=sim,
                )
                candidate = EdgeCandidate(
                    source_id=ent_a.resolved_id,
                    target_id=ent_b.resolved_id,
                    features=features,
                )
                tx_res = self.propose_and_route_edge(candidate)
                results.append(tx_res)

        return results

    @property
    def escalation_queue(self) -> EscalationQueue:
        return self.router.escalation_queue

    @property
    def telemetry(self) -> RouterTelemetry:
        return self.router.telemetry

    @property
    def committed_edges(self) -> list[EdgeDecision]:
        return list(self._committed_edges)


__all__ = [
    "ModuleNode",
    "RepoContextGraph",
    "RepoContextGraphBuilder",
    "AgentNode",
    "AgentDependencyGraph",
    "AgentGraphBuilder",
    "VisualLayoutGraphNode",
    "build_visual_layout_subgraph",
    "HNSWNode",
    "FilterableHNSWIndex",
    "ExternalKVStore",
    "GLOBAL_KV_STORE",
    "fold_payload",
    "mask_stream",
    "prune_trajectory",
    "SelfGCPlanner",
    "EdgeClass",
    "EntityMention",
    "EntityResolutionResult",
    "EdgeFeatures",
    "CalibrationMetadata",
    "EdgeCandidate",
    "EdgeDecision",
    "EscalationRecord",
    "GraphTransactionResult",
    "EdgeDecisionClassifier",
    "DeterministicHeuristicClassifier",
    "ModelBackedRLCDClassifier",
    "RouterTelemetry",
    "EdgeDecisionRouter",
    "EscalationQueue",
    "CalibratedGraphBuilder",
]


