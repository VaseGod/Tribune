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
import heapq
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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

        # 1. Gather Agent (Ingests & consolidates evidence)
        graph.add_node(
            AgentNode(
                agent_id="gather",
                role="gather",
                read_scopes=["/documents"],
                write_scopes=["/evidence", "/shared_facts"],
                dependencies=[],
                broadcast_subscriptions=["/documents"],
            )
        )

        # 2. Navigator Agent (High-level planning & coordination)
        graph.add_node(
            AgentNode(
                agent_id="navigator",
                role="navigator",
                read_scopes=["/evidence", "/assessments", "/materials"],
                write_scopes=["/metadata/plan", "/agent_metadata/navigator"],
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
                    read_scopes=["/evidence", f"/shared_facts/{prog_val}", "/shared_facts"],
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
                    read_scopes=["/evidence", f"/assessments/{prog_val}", f"/criteria_outcomes/{prog_val}"],
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
                    read_scopes=["/evidence", f"/assessments/{prog_val}", f"/verification_verdicts/{prog_val}"],
                    write_scopes=[f"/materials/{prog_val}"],
                    dependencies=[ver_id],
                    broadcast_subscriptions=[f"/verification_verdicts/{prog_val}"],
                )
            )

        return graph


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


__all__ = [
    "ModuleNode",
    "RepoContextGraph",
    "RepoContextGraphBuilder",
    "AgentNode",
    "AgentDependencyGraph",
    "AgentGraphBuilder",
    "HNSWNode",
    "FilterableHNSWIndex",
]
