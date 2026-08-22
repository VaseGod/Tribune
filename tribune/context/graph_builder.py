"""Module & Dependency Context Builder.

Scans project files, parses Python AST to construct cross-file import/export mappings
and dependency graphs, and formats context graphs for prompt injection.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path


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
                    broadcast_subscriptions=[f"/evidence", f"/criteria_outcomes/{prog_val}"],
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


__all__ = [
    "ModuleNode",
    "RepoContextGraph",
    "RepoContextGraphBuilder",
    "AgentNode",
    "AgentDependencyGraph",
    "AgentGraphBuilder",
]

