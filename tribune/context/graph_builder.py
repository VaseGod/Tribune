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
