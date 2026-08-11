"""Unit tests for AST-based repository context graph builder."""

from __future__ import annotations

import os
import tempfile
import unittest

from tribune.context.graph_builder import RepoContextGraphBuilder


class TestContextGraph(unittest.TestCase):
    def test_ast_scanning_and_dependency_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a sample module hierarchy
            mod_a = os.path.join(tmp_dir, "module_a.py")
            mod_b = os.path.join(tmp_dir, "module_b.py")

            with open(mod_a, "w", encoding="utf-8") as f:
                f.write(
                    "__all__ = ['HelperClass']\n\n"
                    "class HelperClass:\n"
                    "    pass\n\n"
                    "def helper_func():\n"
                    "    return 42\n"
                )

            with open(mod_b, "w", encoding="utf-8") as f:
                f.write(
                    "from module_a import HelperClass\n\n"
                    "def run():\n"
                    "    h = HelperClass()\n"
                )

            builder = RepoContextGraphBuilder(tmp_dir)
            graph = builder.build_graph()

            self.assertIn("module_a", graph.modules)
            self.assertIn("module_b", graph.modules)

            node_a = graph.modules["module_a"]
            self.assertIn("HelperClass", node_a.exports)
            self.assertIn("HelperClass", node_a.classes)
            self.assertIn("helper_func", node_a.functions)

            node_b = graph.modules["module_b"]
            self.assertIn("module_a.HelperClass", node_b.imports)

            self.assertIn("module_a", graph.dependency_graph.get("module_b", []))

            context_str = graph.to_context_string()
            self.assertIn("REPOSITORY CONTEXT GRAPH", context_str)
            self.assertIn("module_b -> [module_a]", context_str)
            self.assertIn("Exports (__all__): HelperClass", context_str)


if __name__ == "__main__":
    unittest.main()
