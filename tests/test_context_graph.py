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

    def test_self_gc_planner_capacity_trigger(self) -> None:
        from tribune.context.graph_builder import SelfGCPlanner

        planner = SelfGCPlanner(capacity_threshold=0.30, max_context_capacity_tokens=100_000)

        # 1. Below 30% capacity (20,000 / 100,000 = 20%) -> No trigger
        self.assertFalse(planner.should_trigger_gc(current_token_count=20_000))

        # 2. At or above 30% capacity (35,000 / 100,000 = 35%) -> Trigger
        self.assertTrue(planner.should_trigger_gc(current_token_count=35_000))
        self.assertTrue(planner.should_trigger_gc(current_token_count=30_000))

    def test_fold_primitive_evicts_payload_to_external_kv(self) -> None:
        from tribune.context.graph_builder import ExternalKVStore, fold_payload

        kv_store = ExternalKVStore(namespace="test_kv")
        large_json = {"huge_data": "x" * 5000, "status": "active"}

        pointer_node = fold_payload(large_json, kv_store=kv_store, key_prefix="tool_output")
        self.assertTrue(pointer_node["folded"])
        self.assertIn("$ref", pointer_node)
        self.assertTrue(pointer_node["uri_pointer"].startswith("ref://test_kv/tool_output_"))
        self.assertGreater(pointer_node["size_bytes"], 5000)

        # Re-fetch from KV store
        stored = kv_store.get(pointer_node["uri_pointer"])
        self.assertEqual(stored, large_json)

    def test_mask_primitive_truncates_intermediate_logs(self) -> None:
        from tribune.context.graph_builder import mask_stream

        # Stream with 30 lines
        lines = [f"Log line {i:02d}: system status check" for i in range(30)]
        full_log = "\n".join(lines)

        masked = mask_stream(full_log, head_lines=5, tail_lines=5)
        masked_lines = masked.splitlines()

        # First 5 lines preserved
        self.assertEqual(masked_lines[0], lines[0])
        self.assertEqual(masked_lines[4], lines[4])

        # Trailer 5 lines preserved
        self.assertEqual(masked_lines[-1], lines[-1])
        self.assertEqual(masked_lines[-5], lines[-5])

        # Delimiter contains truncation count
        self.assertIn("truncated 20 lines", masked)

    def test_prune_primitive_excises_redundant_and_aborted_trajectories(self) -> None:
        from tribune.context.graph_builder import prune_trajectory

        frames = [
            {"frame_id": "f1", "action": "query_statute", "query_key": "snap_income", "state": "plan"},
            {"frame_id": "f2", "action": "query_statute", "query_key": "snap_income", "state": "plan"},  # Duplicate consecutive
            {"frame_id": "f3", "action": "aborted_branch", "state": "explore", "aborted": True},  # Aborted
            {"frame_id": "f4", "action": "query_statute", "query_key": "snap_income", "state": "assess"},  # Superseding query
        ]

        pruned = prune_trajectory(frames)
        # Aborted f3 removed, duplicate f1 removed/superseded by f4
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["frame_id"], "f4")

    def test_self_gc_planner_full_compaction_pass(self) -> None:
        from tribune.context.graph_builder import ExternalKVStore, SelfGCPlanner

        kv_store = ExternalKVStore()
        planner = SelfGCPlanner(kv_store=kv_store)

        items = [
            {
                "tool_name": "fetch_large_code",
                "raw_code": "def func():\n" + "    pass\n" * 200,
                "terminal_output": "\n".join([f"Step {i}" for i in range(40)]),
            }
        ]

        report = planner.plan_compaction(items, current_token_count=5000)
        self.assertTrue(report["gc_executed"])
        self.assertEqual(report["folded_count"], 1)
        self.assertEqual(report["masked_count"], 1)
        self.assertGreater(report["tokens_saved"], 0)


if __name__ == "__main__":
    unittest.main()

