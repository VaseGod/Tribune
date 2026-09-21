"""Unit and integration tests for ProactiveContextManager calibrated memory eviction and long session simulation."""

from __future__ import annotations

import unittest

from tribune.context.manager import ProactiveContextManager
from tribune.context.scoring import ContextNode


class TestCalibratedContextManager(unittest.TestCase):
    def setUp(self) -> None:
        self.summarized_nodes: list[str] = []

        def mock_summarizer(nodes: list[ContextNode]) -> dict[str, str]:
            for n in nodes:
                self.summarized_nodes.append(n.node_id)
            return {n.node_id: f"Summary of {n.node_id}" for n in nodes}

        self.manager = ProactiveContextManager(
            total_budget=1000,
            window_size=400,
            eviction_policy="calibrated_kernel",
            summarize_hook=mock_summarizer,
        )

    def test_eviction_orders_by_lowest_utility(self) -> None:
        """Verify irrelevant stale nodes are evicted before high-utility relevant nodes."""
        # 1. High utility node
        high_node = ContextNode(
            node_id="high_utility_rule",
            text="Federal poverty level table for 2026",
            token_count=150,
            created_step=1,
            last_accessed_step=5,
            access_count=4,
            task_relevance=0.95,
            semantic_relevance=0.90,
            graph_centrality=0.8,
            downstream_references=5,
        )
        # 2. Medium utility node
        med_node = ContextNode(
            node_id="med_utility_note",
            text="Caseworker appointment scheduled for next week",
            token_count=150,
            created_step=2,
            last_accessed_step=4,
            access_count=2,
            task_relevance=0.6,
            semantic_relevance=0.5,
        )
        # 3. Low utility irrelevant node
        low_node = ContextNode(
            node_id="low_utility_aside",
            text="Discussion about office parking spaces",
            token_count=200,
            created_step=1,
            last_accessed_step=1,
            access_count=1,
            task_relevance=0.05,
            semantic_relevance=0.05,
        )

        self.manager.register_node(high_node)
        self.manager.register_node(med_node)
        self.manager.register_node(low_node)
        self.manager.advance_step()

        # Total active tokens = 150 + 150 + 200 = 500 tokens > 400 window size
        evictions = self.manager.evict_to_budget(target_budget=400)

        self.assertGreater(len(evictions), 0)
        # The lowest utility node must be the first evicted
        self.assertEqual(evictions[0].node_id, "low_utility_aside")
        self.assertTrue(low_node.is_evicted)
        self.assertFalse(high_node.is_evicted)
        # Summarization hook was called non-blockingly
        self.assertIn("low_utility_aside", self.summarized_nodes)

    def test_long_session_token_stabilization(self) -> None:
        """Long-session simulation test demonstrates token growth remains strictly bounded."""
        window_size = 500
        mgr = ProactiveContextManager(
            total_budget=2000,
            window_size=window_size,
            eviction_policy="calibrated_kernel",
        )

        # Simulate 50 turns generating ~100 tokens per turn (5,000 tokens cumulative)
        for step in range(1, 51):
            mgr.advance_step()
            node = ContextNode(
                node_id=f"step_{step}_data",
                text=f"Dialogue turn {step}: evidence item and statutory notes",
                token_count=100,
                created_step=step,
                last_accessed_step=step,
                task_relevance=0.5 if step % 2 == 0 else 0.8,
            )
            mgr.register_node(node)
            mgr.evict_to_budget(target_budget=window_size)

        telemetry = mgr.get_memory_telemetry()
        # Active tokens must not exceed window size
        self.assertLessEqual(telemetry["token_estimate"], window_size)
        self.assertGreater(telemetry["eviction_count"], 35)
        self.assertEqual(telemetry["eviction_policy"], "calibrated_kernel")

    def test_execution_trace_compression(self) -> None:
        """Compress execution trace text preserving citations and predicates."""
        nodes = [
            ContextNode(
                node_id="n1",
                text="Applicant is eligible because gross income satisfies 7 CFR 273.9(a).",
                token_count=15,
                created_step=1,
                last_accessed_step=1,
            ),
            ContextNode(
                node_id="n2",
                text="General casual background remark that does not contain rules.",
                token_count=12,
                created_step=2,
                last_accessed_step=2,
            ),
        ]
        compressed = self.manager.compress_execution_trace(nodes, ratio=0.6)
        self.assertIn("7 CFR 273.9(a)", compressed)
        self.assertIn("because", compressed)


if __name__ == "__main__":
    unittest.main()
