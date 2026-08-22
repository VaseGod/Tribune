"""Unit tests for Multi-Agent Dependency Graph Builder and Scope Modeling."""

from __future__ import annotations

import unittest

from tribune.context.graph_builder import (
    AgentDependencyGraph,
    AgentGraphBuilder,
    AgentNode,
)


class TestAgentDependencyGraph(unittest.TestCase):
    def test_agent_dependency_topological_sort(self) -> None:
        graph = AgentDependencyGraph()
        graph.add_node(AgentNode(agent_id="gather", role="gather", dependencies=[]))
        graph.add_node(AgentNode(agent_id="navigator", role="navigator", dependencies=["gather"]))
        graph.add_node(AgentNode(agent_id="proposer_snap", role="proposer", dependencies=["gather", "navigator"]))
        graph.add_node(AgentNode(agent_id="verifier_snap", role="verifier", dependencies=["proposer_snap"]))
        graph.add_node(AgentNode(agent_id="preparer_snap", role="preparer", dependencies=["verifier_snap"]))

        order = [node.agent_id for node in graph.topological_order()]
        self.assertEqual(order, ["gather", "navigator", "proposer_snap", "verifier_snap", "preparer_snap"])

    def test_topological_parallel_waves(self) -> None:
        graph = AgentDependencyGraph()
        graph.add_node(AgentNode(agent_id="gather", role="gather", dependencies=[]))
        graph.add_node(AgentNode(agent_id="navigator", role="navigator", dependencies=["gather"]))
        # Parallel proposers
        graph.add_node(AgentNode(agent_id="proposer_snap", role="proposer", dependencies=["navigator"]))
        graph.add_node(AgentNode(agent_id="proposer_medicaid", role="proposer", dependencies=["navigator"]))
        # Parallel verifiers
        graph.add_node(AgentNode(agent_id="verifier_snap", role="verifier", dependencies=["proposer_snap"]))
        graph.add_node(AgentNode(agent_id="verifier_medicaid", role="verifier", dependencies=["proposer_medicaid"]))

        waves = graph.topological_waves()
        wave_ids = [[n.agent_id for n in w] for w in waves]

        self.assertEqual(wave_ids[0], ["gather"])
        self.assertEqual(wave_ids[1], ["navigator"])
        self.assertCountEqual(wave_ids[2], ["proposer_snap", "proposer_medicaid"])
        self.assertCountEqual(wave_ids[3], ["verifier_snap", "verifier_medicaid"])

    def test_cycle_detection(self) -> None:
        graph = AgentDependencyGraph()
        graph.add_node(AgentNode(agent_id="a", role="proposer", dependencies=["b"]))
        graph.add_node(AgentNode(agent_id="b", role="verifier", dependencies=["a"]))
        with self.assertRaises(ValueError):
            graph.topological_order()

    def test_agent_graph_builder_for_programs(self) -> None:
        graph = AgentGraphBuilder.build_for_programs(["snap", "medicaid"], jurisdiction="EX")
        self.assertIn("gather", graph.nodes)
        self.assertIn("navigator", graph.nodes)
        self.assertIn("proposer_snap", graph.nodes)
        self.assertIn("verifier_snap", graph.nodes)
        self.assertIn("preparer_snap", graph.nodes)
        self.assertIn("proposer_medicaid", graph.nodes)
        self.assertIn("verifier_medicaid", graph.nodes)
        self.assertIn("preparer_medicaid", graph.nodes)

        # Ensure scopes are properly set
        snap_prop = graph.get_node("proposer_snap")
        self.assertIsNotNone(snap_prop)
        self.assertIn("/assessments/snap", snap_prop.write_scopes)

        mermaid = graph.to_mermaid()
        self.assertIn("graph TD", mermaid)
        self.assertIn("proposer_snap", mermaid)


if __name__ == "__main__":
    unittest.main()
