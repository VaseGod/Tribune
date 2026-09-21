"""Unit tests for CalibratedGraphBuilder, entity normalization, and edge resolution."""

from __future__ import annotations

import unittest

from tribune.context.entities import (
    EdgeCandidate,
    EdgeClass,
    EdgeFeatures,
    EntityMention,
)
from tribune.context.graph import EntityEventGraph
from tribune.context.graph_builder import (
    CalibratedGraphBuilder,
    DeterministicHeuristicClassifier,
    EdgeDecisionRouter,
    EscalationQueue,
)


class TestCalibratedGraphBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = EntityEventGraph(graph_id="test_calibrated_graph")
        self.queue = EscalationQueue(queue_name="test_escalation")
        self.router = EdgeDecisionRouter(
            classifier=DeterministicHeuristicClassifier(),
            escalation_queue=self.queue,
            confidence_threshold=0.85,
        )
        self.builder = CalibratedGraphBuilder(
            graph=self.graph,
            router=self.router,
            confidence_threshold=0.85,
        )

    def test_entity_normalization(self) -> None:
        mention = EntityMention(
            text="SNAP Gross Income Limit",
            entity_type="statute",
            source_doc_id="doc_7cfr273",
            attributes={"limit": 1450},
        )
        resolved = self.builder.normalize_entity(mention)

        self.assertIsNotNone(resolved.resolved_id)
        self.assertEqual(resolved.canonical_name, "Snap Gross Income Limit")
        self.assertTrue(resolved.is_new_entity)
        # Should be registered in underlying graph
        node = self.graph.get_entity(resolved.resolved_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "Snap Gross Income Limit")

    def test_propose_and_route_edge_commit(self) -> None:
        # High confidence edge
        features = EdgeFeatures(
            source_entity_id="ent_doc_snap_stub",
            target_entity_id="ent_doc_snap_app",
            source_text="Pay stub verifying $1,450",
            target_text="Supplements SNAP application reporting",
            subsumption_detected=True,
            semantic_similarity=0.85,
        )
        cand = EdgeCandidate("ent_doc_snap_stub", "ent_doc_snap_app", features=features)
        tx = self.builder.propose_and_route_edge(cand)

        self.assertTrue(tx.committed)
        self.assertFalse(tx.escalated)
        self.assertEqual(len(self.builder.committed_edges), 1)
        self.assertEqual(self.builder.committed_edges[0].edge_class, EdgeClass.Extends)

    def test_propose_and_route_edge_escalation(self) -> None:
        # Low confidence ambiguous edge
        features = EdgeFeatures(
            source_entity_id="ent_misc_note_1",
            target_entity_id="ent_misc_note_2",
            source_text="General remark regarding household utilities",
            target_text="Generic inquiry from applicant",
            semantic_similarity=0.40,
        )
        cand = EdgeCandidate("ent_misc_note_1", "ent_misc_note_2", features=features)
        tx = self.builder.propose_and_route_edge(cand)

        self.assertFalse(tx.committed)
        self.assertTrue(tx.escalated)
        self.assertEqual(len(self.builder.committed_edges), 0)
        self.assertEqual(self.queue.stats()["pending_count"], 1)

    def test_ingest_mentions_and_resolve(self) -> None:
        mentions = [
            EntityMention(
                text="Income Verification Notice",
                entity_type="notice",
                source_doc_id="doc_1",
            ),
            EntityMention(
                text="Income Verification Supplements",
                entity_type="notice",
                source_doc_id="doc_2",
            ),
        ]
        tx_results = self.builder.ingest_mentions_and_resolve(mentions)
        self.assertEqual(len(tx_results), 1)
        self.assertIsNotNone(tx_results[0].edge_decision)


if __name__ == "__main__":
    unittest.main()
