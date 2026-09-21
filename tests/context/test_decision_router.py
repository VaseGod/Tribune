"""Unit tests for EdgeDecisionRouter, calibrated classifiers, thresholds, and escalation."""

from __future__ import annotations

import unittest

from tribune.context.decision_router import (
    DeterministicHeuristicClassifier,
    EdgeDecisionRouter,
    ModelBackedRLCDClassifier,
)
from tribune.context.entities import (
    EdgeCandidate,
    EdgeClass,
    EdgeFeatures,
)
from tribune.context.escalation import EscalationQueue


class TestDecisionRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.escalation_queue = EscalationQueue(queue_name="test_queue")
        self.heuristic_classifier = DeterministicHeuristicClassifier()
        self.model_classifier = ModelBackedRLCDClassifier()
        self.router = EdgeDecisionRouter(
            classifier=self.heuristic_classifier,
            escalation_queue=self.escalation_queue,
            confidence_threshold=0.85,
        )

    def test_edge_taxonomy_enum_values(self) -> None:
        """Verify strict categorical edge taxonomy."""
        self.assertEqual(EdgeClass.Contradicts.value, "Contradicts")
        self.assertEqual(EdgeClass.Extends.value, "Extends")
        self.assertEqual(EdgeClass.TemporalFollowup.value, "TemporalFollowup")
        self.assertEqual(EdgeClass.Irrelevant.value, "Irrelevant")

    def test_high_confidence_contradicts_committed(self) -> None:
        """Contradiction with clear negation achieves >= 0.85 and commits."""
        features = EdgeFeatures(
            source_entity_id="hearing_claim_1",
            target_entity_id="statutory_ruling_1",
            source_text="Eligible for full benefits",
            target_text="Terminated and disqualified due to excess income",
            negation_detected=True,
            semantic_similarity=0.88,
        )
        candidate = EdgeCandidate(
            source_id="hearing_claim_1",
            target_id="statutory_ruling_1",
            features=features,
        )
        res = self.router.route_candidate(candidate)

        self.assertTrue(res.committed)
        self.assertFalse(res.escalated)
        self.assertIsNotNone(res.edge_decision)
        self.assertEqual(res.edge_decision.edge_class, EdgeClass.Contradicts)
        self.assertGreaterEqual(res.edge_decision.confidence, 0.85)
        self.assertEqual(self.escalation_queue.stats()["pending_count"], 0)

    def test_high_confidence_extends_committed(self) -> None:
        """Extension with high semantic similarity and subsumption commits."""
        features = EdgeFeatures(
            source_entity_id="wage_verification_1",
            target_entity_id="snap_application_1",
            source_text="Supplements verified monthly gross wage $1,450",
            target_text="SNAP application wage reporting",
            subsumption_detected=True,
            semantic_similarity=0.85,
        )
        candidate = EdgeCandidate(
            source_id="wage_verification_1",
            target_id="snap_application_1",
            features=features,
        )
        res = self.router.route_candidate(candidate)

        self.assertTrue(res.committed)
        self.assertFalse(res.escalated)
        self.assertEqual(res.edge_decision.edge_class, EdgeClass.Extends)
        self.assertGreaterEqual(res.edge_decision.confidence, 0.85)

    def test_high_confidence_temporal_followup_committed(self) -> None:
        """Temporal followup with time delta >= 0.85 commits."""
        features = EdgeFeatures(
            source_entity_id="intake_interview_2026_01",
            target_entity_id="redetermination_2026_07",
            source_text="Initial certification period established",
            target_text="Subsequent recertification renewal following six months",
            temporal_distance_s=15552000.0,
            semantic_similarity=0.80,
        )
        candidate = EdgeCandidate(
            source_id="intake_interview_2026_01",
            target_id="redetermination_2026_07",
            features=features,
        )
        res = self.router.route_candidate(candidate)

        self.assertTrue(res.committed)
        self.assertEqual(res.edge_decision.edge_class, EdgeClass.TemporalFollowup)
        self.assertGreaterEqual(res.edge_decision.confidence, 0.85)

    def test_low_confidence_boundary_routes_to_escalation(self) -> None:
        """Confidence < 0.85 MUST route to escalation queue and NOT commit."""
        # Create ambiguous features with moderate similarity and no decisive markers
        features = EdgeFeatures(
            source_entity_id="ambiguous_note_a",
            target_entity_id="ambiguous_note_b",
            source_text="Claimant mentioned an unexpected expense",
            target_text="General household background detail",
            semantic_similarity=0.45,
            negation_detected=False,
            subsumption_detected=False,
            temporal_distance_s=None,
        )
        candidate = EdgeCandidate(
            source_id="ambiguous_note_a",
            target_id="ambiguous_note_b",
            features=features,
        )

        res = self.router.route_candidate(candidate)

        self.assertFalse(res.committed, "Low-confidence candidate must not be committed!")
        self.assertTrue(res.escalated, "Low-confidence candidate must be escalated!")
        self.assertIsNotNone(res.escalation_record)
        self.assertLess(res.edge_decision.confidence, 0.85)

        # Check queue
        stats = self.escalation_queue.stats()
        self.assertEqual(stats["pending_count"], 1)
        esc = self.escalation_queue.peek()
        self.assertIsNotNone(esc)
        self.assertEqual(esc.candidate.source_id, "ambiguous_note_a")

        # Resolve escalation record
        resolved = self.escalation_queue.resolve(esc.record_id, notes="Verified manually by caseworker")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(self.escalation_queue.stats()["pending_count"], 0)

    def test_irrelevant_edge_not_committed_not_escalated(self) -> None:
        """Irrelevant edge with confidence >= 0.85 is discarded cleanly."""
        features = EdgeFeatures(
            source_entity_id="weather_report",
            target_entity_id="medicaid_income_test",
            source_text="It rained yesterday morning in the region",
            target_text="Monthly countable income threshold $1,450",
            semantic_similarity=0.05,
        )
        candidate = EdgeCandidate(
            source_id="weather_report",
            target_id="medicaid_income_test",
            features=features,
        )
        res = self.router.route_candidate(candidate)

        self.assertFalse(res.committed)
        self.assertFalse(res.escalated)
        self.assertEqual(res.edge_decision.edge_class, EdgeClass.Irrelevant)
        self.assertEqual(self.escalation_queue.stats()["pending_count"], 0)

    def test_model_backed_rlcd_classifier(self) -> None:
        """Test ModelBackedRLCDClassifier with batched inference and calibration metadata."""
        features_list = [
            EdgeFeatures(
                source_entity_id="e1",
                target_entity_id="e2",
                source_text="Approval",
                target_text="Denied and disqualified",
                negation_detected=True,
                semantic_similarity=0.85,
            ),
            EdgeFeatures(
                source_entity_id="e3",
                target_entity_id="e4",
                source_text="Initial application",
                target_text="Subsequent follow-up",
                temporal_distance_s=100.0,
                semantic_similarity=0.75,
            ),
        ]
        decisions = self.model_classifier.batch_classify(features_list)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0].edge_class, EdgeClass.Contradicts)
        self.assertEqual(decisions[1].edge_class, EdgeClass.TemporalFollowup)
        for d in decisions:
            self.assertEqual(d.calibration.calibration_method, "rlcd_temperature_scaling")
            self.assertTrue(d.calibration.decision_trace_id.startswith("trace_"))

    def test_telemetry_accumulation(self) -> None:
        """Ensure RouterTelemetry accurately tracks latency, confidence, and counts."""
        tel = self.router.telemetry
        initial_queries = tel.total_queries

        features = EdgeFeatures(
            source_entity_id="s1",
            target_entity_id="t1",
            source_text="Income verification documents",
            target_text="Supplements submitted proof of wages",
            semantic_similarity=0.82,
            subsumption_detected=True,
        )
        self.router.route_candidate(EdgeCandidate("s1", "t1", features=features))

        self.assertEqual(tel.total_queries, initial_queries + 1)
        self.assertGreater(tel.decision_confidence, 0.0)
        self.assertGreaterEqual(tel.edge_classification_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
