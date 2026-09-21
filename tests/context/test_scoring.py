"""Unit tests for CalibratedKernelScorer, scoring weights, decay curves, and explainability."""

from __future__ import annotations

import unittest

from tribune.context.scoring import (
    CalibratedKernelScorer,
    ContextNode,
    ScoringWeights,
)


class TestCalibratedScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = ScoringWeights(
            w_recency=0.25,
            w_frequency=0.15,
            w_centrality=0.15,
            w_task=0.20,
            w_semantic=0.10,
            w_escalation=0.10,
            w_downstream=0.05,
        )
        self.scorer = CalibratedKernelScorer(
            weights=self.weights,
            decay_half_life_steps=10.0,
            decay_curve="exponential",
        )

    def test_deterministic_scoring(self) -> None:
        """Verify identical inputs and steps produce identical utility scores."""
        node = ContextNode(
            node_id="node_1",
            text="SNAP statutory income deduction standard",
            token_count=120,
            created_step=1,
            last_accessed_step=5,
            access_count=3,
            task_relevance=0.8,
            semantic_relevance=0.7,
            graph_centrality=0.4,
            in_escalation=False,
            downstream_references=2,
        )
        score_1 = self.scorer.score(node, current_step=10)
        score_2 = self.scorer.score(node, current_step=10)
        self.assertEqual(score_1, score_2)
        self.assertGreater(score_1, 0.0)
        self.assertLessEqual(score_1, 1.0)

    def test_recency_decay_curves(self) -> None:
        """Verify recency score degrades as step delta increases across decay curves."""
        node = ContextNode(
            node_id="decay_node",
            text="Historical turn",
            token_count=50,
            created_step=1,
            last_accessed_step=1,
        )

        scorer_exp = CalibratedKernelScorer(decay_curve="exponential", decay_half_life_steps=10.0)
        scorer_lin = CalibratedKernelScorer(decay_curve="linear", decay_half_life_steps=10.0)
        scorer_sig = CalibratedKernelScorer(decay_curve="sigmoid", decay_half_life_steps=10.0)

        # Immediate step (delta=0) vs later steps (delta=10, 20)
        score_exp_0 = scorer_exp.score(node, current_step=1)
        score_exp_10 = scorer_exp.score(node, current_step=11)
        score_exp_20 = scorer_exp.score(node, current_step=21)

        self.assertGreater(score_exp_0, score_exp_10)
        self.assertGreater(score_exp_10, score_exp_20)

        score_lin_0 = scorer_lin.score(node, current_step=1)
        score_lin_20 = scorer_lin.score(node, current_step=21)
        self.assertGreater(score_lin_0, score_lin_20)

        score_sig_0 = scorer_sig.score(node, current_step=1)
        score_sig_20 = scorer_sig.score(node, current_step=21)
        self.assertGreater(score_sig_0, score_sig_20)

    def test_high_utility_vs_irrelevant_node(self) -> None:
        """High task relevance and recency nodes must score strictly higher than stale irrelevant nodes."""
        high_node = ContextNode(
            node_id="critical_income_proof",
            text="Pay stub verifying $1,450 monthly wage",
            token_count=80,
            created_step=8,
            last_accessed_step=9,
            access_count=5,
            task_relevance=0.95,
            semantic_relevance=0.90,
            graph_centrality=0.6,
            downstream_references=4,
        )
        stale_node = ContextNode(
            node_id="stale_weather_aside",
            text="Applicant commented that traffic was bad",
            token_count=70,
            created_step=1,
            last_accessed_step=1,
            access_count=1,
            task_relevance=0.05,
            semantic_relevance=0.05,
            graph_centrality=0.0,
            downstream_references=0,
        )
        current_step = 10
        high_score = self.scorer.score(high_node, current_step)
        stale_score = self.scorer.score(stale_node, current_step)

        self.assertGreater(high_score, stale_score + 0.35)

    def test_escalation_boost(self) -> None:
        """Nodes marked in_escalation receive an explicit utility boost."""
        normal_node = ContextNode(
            node_id="normal",
            text="Deduction check",
            token_count=50,
            created_step=5,
            last_accessed_step=5,
            task_relevance=0.5,
            in_escalation=False,
        )
        escalated_node = ContextNode(
            node_id="escalated",
            text="Deduction check",
            token_count=50,
            created_step=5,
            last_accessed_step=5,
            task_relevance=0.5,
            in_escalation=True,
        )
        current_step = 6
        self.assertGreater(
            self.scorer.score(escalated_node, current_step),
            self.scorer.score(normal_node, current_step),
        )

    def test_explainable_scoring(self) -> None:
        """Verify explainable breakdown contains all 7 dimensions and informative rationale."""
        node = ContextNode(
            node_id="sample_node",
            text="Medicaid household residency declaration",
            token_count=90,
            created_step=2,
            last_accessed_step=8,
            task_relevance=0.85,
        )
        explanation = self.scorer.explain(node, current_step=10)
        self.assertEqual(explanation.node_id, "sample_node")
        self.assertIn("recency", explanation.component_scores)
        self.assertIn("frequency", explanation.component_scores)
        self.assertIn("task_relevance", explanation.component_scores)
        self.assertIn("centrality", explanation.component_scores)
        self.assertIn("escalation_boost", explanation.component_scores)
        self.assertIn("downstream_references", explanation.component_scores)
        self.assertIn("sample_node", explanation.rationale)


if __name__ == "__main__":
    unittest.main()
