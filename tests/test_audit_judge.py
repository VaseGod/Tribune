"""Unit tests for Continuous Audit Judge Evaluators and Live Trace Monitoring."""

from __future__ import annotations

import tempfile
import unittest

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.corpus.rule_store import LocalRuleStore
from tribune.governance.audit import AuditLog
from tribune.governance.judge import HeuristicJudge, JudgeResult, LocalClassifierJudge, RemoteJudge
from tribune.orchestration.pipeline import CasePipeline
from tribune.types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    Evidence,
    IngestMethod,
    ProgramId,
    Provenance,
    RecommendedAction,
    SMState,
    VerifierVerdict,
)


class TestAuditJudge(unittest.TestCase):
    def setUp(self) -> None:
        self.store = LocalRuleStore()
        self.citations = self.store.all_citations(ProgramId.SNAP, "EX")
        self.valid_citation = self.citations[0]

        self.valid_evidence = [
            Evidence(
                evidence_id="e1",
                type="monthly_income",
                value=1200.0,
                provenance=Provenance(
                    source_doc_id="doc_1",
                    ingest_method=IngestMethod.STRUCTURED,
                    anonymized=True,
                    content_hash="abc",
                ),
            )
        ]

        self.valid_assessment = Assessment(
            assessment_id="c1:snap:a1",
            case_id="c1",
            program=ProgramId.SNAP,
            jurisdiction="EX",
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=[
                CriterionResult(
                    criterion_id="snap_gross_income",
                    description="Gross income test",
                    outcome=CriterionOutcome.SATISFIED,
                    required=True,
                    citation_ids=[self.valid_citation.citation_id],
                    evidence_ids=["e1"],
                )
            ],
            citations=[self.valid_citation],
            evidence_ids=["e1"],
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.92,
            rationale="Assessed 1 of 1 governing criteria. Met statutory gross income test.",
        )

        self.valid_verdict = VerifierVerdict(
            approved=True,
            recomputed_status=EligibilityStatus.LIKELY_ELIGIBLE,
            missing_citations=[],
            unsupported_claims=[],
            incomplete_coverage=[],
            reasons=["Confirmed by independent re-derivation"],
            self_testing_score=0.95,
        )

    def test_heuristic_judge_evaluation_valid_output(self) -> None:
        judge = HeuristicJudge()
        res = judge.evaluate(self.valid_assessment, self.valid_verdict, self.valid_evidence, "EX")
        self.assertTrue(res.passed)
        self.assertEqual(res.perceived_error_score, 0.0)
        self.assertEqual(res.uncited_claim_score, 0.0)
        self.assertEqual(res.citation_coverage_score, 1.0)
        self.assertGreaterEqual(res.judge_confidence, 0.9)
        self.assertLess(res.cost_estimate, 0.001)

    def test_heuristic_judge_detects_uncited_claims_and_errors(self) -> None:
        # Create invalid verdict with unsupported claims
        bad_verdict = VerifierVerdict(
            approved=False,
            recomputed_status=EligibilityStatus.LIKELY_INELIGIBLE,
            missing_citations=["snap_gross_income"],
            unsupported_claims=["criterion 'snap_gross_income': proposer said satisfied, re-derivation says not_satisfied"],
            incomplete_coverage=["snap_residency"],
            reasons=["Verification failed"],
            self_testing_score=0.4,
        )

        judge = HeuristicJudge()
        res = judge.evaluate(self.valid_assessment, bad_verdict, self.valid_evidence, "EX")
        self.assertFalse(res.passed)
        self.assertGreater(res.perceived_error_score, 0.5)

    def test_local_classifier_judge_cost_profile(self) -> None:
        judge = LocalClassifierJudge()
        res = judge.evaluate(self.valid_assessment, self.valid_verdict, self.valid_evidence, "EX")
        self.assertEqual(res.judge_name, "local_classifier_judge")
        # Target: ~82%+ lower cost than frontier cloud APIs ($0.015) -> cost is $0.00018 (~98.8% lower)
        frontier_cost = 0.015
        savings = (frontier_cost - res.cost_estimate) / frontier_cost
        self.assertGreater(savings, 0.82)

    def test_audit_log_live_execution_hook_and_query_store(self) -> None:
        audit = AuditLog()
        rec, judge_res = audit.evaluate_and_log_verifier(
            case_id="case_audit_1",
            assessment=self.valid_assessment,
            verdict=self.valid_verdict,
            evidence=self.valid_evidence,
            jurisdiction="EX",
        )
        self.assertEqual(rec.agent, "verifier_judge")
        self.assertIn("perceived_error_score", rec.payload)
        self.assertEqual(rec.payload["passed"], "True")

        # Query queryable store
        records = audit.query(case_id="case_audit_1", state=SMState.VERIFY)
        self.assertEqual(len(records), 1)

        # Calculate live metrics
        metrics = audit.calculate_live_metrics(case_id="case_audit_1")
        self.assertEqual(metrics["perceived_error_rate"], 0.0)
        self.assertEqual(metrics["uncited_claim_rate"], 0.0)
        self.assertEqual(metrics["eval_count"], 1.0)

    def test_audit_log_jsonl_export(self) -> None:
        audit = AuditLog()
        audit.evaluate_and_log_verifier(
            case_id="case_export_test",
            assessment=self.valid_assessment,
            verdict=self.valid_verdict,
            evidence=self.valid_evidence,
            jurisdiction="EX",
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            path = tf.name

        count = audit.export_jsonl(path, case_id="case_export_test")
        self.assertEqual(count, 1)

    def test_pipeline_continuous_monitoring_audits_every_verifier_output(self) -> None:
        gen = SyntheticCaseGenerator(seed=7)
        cases = gen.generate_demo_set()
        pipe = CasePipeline()

        for case in cases:
            res = pipe.run_case(case)
            # Ensure every program run produced verifier judge trace
            judge_records = pipe.audit.query(case_id=case.case_id, agent="verifier_judge")
            self.assertGreaterEqual(len(judge_records), 1)
            for j_rec in judge_records:
                self.assertIn("perceived_error_score", j_rec.payload)
                self.assertIn("judge_cost_usd", j_rec.payload)


if __name__ == "__main__":
    unittest.main()
