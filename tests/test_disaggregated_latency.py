"""Tests for Phase 4: Evaluation Metrics & Disaggregated Latency Instrumentation."""

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.corpus.citations import LateInteractionRetriever
from tribune.eval.harness import EvalHarness
from tribune.eval.metrics import EvalRecord, compute_metrics
from tribune.ingestion.ocr import OcrIngest
from tribune.orchestration.pipeline import CasePipeline
from tribune.types import ProgramId, RawDocument


def test_ocr_latency_instrumentation():
    from tribune.config import get_settings
    ocr = OcrIngest(get_settings())
    doc = RawDocument(
        doc_id="doc1",
        doc_type="paystub",
        text="Monthly Gross Earnings: $2100.00 | State: EX | Employer: Acme Corp",
    )
    evidence = ocr.ingest(doc)
    assert len(evidence) > 0
    assert ocr.last_latency_ms >= 0.0

    evidence_many = ocr.ingest_many([doc, doc])
    assert len(evidence_many) > 0
    assert ocr.last_latency_ms >= 0.0


def test_citation_retriever_latency_instrumentation():
    retriever = LateInteractionRetriever()
    score = retriever.score("SNAP gross income limit for 3 people", "7 CFR 273.9 Income and deductions.")
    assert score >= 0.0
    assert retriever.last_latency_ms >= 0.0

    docs = {"c1": "7 CFR 273.9 Income and deductions.", "c2": "7 CFR 273.2 Residency requirements."}
    ranked = retriever.rank("SNAP household income eligibility", docs, k=2)
    assert len(ranked) == 2
    assert retriever.last_latency_ms >= 0.0


def test_disaggregated_latency_in_metrics_report():
    records = [
        EvalRecord(
            case_id="case1",
            program=ProgramId.SNAP,
            abstained=False,
            ground_truth_label="eligible",
            ambiguous=False,
            predicted_label="eligible",
            ocr_latency_ms=12.5,
            citation_latency_ms=4.2,
            llm_latency_ms=85.0,
            total_latency_ms=105.0,
        ),
        EvalRecord(
            case_id="case2",
            program=ProgramId.SNAP,
            abstained=False,
            ground_truth_label="ineligible",
            ambiguous=False,
            predicted_label="ineligible",
            ocr_latency_ms=15.0,
            citation_latency_ms=5.0,
            llm_latency_ms=90.0,
            total_latency_ms=115.0,
        ),
    ]
    report = compute_metrics(records)
    assert abs(report.mean_ocr_latency_ms - 13.75) < 1e-6
    assert abs(report.mean_citation_latency_ms - 4.6) < 1e-6
    assert abs(report.mean_llm_latency_ms - 87.5) < 1e-6
    assert abs(report.mean_total_latency_ms - 110.0) < 1e-6

    rendered = report.render()
    assert "latency breakdown (disaggregated)" in rendered
    assert "OCR ingestion latency" in rendered
    assert "citation matching latency" in rendered
    assert "pure LLM generation" in rendered
    assert "total pipeline latency" in rendered


def test_eval_harness_end_to_end_latencies():
    gen = SyntheticCaseGenerator(seed=42)
    cases = gen.generate_eval_set(n_per_program=2, ambiguous_ratio=0.0)
    harness = EvalHarness()
    result = harness.run(cases=cases[:2])
    assert len(result.records) > 0
    for rec in result.records:
        assert rec.total_latency_ms >= 0.0
    rendered = result.report.render()
    assert "latency breakdown (disaggregated)" in rendered
