"""Tests for Multi-Format Quantization Sensitivity and Statutory Parity Benchmarking."""

from __future__ import annotations

import pytest

from tribune.eval.quant_sensitivity.backends import (
    DisaggregatedLatencyMetrics,
    QuantRung,
    StatutoryParityBenchmark,
    multi_format_quant_ladder,
    run_statutory_parity_audit,
)


def test_multi_format_quant_ladder_contents() -> None:
    ladder = multi_format_quant_ladder()
    formats = {r.quant_format for r in ladder}
    labels = {r.label for r in ladder}

    # Verify FP8 coverage
    assert "fp8_e4m3" in formats or "fp8_e4m3" in labels
    assert "fp8_e5m2" in formats or "fp8_e5m2" in labels

    # Verify 4-bit and 3-bit GGUF coverage
    assert any("iq4" in f or "q4" in f for f in formats)
    assert any("iq3" in f or "q3" in f for f in formats)

    # Reference rung exists
    ref_rungs = [r for r in ladder if r.reference]
    assert len(ref_rungs) == 1
    assert ref_rungs[0].label == "fp16"


def test_disaggregated_latency_profiling() -> None:
    ladder = multi_format_quant_ladder()
    for rung in ladder:
        lat = rung.latency_profile
        assert isinstance(lat, DisaggregatedLatencyMetrics)
        assert lat.ttft_ms > 0
        assert lat.itl_ms > 0
        assert lat.total_latency_ms > 0
        assert lat.tokens_per_second > 0


def test_statutory_parity_audit_zero_regression() -> None:
    audit = run_statutory_parity_audit()
    assert len(audit) > 0

    for rung_eval in audit:
        # Assert 0% regression in citation precision & parity
        assert rung_eval["avg_citation_precision"] >= 0.985
        assert rung_eval["avg_parity_score"] >= 0.980
        assert rung_eval["zero_regression_verified"] is True

        # Verify all statutory domains are evaluated
        domains = rung_eval["domains"]
        for required_dom in ("medicaid", "snap", "housing", "unemployment", "appeals"):
            assert required_dom in domains
            assert domains[required_dom]["regression_detected"] is False
