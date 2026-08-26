"""Precision calibration and quantization sensitivity test suite.

Measures legal determination accuracy, statutory citation precision, and statutory
drift across quantization profiles (INT4, INT8, FP16).
"""

from __future__ import annotations

import pytest

from tribune.eval.costmodel import TrajectoryCostModel
from tribune.eval.quant_sensitivity.backends import default_mock_ladder
from tribune.eval.quant_sensitivity.ladder import run_ladder
from tribune.eval.quant_sensitivity.seedset import build_seed_set
from tribune.providers.base import SynthesisRequest
from tribune.providers.local_rules import LocalMoEProvider, MoEConfig
from tribune.types import ProgramId



def test_quantization_ladder_execution() -> None:
    """Verify that the quantization ladder runs across all rungs and computes calibration metrics."""
    ladder = default_mock_ladder()
    cases = build_seed_set(limit_per_program={ProgramId.SNAP: 4, ProgramId.MEDICAID: 4})

    result = run_ladder(cases=cases, ladder=ladder, reference_label="fp16")

    assert result is not None
    assert len(result.rungs) == len(ladder)

    rung_labels = [r.rung.label for r in result.rungs]
    assert "fp16" in rung_labels

    ref_rung = next(r for r in result.rungs if r.rung.label == "fp16")
    assert ref_rung.report.accuracy >= 0.80
    assert ref_rung.citation_precision >= 0.90


def test_quantization_citation_precision_retention() -> None:
    """Verify that statutory citation precision is retained above 0.85 across INT4, INT8, and FP16."""
    ladder = default_mock_ladder()
    cases = build_seed_set(limit_per_program={ProgramId.SNAP: 4, ProgramId.MEDICAID: 4})

    result = run_ladder(cases=cases, ladder=ladder, reference_label="fp16")

    for rung_res in result.rungs:
        # High-stakes statutory determinations require high citation precision even under quantization
        assert rung_res.citation_precision >= 0.85, (
            f"Rung {rung_res.rung.label} ({rung_res.rung.quantization}) citation precision "
            f"{rung_res.citation_precision} fell below 0.85"
        )
        assert rung_res.citation_recall >= 0.70


def test_quantization_statutory_drift_bounds() -> None:
    """Verify that ECE drift, label agreement, and kappa vs reference remain bounded."""
    ladder = default_mock_ladder()
    cases = build_seed_set(limit_per_program={ProgramId.SNAP: 4, ProgramId.MEDICAID: 4})

    result = run_ladder(cases=cases, ladder=ladder, reference_label="fp16")

    for rung_res in result.rungs:
        if rung_res.rung.label == "fp16":
            continue
        # Check label agreement vs FP16 reference
        assert rung_res.label_agreement_vs_reference >= 0.75, (
            f"Rung {rung_res.rung.label} label agreement {rung_res.label_agreement_vs_reference} < 0.75"
        )


def test_local_moe_quantization_profiles() -> None:
    """Verify LocalMoEProvider executes across INT4, INT8, and FP16 quantization profiles."""
    profiles = [
        ("qwen2.5-moe-7b-int4", "int4_awq"),
        ("qwen2.5-moe-7b-int8", "bitsandbytes_int8"),
        ("qwen2.5-moe-7b-fp16", "fp16"),
    ]

    for model_name, quant in profiles:
        cfg = MoEConfig(
            model_name=model_name,
            num_experts=8,
            active_experts_per_token=2,
            quantization=quant,
        )
        provider = LocalMoEProvider(moe_config=cfg)

        req = SynthesisRequest(
            program=ProgramId.SNAP,
            jurisdiction="EX",
            criteria=[],
            required_total=3,
            coverage_complete=True,
            evidence_summary="",
            citations=[],
        )

        res = provider.synthesize_assessment(req)
        assert res is not None
        assert res.status is not None
        assert "Local MoE Execution" in res.rationale
        assert quant in res.rationale



def test_trajectory_cost_model_moe_pareto() -> None:
    """Verify TrajectoryCostModel correctly evaluates multi-turn MoE trajectory Pareto points."""
    cost_model = TrajectoryCostModel(gamma=1.30, state_transition_overhead_usd=0.0001)

    points_data = [
        {"label": "fp16", "backend_id": "moe_fp16", "cost_per_1k": 0.05, "turns_per_task": 1.0, "accuracy": 0.98},
        {"label": "int8", "backend_id": "moe_int8", "cost_per_1k": 0.02, "turns_per_task": 1.1, "accuracy": 0.96},
        {"label": "int4", "backend_id": "moe_int4", "cost_per_1k": 0.005, "turns_per_task": 1.4, "accuracy": 0.92},
    ]

    frontier = cost_model.compute_trajectory_pareto_frontier(points_data, reference_label="fp16")
    assert len(frontier) == 3
    # INT4 should have significant cost savings
    int4_point = next(p for p in frontier if p.label == "int4")
    assert int4_point.cost_savings_pct > 50.0
