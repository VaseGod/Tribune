"""Tests for Empirical Skill-Lift Benchmarking across statutory programs."""

import os
import tempfile
import warnings

import pytest

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.eval.canary import CanarySentinel, SkillLiftCanaryReport
from tribune.eval.harness import EvalHarness, SkillLiftHarness, SkillLiftResult
from tribune.eval.metrics import (
    ProgramSkillLift,
    SkillLiftRecord,
    SkillLiftReport,
    compute_skill_lift,
)
from tribune.types import ProgramId


def test_skill_lift_record_and_scoring():
    """Verify SkillLiftRecord creation and compute_skill_lift aggregation math."""
    records = [
        SkillLiftRecord(
            case_id="case_1",
            program=ProgramId.SNAP,
            ground_truth_label="eligible",
            ambiguous=False,
            baseline_predicted_label="ineligible",
            skilled_predicted_label="eligible",
            baseline_score=-2.0,
            skilled_score=1.0,
            skill_lift=3.0,
            baseline_citations_count=0,
            skilled_citations_count=2,
            statutory_citation_accuracy=1.0,
            reasoning_fidelity=1.0,
        ),
        SkillLiftRecord(
            case_id="case_2",
            program=ProgramId.SNAP,
            ground_truth_label="ineligible",
            ambiguous=False,
            baseline_predicted_label=None,
            skilled_predicted_label="ineligible",
            baseline_abstained=True,
            skilled_abstained=False,
            baseline_score=0.5,
            skilled_score=1.0,
            skill_lift=0.5,
            baseline_citations_count=0,
            skilled_citations_count=1,
            statutory_citation_accuracy=1.0,
            reasoning_fidelity=1.0,
        ),
    ]

    report = compute_skill_lift(records)
    assert report.n_total_cases == 2
    assert report.mean_skill_lift == 1.75
    assert report.accuracy_lift > 0.0
    assert "snap" in report.per_program
    snap_lift = report.per_program["snap"]
    assert snap_lift.baseline_accuracy == 0.0
    assert snap_lift.skilled_accuracy == 1.0
    assert snap_lift.accuracy_lift == 1.0


def test_skill_lift_report_render():
    """Verify SkillLiftReport render produces formatted multi-line summary."""
    report = SkillLiftReport(
        scope="overall",
        n_total_cases=10,
        mean_skill_lift=0.85,
        accuracy_lift=0.40,
        utility_lift=0.65,
        citation_accuracy_lift=0.95,
        appeal_fidelity_lift=0.90,
        per_program={
            "snap": ProgramSkillLift(
                program="snap",
                n_cases=5,
                baseline_accuracy=0.5,
                skilled_accuracy=0.9,
                accuracy_lift=0.4,
                baseline_utility=0.2,
                skilled_utility=0.8,
                utility_lift=0.6,
                baseline_fcr=0.3,
                skilled_fcr=0.0,
                fcr_reduction=0.3,
                statutory_citation_accuracy=1.0,
                appeal_reasoning_fidelity=1.0,
                mean_skill_lift=0.8,
            )
        },
    )

    rendered = report.render()
    assert "TRIBUNE Empirical Skill-Lift Report" in rendered
    assert "NET EMPIRICAL SKILL LIFT" in rendered
    assert "[snap        ]" in rendered


def test_skill_lift_harness_paired_execution():
    """Verify SkillLiftHarness executes paired baseline vs skilled runs across all programs."""
    harness = SkillLiftHarness()
    programs = [
        ProgramId.SNAP,
        ProgramId.MEDICAID,
        ProgramId.HOUSING,
        ProgramId.UNEMPLOYMENT,
        ProgramId.APPEALS,
    ]

    result: SkillLiftResult = harness.run_skill_lift(
        n_per_program=2,
        programs=programs,
    )

    assert len(result.paired_records) > 0
    assert len(result.skilled_records) > 0
    assert len(result.baseline_records) > 0

    report = result.report
    assert report.n_total_cases > 0
    # Skilled pipeline with full statutory rules should achieve non-negative net skill lift
    assert report.mean_skill_lift >= 0.0


def test_skill_lift_canary_sentinel():
    """Verify CanarySentinel tracks empirical Skill Lift baselines and detects regression deltas."""
    sentinel = CanarySentinel()

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_file = os.path.join(tmpdir, "canary_skill_lift.json")

        # 1. Initialize baseline
        report1 = sentinel.run_skill_lift_canary(
            baseline_path=baseline_file,
            freeze=True,
            n_per_program=2,
        )
        assert report1.baseline_initialized is True
        assert report1.ok is True
        assert os.path.exists(baseline_file)

        # 2. Subsequent run with tight tolerance
        report2 = sentinel.run_skill_lift_canary(
            baseline_path=baseline_file,
            freeze=False,
            epsilon=0.50,
            n_per_program=2,
        )
        assert report2.baseline_initialized is False
        assert report2.ok is True


def test_deprecated_static_verification_warning():
    """Verify legacy run_static_ast_verification emits DeprecationWarning."""
    harness = EvalHarness()
    with pytest.deprecated_call():
        harness.run_static_ast_verification()
