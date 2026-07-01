"""The kappa / alpha / false-confidence / abstention math matches hand-checked fixtures."""

from tribune.eval.metrics import (
    EvalRecord,
    abstention_aware_utility,
    cohens_kappa,
    compute_metrics,
    krippendorff_alpha,
)
from tribune.types import ProgramId


def test_cohens_kappa_fixture():
    a = ["a", "a", "b", "a"]
    b = ["a", "a", "b", "b"]
    # po=0.75, pe=0.5 -> kappa = 0.5
    assert abs(cohens_kappa(a, b) - 0.5) < 1e-9


def test_krippendorff_alpha_fixture():
    units = [("a", "a"), ("a", "a"), ("b", "b"), ("a", "b")]
    # Hand-computed via the coincidence matrix: alpha = 0.5333...
    assert abs(krippendorff_alpha(units) - 0.5333333333) < 1e-6


def test_abstention_aware_utility_fixture():
    recs = [
        EvalRecord("1", ProgramId.SNAP, abstained=False, ground_truth_label="eligible",
                   ambiguous=False, predicted_label="eligible"),       # +1.0 correct
        EvalRecord("2", ProgramId.SNAP, abstained=False, ground_truth_label="eligible",
                   ambiguous=False, predicted_label="ineligible"),     # -2.0 confidently wrong
        EvalRecord("3", ProgramId.SNAP, abstained=True, ground_truth_label="ineligible",
                   ambiguous=True, predicted_label=None),              # +1.0 abstain on ambiguous
        EvalRecord("4", ProgramId.SNAP, abstained=True, ground_truth_label="eligible",
                   ambiguous=False, predicted_label=None),             # +0.5 abstain on clear
    ]
    # (1 - 2 + 1 + 0.5) / 4 = 0.125
    assert abs(abstention_aware_utility(recs) - 0.125) < 1e-9


def test_false_confidence_rate_counts_wrong_assertions():
    recs = [
        EvalRecord("1", ProgramId.SNAP, abstained=False, ground_truth_label="eligible",
                   ambiguous=False, predicted_label="ineligible"),  # wrong assertion
        EvalRecord("2", ProgramId.SNAP, abstained=True, ground_truth_label="ineligible",
                   ambiguous=True, predicted_label=None),           # abstain (not false confidence)
        EvalRecord("3", ProgramId.SNAP, abstained=False, ground_truth_label="eligible",
                   ambiguous=False, predicted_label="eligible"),    # correct
    ]
    report = compute_metrics(recs)
    assert abs(report.false_confidence_rate - (1 / 3)) < 1e-9
    assert abs(report.false_confidence_rate_over_asserted - 0.5) < 1e-9
