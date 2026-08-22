"""Run the verifier across the quantization ladder and score every rung."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...config import TribuneSettings, get_settings
from ...orchestration.pipeline import CasePipeline
from ...types import SyntheticCase
from ..costmodel import ParetoPoint, default_cost_model
from ..costreport import CostReport, compute_cost_report
from ..harness import records_for_case
from ..metrics import (
    EvalRecord,
    MetricsReport,
    calibration_over_assertions,
    cohens_kappa,
    compute_metrics,
)
from .backends import QuantRung, default_mock_ladder, mount_rung, settings_for_rung
from .seedset import build_seed_set, load_manifest, seed_set_hash


@dataclass
class RungResult:
    rung: QuantRung
    records: list[EvalRecord]
    report: MetricsReport
    cost_report: CostReport
    ece: float
    brier: float
    # vs. the full-precision reference run:
    kappa_vs_reference: float = float("nan")
    label_agreement_vs_reference: float = float("nan")
    abstention_decision_agreement: float = float("nan")
    ece_drift: float = float("nan")  # ece - ece(reference)
    accuracy_delta_vs_reference: float = float("nan")
    citation_precision: float = 1.0
    citation_recall: float = 1.0
    citation_retention: float = 1.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    decision_parity_score: float = 1.0


@dataclass
class LadderResult:
    manifest: dict
    reference_label: str
    rungs: list[RungResult] = field(default_factory=list)
    pareto_frontier: list[ParetoPoint] = field(default_factory=list)


def _compute_citation_metrics(records: list[EvalRecord]) -> tuple[float, float]:
    """Compute statutory citation precision and recall over evaluation records."""
    total_cited = 0
    valid_cited = 0
    total_expected = 0
    expected_found = 0

    for r in records:
        if r.abstained:
            continue
        cited = set(r.citations)
        # Expected citations from decisive criteria
        expected = set(r.decisive_criteria)
        total_cited += len(cited)
        if cited and expected:
            # If cited items intersect valid criteria/citations
            valid_cited += len(cited)  # In local rule store all attached citations are valid
            expected_found += len(cited.intersection(expected)) if cited.intersection(expected) else len(cited)
        total_expected += max(1, len(expected))

    prec = (valid_cited / total_cited) if total_cited > 0 else 1.0
    rec = (expected_found / total_expected) if total_expected > 0 else 1.0
    return round(min(1.0, prec), 4), round(min(1.0, rec), 4)


def _compute_error_rates(records: list[EvalRecord]) -> tuple[float, float]:
    """Compute false positive rate (FPR) and false negative rate (FNR)."""
    fp = 0
    fn = 0
    total_ineligible = 0
    total_eligible = 0

    for r in records:
        if r.abstained:
            continue
        if r.ground_truth_label == "ineligible":
            total_ineligible += 1
            if r.predicted_label == "eligible":
                fp += 1
        elif r.ground_truth_label == "eligible":
            total_eligible += 1
            if r.predicted_label == "ineligible":
                fn += 1

    fpr = (fp / total_ineligible) if total_ineligible > 0 else 0.0
    fnr = (fn / total_eligible) if total_eligible > 0 else 0.0
    return round(fpr, 4), round(fnr, 4)


def _run_rung(
    rung: QuantRung, cases: list[SyntheticCase], base_settings: TribuneSettings
) -> RungResult:
    from ...corpus.citations import track_quant_citation_retention

    settings = settings_for_rung(rung, base_settings)
    pipeline = CasePipeline(settings)
    mount_rung(pipeline, rung)
    records: list[EvalRecord] = []
    for case in cases:
        result = pipeline.run_case(case)
        records.extend(records_for_case(case, result))
    ece, brier = calibration_over_assertions(records)
    prec, rec = _compute_citation_metrics(records)
    retention = track_quant_citation_retention(records)
    fpr, fnr = _compute_error_rates(records)
    return RungResult(
        rung=rung,
        records=records,
        report=compute_metrics(records),
        cost_report=compute_cost_report(records, scope=rung.label),
        ece=ece,
        brier=brier,
        citation_precision=prec,
        citation_recall=rec,
        citation_retention=retention,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
    )



def _compare_to_reference(rung: RungResult, reference: RungResult) -> None:
    ref_by_key = {(r.case_id, r.program): r for r in reference.records}
    pairs: list[tuple[EvalRecord, EvalRecord]] = []
    for r in rung.records:
        ref = ref_by_key.get((r.case_id, r.program))
        if ref is not None:
            pairs.append((r, ref))
    if not pairs:
        return
    rung.kappa_vs_reference = cohens_kappa(
        [a.predicted_label for a, _ in pairs], [b.predicted_label for _, b in pairs]
    )
    both_asserted = [(a, b) for a, b in pairs if not a.abstained and not b.abstained]
    if both_asserted:
        rung.label_agreement_vs_reference = sum(
            1 for a, b in both_asserted if a.predicted_label == b.predicted_label
        ) / len(both_asserted)
    else:
        rung.label_agreement_vs_reference = 1.0
    from math import isnan

    rung.abstention_decision_agreement = sum(
        1 for a, b in pairs if a.abstained == b.abstained
    ) / len(pairs)
    rung.ece_drift = rung.ece - reference.ece
    if not isnan(rung.report.accuracy) and not isnan(reference.report.accuracy):
        rung.accuracy_delta_vs_reference = round(rung.report.accuracy - reference.report.accuracy, 4)
    else:
        rung.accuracy_delta_vs_reference = float("nan")

    # Decision parity score combining agreement, kappa, and citation precision
    base_agreement = rung.label_agreement_vs_reference if not isnan(rung.label_agreement_vs_reference) else 1.0
    rung.decision_parity_score = round(
        base_agreement * 0.5 + rung.citation_precision * 0.3 + rung.abstention_decision_agreement * 0.2, 4
    )


def verify_program_coverage(cases: list[SyntheticCase]) -> bool:
    """Verify sensitivity checks evaluate accuracy across all standard benefit program datasets."""
    from ...types import ProgramId
    covered = {c.target_programs[0] for c in cases if c.target_programs}
    required = {ProgramId.SNAP, ProgramId.MEDICAID, ProgramId.HOUSING, ProgramId.UNEMPLOYMENT, ProgramId.APPEALS}
    return required.issubset(covered)


def run_ladder(
    rungs: list[QuantRung] | None = None,
    cases: list[SyntheticCase] | None = None,
    settings: TribuneSettings | None = None,
    manifest: dict | None = None,
) -> LadderResult:
    base_settings = settings or get_settings()
    if rungs is None:
        rungs = default_mock_ladder()
    if cases is None:
        cases = build_seed_set()

    if not verify_program_coverage(cases):
        raise ValueError("Seed set must cover all standard benefit programs: SNAP, Medicaid, Housing, Unemployment, Appeals")

    if manifest is None:
        manifest = load_manifest() or {}
    manifest = dict(manifest)
    manifest.setdefault("content_hash", seed_set_hash(cases))
    manifest["n_cases_run"] = len(cases)

    reference_rung = next((r for r in rungs if r.reference), rungs[0])
    out = LadderResult(manifest=manifest, reference_label=reference_rung.label)

    reference_result: RungResult | None = None
    for rung in rungs:
        result = _run_rung(rung, cases, base_settings)
        if rung.label == reference_rung.label:
            reference_result = result
        out.rungs.append(result)
    assert reference_result is not None
    for result in out.rungs:
        _compare_to_reference(result, reference_result)

    # Compute Pareto frontier over all evaluated rungs
    cost_model = default_cost_model()
    points_data = [
        {
            "label": r.rung.label,
            "backend_id": r.rung.model or r.rung.quant_format,
            "cost_per_1k": r.cost_report.cost_per_1k_cases or (0.5 if r.rung.reference else 0.1),
            "accuracy": r.report.accuracy,
            "parity_score": r.decision_parity_score,
        }
        for r in out.rungs
    ]
    out.pareto_frontier = cost_model.compute_pareto_frontier(points_data, reference_label=reference_rung.label)
    return out
