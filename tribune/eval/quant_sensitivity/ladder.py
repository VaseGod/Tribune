"""Run the verifier across the quantization ladder and score every rung."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...config import TribuneSettings, get_settings
from ...orchestration.pipeline import CasePipeline
from ...types import SyntheticCase
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


@dataclass
class LadderResult:
    manifest: dict
    reference_label: str
    rungs: list[RungResult] = field(default_factory=list)


def _run_rung(
    rung: QuantRung, cases: list[SyntheticCase], base_settings: TribuneSettings
) -> RungResult:
    settings = settings_for_rung(rung, base_settings)
    pipeline = CasePipeline(settings)
    mount_rung(pipeline, rung)
    records: list[EvalRecord] = []
    for case in cases:
        result = pipeline.run_case(case)
        records.extend(records_for_case(case, result))
    ece, brier = calibration_over_assertions(records)
    return RungResult(
        rung=rung,
        records=records,
        report=compute_metrics(records),
        cost_report=compute_cost_report(records, scope=rung.label),
        ece=ece,
        brier=brier,
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
    rung.abstention_decision_agreement = sum(
        1 for a, b in pairs if a.abstained == b.abstained
    ) / len(pairs)
    rung.ece_drift = rung.ece - reference.ece


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
    return out
