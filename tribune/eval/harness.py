"""Evaluation harness — the task x harness x verifier matrix runner.

Runs the full pipeline over a synthetic evaluation set with ground truth and turns
each program outcome into an :class:`EvalRecord`. The "matrix" is realized as:

* **task** — broken out per program (``per_program`` in the report),
* **harness** — the active provider/store/threshold configuration,
* **verifier** — the TRIBUNE-vs-verifier agreement column.

Returns both the raw records (for custom analysis) and the computed report.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..casegen.synthetic import SyntheticCaseGenerator
from ..config import TribuneSettings, get_settings
from ..orchestration.pipeline import CasePipeline
from ..types import EligibilityStatus, ProgramId, SyntheticCase
from .costreport import CostReport, compute_cost_report
from .metrics import EvalRecord, MetricsReport, compute_metrics

_STATUS_TO_LABEL = {
    EligibilityStatus.LIKELY_ELIGIBLE: "eligible",
    EligibilityStatus.LIKELY_INELIGIBLE: "ineligible",
    EligibilityStatus.INDETERMINATE: None,
}


@dataclass
class EvalResult:
    records: list[EvalRecord]
    report: MetricsReport
    cost_report: CostReport


def records_for_case(case: SyntheticCase, result) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    for program in case.target_programs:
        gt = case.ground_truth[program]
        outcome = result.outcome_for(program)
        if outcome is None:
            continue
        abstained = outcome.abstained
        predicted = None
        asserted_status = None
        if not abstained and outcome.assessment is not None:
            predicted = _STATUS_TO_LABEL.get(outcome.assessment.status)
            asserted_status = outcome.assessment.status.value
        verifier_status = outcome.verdict.recomputed_status.value if outcome.verdict else None
        usage = outcome.usage
        out.append(
            EvalRecord(
                case_id=case.case_id,
                program=program,
                abstained=abstained,
                ground_truth_label=gt.label.value,
                ambiguous=gt.ambiguous,
                predicted_label=predicted,
                asserted_status=asserted_status,
                verifier_status=verifier_status,
                tokens_input=usage.tokens_input if usage else 0,
                tokens_output=usage.tokens_output if usage else 0,
                cache_read_tokens=usage.cache_read_tokens if usage else 0,
                turns=usage.turns if usage else 0,
                cost_usd=usage.cost_usd if usage else None,
                tokenizer_ids=list(usage.tokenizer_ids) if usage else [],
                language=case.language,
                confidence=(
                    outcome.abstention.calibrated_confidence if outcome.abstention else None
                ),
                ocr_latency_ms=getattr(outcome, "ocr_latency_ms", 0.0) or getattr(result, "ocr_latency_ms", 0.0),
                citation_latency_ms=getattr(outcome, "citation_latency_ms", 0.0) or getattr(result, "citation_latency_ms", 0.0),
                llm_latency_ms=getattr(outcome, "llm_latency_ms", 0.0) or getattr(result, "llm_latency_ms", 0.0),
                total_latency_ms=getattr(outcome, "total_latency_ms", 0.0) or getattr(result, "total_latency_ms", 0.0),
            )
        )
    return out


class EvalHarness:
    def __init__(self, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        n_per_program: int = 24,
        ambiguous_ratio: float = 0.25,
        programs: list[ProgramId] | None = None,
        cases: list[SyntheticCase] | None = None,
    ) -> EvalResult:
        """Run the matrix. ``programs`` filters the generated set; ``cases``
        bypasses generation entirely (used by the frozen-seed-set harnesses)."""
        if cases is None:
            generator = SyntheticCaseGenerator(seed=self.settings.seed)
            cases = generator.generate_eval_set(
                n_per_program=n_per_program, ambiguous_ratio=ambiguous_ratio
            )
        if programs is not None:
            wanted = set(programs)
            cases = [c for c in cases if wanted.intersection(c.target_programs)]
        pipeline = CasePipeline(self.settings)
        records: list[EvalRecord] = []
        for case in cases:
            result = pipeline.run_case(case)
            records.extend(records_for_case(case, result))
        return EvalResult(
            records=records,
            report=compute_metrics(records),
            cost_report=compute_cost_report(records),
        )
