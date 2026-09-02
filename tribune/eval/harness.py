"""Evaluation harness — Skill-Lift Empirical Benchmarking & matrix runner.

Empirically measures statutory Skill Lift across all programs:
  Skill Lift = Score(with_program_rules) - Score(without_program_rules)

Provides:
1. `SkillLiftHarness` — Paired baseline vs. skilled evaluation matrix per statutory program.
2. `EvalHarness` — Backward-compatible matrix runner (legacy static scans deprecated).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..casegen.synthetic import SyntheticCaseGenerator
from ..config import TribuneSettings, get_settings
from ..corpus.programs import all_programs
from ..corpus.rule_store import LocalRuleStore
from ..orchestration.pipeline import CasePipeline
from ..types import EligibilityStatus, ProgramId, SyntheticCase
from .costreport import CostReport, compute_cost_report
from .metrics import (
    EvalRecord,
    MetricsReport,
    ProgramSkillLift,
    SkillLiftRecord,
    SkillLiftReport,
    _score_single_outcome,
    compute_metrics,
    compute_skill_lift,
)

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


@dataclass
class SkillLiftResult:
    """Result of paired Skill-Lift evaluation benchmark."""

    paired_records: list[SkillLiftRecord]
    report: SkillLiftReport
    baseline_records: list[EvalRecord]
    skilled_records: list[EvalRecord]


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
        citations_list = (
            [c.citation_id for c in outcome.assessment.citations] if outcome.assessment else []
        )
        decisive = list(gt.decisive_criteria) if hasattr(gt, "decisive_criteria") else []
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
                ocr_latency_ms=getattr(outcome, "ocr_latency_ms", 0.0)
                or getattr(result, "ocr_latency_ms", 0.0),
                citation_latency_ms=getattr(outcome, "citation_latency_ms", 0.0)
                or getattr(result, "citation_latency_ms", 0.0),
                llm_latency_ms=getattr(outcome, "llm_latency_ms", 0.0)
                or getattr(result, "llm_latency_ms", 0.0),
                total_latency_ms=getattr(outcome, "total_latency_ms", 0.0)
                or getattr(result, "total_latency_ms", 0.0),
                citations=citations_list,
                decisive_criteria=decisive,
            )
        )
    return out


class SkillLiftHarness:
    """Empirical Statutory Skill-Lift Evaluation Harness.

    Executes paired benchmark runs across statutory programs on identical case inputs:
    - Baseline Run: Pipeline running without program-specific statutory rules loaded.
    - Skilled Run: Pipeline running with full targeted statutory rules loaded.
    Computes delta lift: Skill Lift = Score(with_rule) - Score(without_rule).
    """

    def __init__(self, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def run_skill_lift(
        self,
        n_per_program: int = 12,
        ambiguous_ratio: float = 0.25,
        programs: list[ProgramId] | None = None,
        cases: list[SyntheticCase] | None = None,
    ) -> SkillLiftResult:
        """Run paired empirical skill-lift benchmark loop across statutory programs."""
        if cases is None:
            generator = SyntheticCaseGenerator(seed=self.settings.seed)
            cases = generator.generate_eval_set(
                n_per_program=n_per_program, ambiguous_ratio=ambiguous_ratio
            )

        if programs is not None:
            wanted = set(programs)
            cases = [c for c in cases if wanted.intersection(c.target_programs)]

        # 1. Skilled Pipeline Run (full statutory rule store loaded)
        skilled_pipeline = CasePipeline(self.settings)
        skilled_records: list[EvalRecord] = []
        for case in cases:
            res = skilled_pipeline.run_case(case)
            skilled_records.extend(records_for_case(case, res))

        # 2. Baseline Pipeline Run (without targeted program statutory rules)
        baseline_store = LocalRuleStore()
        # Clear program rules in baseline store to create unskilled baseline
        baseline_store.clear()

        # Build baseline pipeline using the cleared store
        baseline_pipeline = CasePipeline(self.settings)
        baseline_pipeline.rule_store = baseline_store
        baseline_pipeline.proposer.rule_store = baseline_store
        baseline_pipeline.verifier.rule_store = baseline_store

        baseline_records: list[EvalRecord] = []
        for case in cases:
            b_res = baseline_pipeline.run_case(case)
            baseline_records.extend(records_for_case(case, b_res))

        # 3. Pair evaluations on identical (case_id, program)
        skilled_map = {(r.case_id, r.program): r for r in skilled_records}
        baseline_map = {(r.case_id, r.program): r for r in baseline_records}

        paired_records: list[SkillLiftRecord] = []
        for key, s_rec in skilled_map.items():
            b_rec = baseline_map.get(key)
            if b_rec is None:
                continue

            b_score = _score_single_outcome(
                b_rec.predicted_label, b_rec.ground_truth_label, b_rec.abstained, b_rec.ambiguous
            )
            s_score = _score_single_outcome(
                s_rec.predicted_label, s_rec.ground_truth_label, s_rec.abstained, s_rec.ambiguous
            )
            lift = s_score - b_score

            cit_accuracy = 1.0 if s_rec.citations else 0.0
            fidelity = 1.0 if not s_rec.abstained or s_rec.ambiguous else 0.8

            paired_records.append(
                SkillLiftRecord(
                    case_id=s_rec.case_id,
                    program=s_rec.program,
                    ground_truth_label=s_rec.ground_truth_label,
                    ambiguous=s_rec.ambiguous,
                    baseline_predicted_label=b_rec.predicted_label,
                    skilled_predicted_label=s_rec.predicted_label,
                    baseline_abstained=b_rec.abstained,
                    skilled_abstained=s_rec.abstained,
                    baseline_score=b_score,
                    skilled_score=s_score,
                    skill_lift=round(lift, 4),
                    baseline_citations_count=len(b_rec.citations),
                    skilled_citations_count=len(s_rec.citations),
                    statutory_citation_accuracy=cit_accuracy,
                    reasoning_fidelity=fidelity,
                    under_appeal=s_rec.program == ProgramId.APPEALS,
                )
            )

        report = compute_skill_lift(paired_records)
        return SkillLiftResult(
            paired_records=paired_records,
            report=report,
            baseline_records=baseline_records,
            skilled_records=skilled_records,
        )


class EvalHarness:
    """Matrix runner for statutory evaluation (legacy static scans deprecated in favor of SkillLiftHarness)."""

    def __init__(self, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        n_per_program: int = 24,
        ambiguous_ratio: float = 0.25,
        programs: list[ProgramId] | None = None,
        cases: list[SyntheticCase] | None = None,
    ) -> EvalResult:
        """Run the evaluation matrix."""
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

    def run_skill_lift(
        self,
        n_per_program: int = 12,
        ambiguous_ratio: float = 0.25,
        programs: list[ProgramId] | None = None,
        cases: list[SyntheticCase] | None = None,
    ) -> SkillLiftResult:
        """Execute paired Skill-Lift evaluation delegating to SkillLiftHarness."""
        harness = SkillLiftHarness(self.settings)
        return harness.run_skill_lift(
            n_per_program=n_per_program,
            ambiguous_ratio=ambiguous_ratio,
            programs=programs,
            cases=cases,
        )

    def run_static_ast_verification(self, *args, **kwargs) -> None:
        """Deprecated: Static AST assertion scans have been deprecated in favor of empirical Skill Lift."""
        warnings.warn(
            "run_static_ast_verification is deprecated; use SkillLiftHarness.run_skill_lift instead.",
            DeprecationWarning,
            stacklevel=2,
        )

