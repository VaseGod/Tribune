"""Evaluation metrics.

All math here is implemented directly (no metric libraries) so it runs offline and
can be hand-checked against fixtures:

* **raw agreement** vs. ground truth (over asserted cases),
* **Cohen's kappa** (two raters, chance-corrected),
* **Krippendorff's alpha** (nominal, via the coincidence matrix, missing-data
  aware — abstentions are treated as missing ratings),
* **false-confidence rate** — the headline safety metric: how often an asserted
  result disagrees with ground truth instead of abstaining,
* **abstention-aware utility** — full-recall scoring that rewards abstaining on a
  genuinely ambiguous case and heavily penalizes a confidently-wrong assertion.

Why both raw agreement and chance-corrected agreement? On class-imbalanced case
data, raw agreement overstates reliability; kappa/alpha reveal how much.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isnan

from ..types import ProgramId

# Utility weights for abstention-aware scoring.
_W_CORRECT = 1.0
_W_WRONG = -2.0  # confidently wrong is the cardinal harm
_W_ABSTAIN_AMBIGUOUS = 1.0  # rewarded
_W_ABSTAIN_CLEAR = 0.5  # safe but a missed opportunity


class TaskOutcomeType(str, enum.Enum):
    """The four completed-task outcome types cost reports break down by.

    All four are *completed* tasks. A correct abstention is a success outcome —
    when the determinative fact lives with the claimant or a caseworker, routing
    to a human is the right answer — and it is costed at its actual (low) cost,
    never as a failure and never as infinite cost.
    """

    CORRECT_DETERMINATION = "correct_determination"
    CORRECT_ABSTENTION = "correct_abstention"
    INCORRECT_DETERMINATION = "incorrect_determination"
    OVER_REFUSAL = "over_refusal"  # abstained on a clear case: safe, but a missed opportunity


@dataclass
class EvalRecord:
    case_id: str
    program: ProgramId
    abstained: bool
    ground_truth_label: str  # "eligible" | "ineligible"
    ambiguous: bool
    predicted_label: str | None = None  # None when abstained
    asserted_status: str | None = None
    verifier_status: str | None = None
    # -- additive usage/cost fields (Phase 1); defaults keep old callers valid -- #
    tokens_input: int = 0
    tokens_output: int = 0
    cache_read_tokens: int = 0
    turns: int = 0
    cost_usd: float | None = None
    tokenizer_ids: list[str] = field(default_factory=list)
    language: str = "en"
    confidence: float | None = None  # calibrated confidence behind assert/abstain
    # -- disaggregated latency breakdown (Phase 4) -- #
    ocr_latency_ms: float = 0.0
    citation_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    step_latencies: dict[str, float] = field(default_factory=dict)
    fold_planning_latency_ms: float = 0.0
    json_extraction_latency_ms: float = 0.0
    synthesis_latency_ms: float = 0.0
    # -- statutory citations & decisive criteria -- #
    citations: list[str] = field(default_factory=list)
    decisive_criteria: list[str] = field(default_factory=list)
    # -- continuous audit judge metrics (Phase 3) -- #
    perceived_error_score: float | None = None
    uncited_claim_score: float | None = None
    citation_coverage_score: float | None = None
    rule_reference_coverage: float | None = None
    judge_confidence: float | None = None
    judge_cost_usd: float = 0.0


def classify_outcome(r: EvalRecord) -> TaskOutcomeType:
    if r.abstained:
        return (
            TaskOutcomeType.CORRECT_ABSTENTION if r.ambiguous else TaskOutcomeType.OVER_REFUSAL
        )
    if r.predicted_label == r.ground_truth_label:
        return TaskOutcomeType.CORRECT_DETERMINATION
    return TaskOutcomeType.INCORRECT_DETERMINATION


def raw_agreement(a: list[str | None], b: list[str | None]) -> float:
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohens_kappa(a: list[str | None], b: list[str | None]) -> float:
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({x for x, _ in pairs} | {y for _, y in pairs})
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum((ca[lab] / n) * (cb[lab] / n) for lab in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha(units: list[tuple[str | None, ...]]) -> float:
    """Nominal Krippendorff's alpha over a units x coders table; None = missing."""
    coincidence: dict[tuple[str, str], float] = defaultdict(float)
    values: set[str] = set()
    for unit in units:
        present = [v for v in unit if v is not None]
        m = len(present)
        if m < 2:
            continue
        values.update(present)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(present[i], present[j])] += 1.0 / (m - 1)
    vals = sorted(values)
    if not vals:
        return float("nan")
    n_c = {c: sum(coincidence[(c, k)] for k in vals) for c in vals}
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    do = sum(coincidence[(c, k)] for c in vals for k in vals if c != k) / n
    de = sum(n_c[c] * n_c[k] for c in vals for k in vals if c != k) / (n * (n - 1))
    if de == 0:
        return 1.0
    return 1.0 - do / de


def brier_score(confidences: list[float], corrects: list[bool]) -> float:
    """Mean squared error of confidence vs. correctness (lower is better)."""
    pairs = list(zip(confidences, corrects, strict=True))
    if not pairs:
        return float("nan")
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / len(pairs)


def expected_calibration_error(
    confidences: list[float], corrects: list[bool], n_bins: int = 10
) -> float:
    """ECE with equal-width bins: sum_b (n_b/n) * |acc_b - conf_b| (lower is better)."""
    pairs = list(zip(confidences, corrects, strict=True))
    n = len(pairs)
    if n == 0:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in pairs:
        idx = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[idx].append((c, ok))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n) * abs(accuracy - avg_conf)
    return ece


def calibration_over_assertions(records: list[EvalRecord]) -> tuple[float, float]:
    """(ECE, Brier) of the calibrated confidence over *asserted* records.

    Abstained records carry no assertion to score, so they are excluded here —
    their calibration behavior is captured by the abstention-rate and
    over-refusal metrics instead. This never scores an abstention as a failure.
    """
    asserted = [r for r in records if not r.abstained and r.confidence is not None]
    confidences = [float(r.confidence) for r in asserted]  # type: ignore[arg-type]
    corrects = [r.predicted_label == r.ground_truth_label for r in asserted]
    return (
        expected_calibration_error(confidences, corrects),
        brier_score(confidences, corrects),
    )


def abstention_aware_utility(records: list[EvalRecord]) -> float:
    if not records:
        return float("nan")
    total = 0.0
    for r in records:
        if r.abstained:
            total += _W_ABSTAIN_AMBIGUOUS if r.ambiguous else _W_ABSTAIN_CLEAR
        elif r.predicted_label == r.ground_truth_label:
            total += _W_CORRECT
        else:
            total += _W_WRONG
    return total / len(records)


@dataclass
class MetricsReport:
    scope: str
    n: int
    asserted: int
    abstained: int
    raw_agreement_vs_truth: float
    cohen_kappa: float
    krippendorff_alpha: float
    false_confidence_rate: float
    false_confidence_rate_over_asserted: float
    abstention_rate: float
    abstention_precision: float
    abstention_recall: float
    over_refusal_rate: float
    abstention_aware_utility: float
    verifier_agreement: float
    mean_ocr_latency_ms: float = float("nan")
    mean_citation_latency_ms: float = float("nan")
    mean_llm_latency_ms: float = float("nan")
    mean_total_latency_ms: float = float("nan")
    mean_step_latencies: dict[str, float] = field(default_factory=dict)
    perceived_error_rate: float = float("nan")
    uncited_claim_rate: float = float("nan")
    mean_judge_confidence: float = float("nan")
    mean_citation_coverage: float = float("nan")
    mean_rule_reference_coverage: float = float("nan")
    mean_judge_cost_usd: float = float("nan")
    total_judge_cost_usd: float = 0.0
    per_program: dict[str, MetricsReport] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Alias for raw_agreement_vs_truth."""
        return self.raw_agreement_vs_truth

    def render(self) -> str:
        def f(x: float) -> str:
            return "  n/a" if isnan(x) else f"{x:6.3f}"

        def f_ms(x: float) -> str:
            return "    n/a" if isnan(x) else f"{x:7.2f} ms"

        lines = [
            f"=== TRIBUNE evaluation — {self.scope} (n={self.n}) ===",
            f"  asserted / abstained            : {self.asserted} / {self.abstained}",
            f"  raw agreement vs ground truth   : {f(self.raw_agreement_vs_truth)}   (asserted only)",
            f"  Cohen's kappa                   : {f(self.cohen_kappa)}   (chance-corrected)",
            f"  Krippendorff's alpha            : {f(self.krippendorff_alpha)}   (abstention = missing)",
            f"  FALSE-CONFIDENCE RATE           : {f(self.false_confidence_rate)}   <-- headline safety metric",
            f"    (over asserted only)          : {f(self.false_confidence_rate_over_asserted)}",
            f"  abstention rate                 : {f(self.abstention_rate)}",
            f"  abstention precision (on ambig.): {f(self.abstention_precision)}",
            f"  abstention recall (of ambig.)   : {f(self.abstention_recall)}",
            f"  over-refusal rate               : {f(self.over_refusal_rate)}   (abstained on clear cases)",
            f"  abstention-aware utility        : {f(self.abstention_aware_utility)}   (higher is better; max 1.0)",
            f"  TRIBUNE vs verifier agreement   : {f(self.verifier_agreement)}",
            "  --- continuous audit & judge metrics ---",
            f"  perceived error rate (mean)     : {f(self.perceived_error_rate)}",
            f"  uncited claim rate (mean)       : {f(self.uncited_claim_rate)}",
            f"  statutory citation coverage     : {f(self.mean_citation_coverage)}",
            f"  rule reference coverage         : {f(self.mean_rule_reference_coverage)}",
            f"  judge confidence (mean)         : {f(self.mean_judge_confidence)}",
            f"  judge evaluation cost (total)   : ${self.total_judge_cost_usd:.6f}",
            "  --- latency breakdown (disaggregated) ---",
            f"  OCR ingestion latency (mean)    : {f_ms(self.mean_ocr_latency_ms)}",
            f"  citation matching latency (mean): {f_ms(self.mean_citation_latency_ms)}",
            f"  pure LLM generation (mean)      : {f_ms(self.mean_llm_latency_ms)}",
            f"  total pipeline latency (mean)   : {f_ms(self.mean_total_latency_ms)}",
        ]
        if self.mean_step_latencies:
            for s_name, s_val in sorted(self.mean_step_latencies.items()):
                lines.append(f"  {s_name} (mean) : {f_ms(s_val)}")
        return "\n".join(lines)

    def render_full(self) -> str:
        parts = [self.render()]
        if self.per_program:
            parts.append("\n  --- per program ---")
            for name, rep in self.per_program.items():
                parts.append(f"\n  [{name}]")
                parts.append("\n".join("  " + ln for ln in rep.render().splitlines()[1:]))
        return "\n".join(parts)


def _compute(scope: str, records: list[EvalRecord]) -> MetricsReport:
    n = len(records)
    asserted = [r for r in records if not r.abstained]
    abstained = [r for r in records if r.abstained]

    tribune_labels = [r.predicted_label for r in records]  # None when abstained
    truth_labels = [r.ground_truth_label for r in records]

    raw = raw_agreement([r.predicted_label for r in asserted], [r.ground_truth_label for r in asserted])
    kappa = cohens_kappa([r.predicted_label for r in asserted], [r.ground_truth_label for r in asserted])
    alpha = krippendorff_alpha(list(zip(tribune_labels, truth_labels, strict=False)))

    wrong = [r for r in asserted if r.predicted_label != r.ground_truth_label]
    fcr = (len(wrong) / n) if n else float("nan")
    fcr_asserted = (len(wrong) / len(asserted)) if asserted else float("nan")

    ambiguous = [r for r in records if r.ambiguous]
    abstained_ambiguous = [r for r in abstained if r.ambiguous]
    abst_rate = (len(abstained) / n) if n else float("nan")
    abst_prec = (len(abstained_ambiguous) / len(abstained)) if abstained else float("nan")
    abst_recall = (len(abstained_ambiguous) / len(ambiguous)) if ambiguous else float("nan")
    over_refusal = ((len(abstained) - len(abstained_ambiguous)) / n) if n else float("nan")

    util = abstention_aware_utility(records)

    matched = [
        r for r in asserted if r.asserted_status is not None and r.asserted_status == r.verifier_status
    ]
    verifier_agreement = (len(matched) / len(asserted)) if asserted else float("nan")

    ocr_latencies = [r.ocr_latency_ms for r in records if r.ocr_latency_ms > 0]
    cit_latencies = [r.citation_latency_ms for r in records if r.citation_latency_ms > 0]
    llm_latencies = [r.llm_latency_ms for r in records if r.llm_latency_ms > 0]
    tot_latencies = [r.total_latency_ms for r in records if r.total_latency_ms > 0]

    mean_ocr = (sum(ocr_latencies) / len(ocr_latencies)) if ocr_latencies else 0.0
    mean_cit = (sum(cit_latencies) / len(cit_latencies)) if cit_latencies else 0.0
    mean_llm = (sum(llm_latencies) / len(llm_latencies)) if llm_latencies else 0.0
    mean_tot = (sum(tot_latencies) / len(tot_latencies)) if tot_latencies else 0.0

    # Continuous Audit Judge aggregations
    perceived_errs = [r.perceived_error_score for r in records if r.perceived_error_score is not None]
    uncited_claims = [r.uncited_claim_score for r in records if r.uncited_claim_score is not None]
    cit_coverages = [r.citation_coverage_score for r in records if r.citation_coverage_score is not None]
    rule_coverages = [r.rule_reference_coverage for r in records if r.rule_reference_coverage is not None]
    judge_confs = [r.judge_confidence for r in records if r.judge_confidence is not None]
    judge_costs = [r.judge_cost_usd for r in records if r.judge_cost_usd > 0]

    p_err = (sum(perceived_errs) / len(perceived_errs)) if perceived_errs else float("nan")
    u_claim = (sum(uncited_claims) / len(uncited_claims)) if uncited_claims else float("nan")
    c_cov = (sum(cit_coverages) / len(cit_coverages)) if cit_coverages else float("nan")
    r_cov = (sum(rule_coverages) / len(rule_coverages)) if rule_coverages else float("nan")
    j_conf = (sum(judge_confs) / len(judge_confs)) if judge_confs else float("nan")
    mean_j_cost = (sum(judge_costs) / len(judge_costs)) if judge_costs else 0.0
    tot_j_cost = sum(judge_costs)

    # Disaggregated step latencies aggregation
    step_latency_acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.fold_planning_latency_ms > 0:
            step_latency_acc["fold_planning"].append(r.fold_planning_latency_ms)
        if r.json_extraction_latency_ms > 0:
            step_latency_acc["json_extraction"].append(r.json_extraction_latency_ms)
        if r.synthesis_latency_ms > 0:
            step_latency_acc["synthesis"].append(r.synthesis_latency_ms)
        for s_k, s_v in getattr(r, "step_latencies", {}).items():
            if s_v > 0:
                step_latency_acc[s_k].append(s_v)

    mean_step_latencies = {
        s_k: sum(s_vals) / len(s_vals) for s_k, s_vals in step_latency_acc.items() if s_vals
    }

    return MetricsReport(
        scope=scope,
        n=n,
        asserted=len(asserted),
        abstained=len(abstained),
        raw_agreement_vs_truth=raw,
        cohen_kappa=kappa,
        krippendorff_alpha=alpha,
        false_confidence_rate=fcr,
        false_confidence_rate_over_asserted=fcr_asserted,
        abstention_rate=abst_rate,
        abstention_precision=abst_prec,
        abstention_recall=abst_recall,
        over_refusal_rate=over_refusal,
        abstention_aware_utility=util,
        verifier_agreement=verifier_agreement,
        mean_ocr_latency_ms=mean_ocr,
        mean_citation_latency_ms=mean_cit,
        mean_llm_latency_ms=mean_llm,
        mean_total_latency_ms=mean_tot,
        mean_step_latencies=mean_step_latencies,
        perceived_error_rate=p_err,
        uncited_claim_rate=u_claim,
        mean_judge_confidence=j_conf,
        mean_citation_coverage=c_cov,
        mean_rule_reference_coverage=r_cov,
        mean_judge_cost_usd=mean_j_cost,
        total_judge_cost_usd=tot_j_cost,
    )



def compute_metrics(records: list[EvalRecord]) -> MetricsReport:
    overall = _compute("overall", records)
    by_program: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_program[r.program.value].append(r)
    overall.per_program = {name: _compute(name, recs) for name, recs in sorted(by_program.items())}
    return overall


# --------------------------------------------------------------------------- #
# Skill-Lift Empirical Benchmarking Dataclasses & Math
# --------------------------------------------------------------------------- #


@dataclass
class SkillLiftRecord:
    """A paired evaluation run on identical case input: baseline (unskilled) vs. skilled."""

    case_id: str
    program: ProgramId
    ground_truth_label: str  # "eligible" | "ineligible"
    ambiguous: bool
    baseline_predicted_label: str | None = None
    skilled_predicted_label: str | None = None
    baseline_abstained: bool = False
    skilled_abstained: bool = False
    baseline_score: float = 0.0
    skilled_score: float = 0.0
    skill_lift: float = 0.0  # skilled_score - baseline_score
    baseline_citations_count: int = 0
    skilled_citations_count: int = 0
    statutory_citation_accuracy: float = 1.0
    reasoning_fidelity: float = 1.0
    under_appeal: bool = False


@dataclass
class ProgramSkillLift:
    """Skill lift breakdown for an individual statutory benefit program."""

    program: str
    n_cases: int
    baseline_accuracy: float
    skilled_accuracy: float
    accuracy_lift: float
    baseline_utility: float
    skilled_utility: float
    utility_lift: float
    baseline_fcr: float
    skilled_fcr: float
    fcr_reduction: float
    statutory_citation_accuracy: float
    appeal_reasoning_fidelity: float
    mean_skill_lift: float


@dataclass
class SkillLiftReport:
    """Comprehensive Empirical Skill-Lift Report comparing baseline vs skilled pipeline runs."""

    scope: str = "overall"
    n_total_cases: int = 0
    mean_skill_lift: float = 0.0
    accuracy_lift: float = 0.0
    utility_lift: float = 0.0
    citation_accuracy_lift: float = 0.0
    appeal_fidelity_lift: float = 0.0
    per_program: dict[str, ProgramSkillLift] = field(default_factory=dict)

    def render(self) -> str:
        def f(x: float) -> str:
            return "  n/a" if isnan(x) else f"{x:+6.3f}"

        def fp(x: float) -> str:
            return "  n/a" if isnan(x) else f"{x:6.3f}"

        lines = [
            f"=== TRIBUNE Empirical Skill-Lift Report — {self.scope} (n={self.n_total_cases}) ===",
            f"  NET EMPIRICAL SKILL LIFT        : {f(self.mean_skill_lift)}   <-- (Skilled - Baseline)",
            f"  Accuracy Lift (delta)           : {f(self.accuracy_lift)}",
            f"  Utility Lift (delta)            : {f(self.utility_lift)}",
            f"  Statutory Citation Accuracy Lift: {f(self.citation_accuracy_lift)}",
            f"  Appeal Reasoning Fidelity Lift  : {f(self.appeal_fidelity_lift)}",
            "  --- Per-Program Skill Lift Breakdown ---",
        ]
        for prog, lift in sorted(self.per_program.items()):
            lines.append(
                f"  [{prog:<12}] Lift: {f(lift.mean_skill_lift)} | "
                f"Acc: {fp(lift.baseline_accuracy)} -> {fp(lift.skilled_accuracy)} ({f(lift.accuracy_lift)}) | "
                f"CitAcc: {fp(lift.statutory_citation_accuracy)} | "
                f"Fidelity: {fp(lift.appeal_reasoning_fidelity)}"
            )
        return "\n".join(lines)


def _score_single_outcome(predicted: str | None, truth: str, abstained: bool, ambiguous: bool) -> float:
    if abstained:
        return _W_ABSTAIN_AMBIGUOUS if ambiguous else _W_ABSTAIN_CLEAR
    if predicted == truth:
        return _W_CORRECT
    return _W_WRONG


def compute_skill_lift(paired_records: list[SkillLiftRecord]) -> SkillLiftReport:
    """Compute empirical Skill Lift delta metrics across paired baseline and skilled evaluation runs."""
    if not paired_records:
        return SkillLiftReport()

    n_total = len(paired_records)
    by_program: dict[str, list[SkillLiftRecord]] = defaultdict(list)
    for r in paired_records:
        by_program[r.program.value].append(r)

    program_lifts: dict[str, ProgramSkillLift] = {}

    for prog_name, recs in by_program.items():
        n = len(recs)
        base_correct = sum(
            1 for r in recs if not r.baseline_abstained and r.baseline_predicted_label == r.ground_truth_label
        )
        base_asserted = sum(1 for r in recs if not r.baseline_abstained)
        base_acc = (base_correct / base_asserted) if base_asserted > 0 else 0.0

        skill_correct = sum(
            1 for r in recs if not r.skilled_abstained and r.skilled_predicted_label == r.ground_truth_label
        )
        skill_asserted = sum(1 for r in recs if not r.skilled_abstained)
        skill_acc = (skill_correct / skill_asserted) if skill_asserted > 0 else 0.0
        acc_lift = skill_acc - base_acc

        base_scores = [
            _score_single_outcome(r.baseline_predicted_label, r.ground_truth_label, r.baseline_abstained, r.ambiguous)
            for r in recs
        ]
        skill_scores = [
            _score_single_outcome(r.skilled_predicted_label, r.ground_truth_label, r.skilled_abstained, r.ambiguous)
            for r in recs
        ]
        base_util = sum(base_scores) / n
        skill_util = sum(skill_scores) / n
        util_lift = skill_util - base_util

        base_wrong = sum(
            1 for r in recs if not r.baseline_abstained and r.baseline_predicted_label != r.ground_truth_label
        )
        skill_wrong = sum(
            1 for r in recs if not r.skilled_abstained and r.skilled_predicted_label != r.ground_truth_label
        )
        base_fcr = base_wrong / n
        skill_fcr = skill_wrong / n
        fcr_red = base_fcr - skill_fcr

        cit_accs = [r.statutory_citation_accuracy for r in recs]
        mean_cit_acc = sum(cit_accs) / len(cit_accs) if cit_accs else 1.0

        fidelities = [r.reasoning_fidelity for r in recs]
        mean_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0

        lifts = [r.skill_lift for r in recs]
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0

        program_lifts[prog_name] = ProgramSkillLift(
            program=prog_name,
            n_cases=n,
            baseline_accuracy=round(base_acc, 4),
            skilled_accuracy=round(skill_acc, 4),
            accuracy_lift=round(acc_lift, 4),
            baseline_utility=round(base_util, 4),
            skilled_utility=round(skill_util, 4),
            utility_lift=round(util_lift, 4),
            baseline_fcr=round(base_fcr, 4),
            skilled_fcr=round(skill_fcr, 4),
            fcr_reduction=round(fcr_red, 4),
            statutory_citation_accuracy=round(mean_cit_acc, 4),
            appeal_reasoning_fidelity=round(mean_fidelity, 4),
            mean_skill_lift=round(mean_lift, 4),
        )

    all_lifts = [r.skill_lift for r in paired_records]
    overall_mean_lift = sum(all_lifts) / len(all_lifts) if all_lifts else 0.0
    overall_acc_lift = sum(p.accuracy_lift for p in program_lifts.values()) / max(1, len(program_lifts))
    overall_util_lift = sum(p.utility_lift for p in program_lifts.values()) / max(1, len(program_lifts))
    overall_cit_lift = sum(p.statutory_citation_accuracy for p in program_lifts.values()) / max(1, len(program_lifts))
    overall_fid_lift = sum(p.appeal_reasoning_fidelity for p in program_lifts.values()) / max(1, len(program_lifts))

    return SkillLiftReport(
        scope="overall",
        n_total_cases=n_total,
        mean_skill_lift=round(overall_mean_lift, 4),
        accuracy_lift=round(overall_acc_lift, 4),
        utility_lift=round(overall_util_lift, 4),
        citation_accuracy_lift=round(overall_cit_lift, 4),
        appeal_fidelity_lift=round(overall_fid_lift, 4),
        per_program=program_lifts,
    )
