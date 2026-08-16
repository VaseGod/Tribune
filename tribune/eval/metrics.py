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
    # -- statutory citations & decisive criteria -- #
    citations: list[str] = field(default_factory=list)
    decisive_criteria: list[str] = field(default_factory=list)


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
            "  --- latency breakdown (disaggregated) ---",
            f"  OCR ingestion latency (mean)    : {f_ms(self.mean_ocr_latency_ms)}",
            f"  citation matching latency (mean): {f_ms(self.mean_citation_latency_ms)}",
            f"  pure LLM generation (mean)      : {f_ms(self.mean_llm_latency_ms)}",
            f"  total pipeline latency (mean)   : {f_ms(self.mean_total_latency_ms)}",
        ]
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
    )


def compute_metrics(records: list[EvalRecord]) -> MetricsReport:
    overall = _compute("overall", records)
    by_program: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_program[r.program.value].append(r)
    overall.per_program = {name: _compute(name, recs) for name, recs in sorted(by_program.items())}
    return overall
