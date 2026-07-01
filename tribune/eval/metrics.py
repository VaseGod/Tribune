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

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isnan

from ..types import ProgramId

# Utility weights for abstention-aware scoring.
_W_CORRECT = 1.0
_W_WRONG = -2.0  # confidently wrong is the cardinal harm
_W_ABSTAIN_AMBIGUOUS = 1.0  # rewarded
_W_ABSTAIN_CLEAR = 0.5  # safe but a missed opportunity


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
    abstention_aware_utility: float
    verifier_agreement: float
    per_program: dict[str, MetricsReport] = field(default_factory=dict)

    def render(self) -> str:
        def f(x: float) -> str:
            return "  n/a" if isnan(x) else f"{x:6.3f}"

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
            f"  abstention-aware utility        : {f(self.abstention_aware_utility)}   (higher is better; max 1.0)",
            f"  TRIBUNE vs verifier agreement   : {f(self.verifier_agreement)}",
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

    util = abstention_aware_utility(records)

    matched = [
        r for r in asserted if r.asserted_status is not None and r.asserted_status == r.verifier_status
    ]
    verifier_agreement = (len(matched) / len(asserted)) if asserted else float("nan")

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
        abstention_aware_utility=util,
        verifier_agreement=verifier_agreement,
    )


def compute_metrics(records: list[EvalRecord]) -> MetricsReport:
    overall = _compute("overall", records)
    by_program: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_program[r.program.value].append(r)
    overall.per_program = {name: _compute(name, recs) for name, recs in sorted(by_program.items())}
    return overall
