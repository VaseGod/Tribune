"""Cost & turn-efficiency reporting.

Aggregates the per-task usage fields on :class:`~tribune.eval.metrics.EvalRecord`
into a summary table broken down by :class:`~tribune.eval.metrics.TaskOutcomeType`:
mean/median cost per task, turns, and token totals, plus per-language token counts.

Accounting rules (these enforce the calibrated-abstention invariant):

* every task — including every abstention — is a *completed* task,
* a correct abstention is reported at its actual (typically low) cost; it is never
  scored as a failure and never assigned an infinite or penalty cost,
* cost-per-completed-verification is therefore a simple mean/median over completed
  tasks, with the outcome-type breakdown making the trade-offs legible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isnan
from statistics import mean, median

from .metrics import EvalRecord, TaskOutcomeType, classify_outcome

#: Outcome types counted as *successes* for cost-per-successful-outcome. Correct
#: abstention is a success by design; do not remove it from this set.
SUCCESS_OUTCOMES = frozenset(
    {TaskOutcomeType.CORRECT_DETERMINATION, TaskOutcomeType.CORRECT_ABSTENTION}
)


@dataclass
class CostBucket:
    n: int
    mean_cost_usd: float
    median_cost_usd: float
    total_cost_usd: float
    mean_turns: float
    tokens_input: int
    tokens_output: int


@dataclass
class CostReport:
    scope: str
    n: int
    priced: int  # records that carried a cost figure
    buckets: dict[str, CostBucket] = field(default_factory=dict)  # by outcome type
    tokens_by_language: dict[str, dict[str, int]] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    mean_cost_usd: float = float("nan")
    median_cost_usd: float = float("nan")
    mean_cost_per_success_usd: float = float("nan")
    total_turns: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    estimated_tokens_present: bool = False

    def render(self) -> str:
        def usd(x: float) -> str:
            return "     n/a" if isnan(x) else f"{x:8.5f}"

        lines = [
            f"=== TRIBUNE cost & turn efficiency — {self.scope} (n={self.n}, priced={self.priced}) ===",
            f"  total cost (USD)                : {self.total_cost_usd:.5f}",
            f"  mean / median cost per task     : {usd(self.mean_cost_usd)} / {usd(self.median_cost_usd)}",
            f"  mean cost per successful outcome: {usd(self.mean_cost_per_success_usd)}   (correct determinations AND correct abstentions)",
            f"  total turns / tokens in / out   : {self.total_turns} / {self.total_tokens_input} / {self.total_tokens_output}",
            "  --- by outcome type (every abstention is a completed task) ---",
            "  outcome type             n   mean$      median$    turns  tok_in    tok_out",
        ]
        for name in [t.value for t in TaskOutcomeType]:
            b = self.buckets.get(name)
            if b is None:
                continue
            lines.append(
                f"  {name:22s} {b.n:4d}  {usd(b.mean_cost_usd)}  {usd(b.median_cost_usd)}"
                f"  {b.mean_turns:5.2f}  {b.tokens_input:8d}  {b.tokens_output:8d}"
            )
        if self.tokens_by_language:
            lines.append("  --- tokens by case language ---")
            for lang in sorted(self.tokens_by_language):
                t = self.tokens_by_language[lang]
                lines.append(
                    f"  [{lang}] tasks={t['n']}  tokens_input={t['tokens_input']}  "
                    f"tokens_output={t['tokens_output']}"
                )
        if self.estimated_tokens_present:
            lines.append(
                "  note: some token counts are deterministic estimates (offline provider)."
            )
        return "\n".join(lines)


def compute_cost_report(records: list[EvalRecord], scope: str = "overall") -> CostReport:
    n = len(records)
    priced = [r for r in records if r.cost_usd is not None]
    costs = [float(r.cost_usd) for r in priced]  # type: ignore[arg-type]

    by_outcome: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_outcome[classify_outcome(r).value].append(r)

    buckets: dict[str, CostBucket] = {}
    for name, recs in by_outcome.items():
        bucket_costs = [float(r.cost_usd) for r in recs if r.cost_usd is not None]
        buckets[name] = CostBucket(
            n=len(recs),
            mean_cost_usd=mean(bucket_costs) if bucket_costs else float("nan"),
            median_cost_usd=median(bucket_costs) if bucket_costs else float("nan"),
            total_cost_usd=sum(bucket_costs),
            mean_turns=mean([r.turns for r in recs]) if recs else float("nan"),
            tokens_input=sum(r.tokens_input for r in recs),
            tokens_output=sum(r.tokens_output for r in recs),
        )

    by_lang: dict[str, dict[str, int]] = {}
    for r in records:
        t = by_lang.setdefault(r.language, {"n": 0, "tokens_input": 0, "tokens_output": 0})
        t["n"] += 1
        t["tokens_input"] += r.tokens_input
        t["tokens_output"] += r.tokens_output

    successes = [
        float(r.cost_usd)
        for r in priced
        if classify_outcome(r) in SUCCESS_OUTCOMES
    ]

    return CostReport(
        scope=scope,
        n=n,
        priced=len(priced),
        buckets=buckets,
        tokens_by_language=by_lang,
        total_cost_usd=sum(costs),
        mean_cost_usd=mean(costs) if costs else float("nan"),
        median_cost_usd=median(costs) if costs else float("nan"),
        mean_cost_per_success_usd=mean(successes) if successes else float("nan"),
        total_turns=sum(r.turns for r in records),
        total_tokens_input=sum(r.tokens_input for r in records),
        total_tokens_output=sum(r.tokens_output for r in records),
        estimated_tokens_present=any("ws-estimator" in t for r in records for t in r.tokenizer_ids),
    )
