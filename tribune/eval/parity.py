"""Multilingual parity audit (equity gate).

Runs parallel EN/ES versions of the same seed cases through the full pipeline
and compares, per language (and per backend configuration): token counts,
effective cost, abstention rate, kappa vs. gold, over-refusal rate, and
false-confidence rate. Deltas beyond the configured thresholds are filed as
**equity bugs** — structured findings with a distinct label, meant to become
tracker issues — not footnotes.

A quality gap by language is a fairness defect in a benefits tool: if the
Spanish-language path abstains more, over-refuses more, or costs more per
completed verification, Spanish-speaking claimants get a worse service.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import isnan

from ..casegen import i18n
from ..config import TribuneSettings, get_settings
from ..orchestration.pipeline import CasePipeline
from ..types import SyntheticCase
from .costreport import CostReport, compute_cost_report
from .harness import records_for_case
from .metrics import EvalRecord, MetricsReport, compute_metrics
from .quant_sensitivity.seedset import SEED_WEIGHTS, build_seed_set

EQUITY_BUG_LABEL = "equity-bug"

_PACKAGED_THRESHOLDS = os.path.join(os.path.dirname(__file__), "parity_thresholds.json")


def load_thresholds(path: str | None = None) -> dict[str, float]:
    with open(path or _PACKAGED_THRESHOLDS, encoding="utf-8") as fh:
        payload = json.load(fh)
    return {k: float(v) for k, v in payload.items() if isinstance(v, int | float)}


@dataclass
class EquityBug:
    """A parity delta beyond threshold. File it in the tracker under the
    ``equity-bug`` label — these block Spanish-facing deployment claims."""

    metric: str
    value_en: float
    value_es: float
    delta: float
    threshold: float
    label: str = EQUITY_BUG_LABEL

    def title(self) -> str:
        return (
            f"[{self.label}] EN/ES parity breach on {self.metric}: "
            f"en={self.value_en:.4f} es={self.value_es:.4f} "
            f"(delta {self.delta:.4f} > threshold {self.threshold:.4f})"
        )


@dataclass
class LanguageSlice:
    language: str
    records: list[EvalRecord]
    report: MetricsReport
    cost_report: CostReport


@dataclass
class ParityReport:
    en: LanguageSlice
    es: LanguageSlice
    thresholds: dict[str, float]
    bugs: list[EquityBug] = field(default_factory=list)
    unreviewed_legal_terms: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bugs

    def render(self) -> str:
        def f(x: float, places: int = 4) -> str:
            return "n/a" if isnan(x) else f"{x:.{places}f}"

        def mean_tokens(s: LanguageSlice) -> float:
            n = max(1, s.cost_report.n)
            return (s.cost_report.total_tokens_input + s.cost_report.total_tokens_output) / n

        rows = [
            "| metric | EN | ES |",
            "|---|---|---|",
            f"| n tasks | {self.en.report.n} | {self.es.report.n} |",
            f"| kappa vs gold | {f(self.en.report.cohen_kappa)} | {f(self.es.report.cohen_kappa)} |",
            f"| abstention rate | {f(self.en.report.abstention_rate)} | {f(self.es.report.abstention_rate)} |",
            f"| over-refusal rate | {f(self.en.report.over_refusal_rate)} | {f(self.es.report.over_refusal_rate)} |",
            f"| false-confidence rate | {f(self.en.report.false_confidence_rate)} | {f(self.es.report.false_confidence_rate)} |",
            f"| mean cost/task (USD) | {f(self.en.cost_report.mean_cost_usd, 5)} | {f(self.es.cost_report.mean_cost_usd, 5)} |",
            f"| mean tokens/task | {mean_tokens(self.en):.1f} | {mean_tokens(self.es):.1f} |",
        ]
        parts = [
            "# TRIBUNE multilingual parity audit (EN/ES)",
            "",
            "## Side-by-side",
            "",
            "\n".join(rows),
            "",
            "## Equity gate",
            "",
        ]
        if self.bugs:
            parts.append(
                f"**{len(self.bugs)} equity bug(s)** — file each under the "
                f"`{EQUITY_BUG_LABEL}` tracker label; these are defects, not footnotes:"
            )
            parts.extend(f"- {b.title()}" for b in self.bugs)
        else:
            parts.append("No parity deltas beyond thresholds.")
        parts += [
            "",
            "## Translation review status",
            "",
        ]
        if self.unreviewed_legal_terms:
            parts.append(
                "⚠️ **The following legal/eligibility terms are machine-drafted and "
                "NOT yet human-reviewed.** A bilingual navigator, benefits counselor, "
                "or legal-aid staffer must review them before any Spanish-facing "
                "deployment:"
            )
            parts.extend(f"- `{t}`" for t in self.unreviewed_legal_terms)
        else:
            parts.append("All legal/eligibility glossary terms are human-reviewed.")
        parts += [
            "",
            f"Thresholds: `{json.dumps(self.thresholds, sort_keys=True)}`",
            "",
        ]
        return "\n".join(parts)


def _abs_delta_bug(metric: str, en: float, es: float, threshold: float) -> EquityBug | None:
    if isnan(en) or isnan(es):
        return None
    delta = abs(en - es)
    if delta > threshold:
        return EquityBug(metric, en, es, delta, threshold)
    return None


def _rel_delta_bug(metric: str, en: float, es: float, threshold: float) -> EquityBug | None:
    if isnan(en) or isnan(es):
        return None
    base = max(abs(en), 1e-12)
    delta = abs(en - es) / base
    if delta > threshold:
        return EquityBug(metric, en, es, delta, threshold)
    return None


def compare_language_slices(
    en: LanguageSlice, es: LanguageSlice, thresholds: dict[str, float]
) -> list[EquityBug]:
    def mean_tokens(s: LanguageSlice) -> float:
        n = max(1, s.cost_report.n)
        return (s.cost_report.total_tokens_input + s.cost_report.total_tokens_output) / n

    candidates = [
        _abs_delta_bug(
            "abstention_rate",
            en.report.abstention_rate,
            es.report.abstention_rate,
            thresholds["max_abstention_rate_delta"],
        ),
        _abs_delta_bug(
            "over_refusal_rate",
            en.report.over_refusal_rate,
            es.report.over_refusal_rate,
            thresholds["max_over_refusal_rate_delta"],
        ),
        _abs_delta_bug(
            "cohen_kappa",
            en.report.cohen_kappa,
            es.report.cohen_kappa,
            thresholds["max_kappa_delta"],
        ),
        _abs_delta_bug(
            "false_confidence_rate",
            en.report.false_confidence_rate,
            es.report.false_confidence_rate,
            thresholds["max_false_confidence_rate_delta"],
        ),
        _rel_delta_bug(
            "mean_cost_usd",
            en.cost_report.mean_cost_usd,
            es.cost_report.mean_cost_usd,
            thresholds["max_cost_rel_delta"],
        ),
        _rel_delta_bug(
            "mean_tokens_per_task",
            mean_tokens(en),
            mean_tokens(es),
            thresholds["max_tokens_rel_delta"],
        ),
    ]
    return [c for c in candidates if c is not None]


def _run_language(cases: list[SyntheticCase], settings: TribuneSettings) -> LanguageSlice:
    language = cases[0].language if cases else "en"
    pipeline = CasePipeline(settings)
    records: list[EvalRecord] = []
    for case in cases:
        records.extend(records_for_case(case, pipeline.run_case(case)))
    return LanguageSlice(
        language=language,
        records=records,
        report=compute_metrics(records),
        cost_report=compute_cost_report(records, scope=language),
    )


def run_parity(
    settings: TribuneSettings | None = None,
    cases: list[SyntheticCase] | None = None,
    thresholds: dict[str, float] | None = None,
) -> ParityReport:
    settings = settings or get_settings()
    if thresholds is None:
        thresholds = load_thresholds(settings.parity_thresholds_path or None)
    if cases is None:
        cases = build_seed_set(SEED_WEIGHTS)
    en_slice = _run_language(cases, settings)
    es_cases = [i18n.translate_case(c, "es") for c in cases]
    es_slice = _run_language(es_cases, settings)
    bugs = compare_language_slices(en_slice, es_slice, thresholds)
    return ParityReport(
        en=en_slice,
        es=es_slice,
        thresholds=thresholds,
        bugs=bugs,
        unreviewed_legal_terms=[t.en for t in i18n.unreviewed_legal_terms("es")],
    )
