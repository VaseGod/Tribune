"""Phase 3 — multilingual parity audit (equity gate).

Covers: ES twin construction (language plumbed, ingestion identical, legal terms
never silently translated), end-to-end EN/ES parity with the offline provider,
the equity-bug flag firing on a synthetic bad delta, and threshold
configurability.
"""

from tribune.casegen import i18n
from tribune.casegen.i18n import es as es_glossary
from tribune.eval.costreport import compute_cost_report
from tribune.eval.metrics import EvalRecord, compute_metrics
from tribune.eval.parity import (
    EQUITY_BUG_LABEL,
    LanguageSlice,
    ParityReport,
    compare_language_slices,
    load_thresholds,
    run_parity,
)
from tribune.eval.quant_sensitivity.seedset import SEED_WEIGHTS, build_seed_set
from tribune.types import ProgramId

# --------------------------------------------------------------------------- #
# Translation scaffolding
# --------------------------------------------------------------------------- #


def _small_seed():
    return build_seed_set({p: 1 for p in SEED_WEIGHTS})


def test_translate_case_builds_a_faithful_spanish_twin():
    case = _small_seed()[0]
    twin = i18n.translate_case(case, "es")
    assert twin.language == "es" and case.language == "en"
    # Structured fields keep canonical keys -> ingestion and ground truth identical.
    assert [d.fields for d in twin.documents] == [d.fields for d in case.documents]
    assert twin.ground_truth == case.ground_truth
    # The rendered text is actually Spanish (glossary labels).
    text = twin.documents[0].text
    assert "ingreso mensual" in text or "tamaño del hogar" in text


def test_no_silent_machine_translation_of_legal_terms():
    # Every legal/eligibility term starts unreviewed and is surfaced as such.
    unreviewed = {t.en for t in es_glossary.unreviewed_legal_terms()}
    assert "monthly_income" in unreviewed and "citizenship_status" in unreviewed
    report = run_parity(cases=_small_seed())
    assert set(report.unreviewed_legal_terms) == unreviewed


def test_unknown_language_is_an_error_not_a_silent_fallback():
    case = _small_seed()[0]
    try:
        i18n.translate_case(case, "xx")
        raise AssertionError("expected ValueError for unknown language")
    except ValueError as exc:
        assert "glossary" in str(exc)


# --------------------------------------------------------------------------- #
# End-to-end parity with the offline provider
# --------------------------------------------------------------------------- #


def test_offline_pipeline_is_language_neutral():
    """The deterministic rules engine must treat EN and ES twins identically —
    a language-driven outcome difference offline would be a pipeline bug."""
    report = run_parity(cases=_small_seed())
    assert report.ok, [b.title() for b in report.bugs]
    assert report.en.report.abstention_rate == report.es.report.abstention_rate
    assert report.en.report.n == report.es.report.n
    rendered = report.render()
    assert "| metric | EN | ES |" in rendered
    assert "NOT yet human-reviewed" in rendered


# --------------------------------------------------------------------------- #
# The equity gate fires on bad deltas
# --------------------------------------------------------------------------- #


def _slice(language: str, n_abstained: int, n_total: int = 10) -> LanguageSlice:
    records = []
    for i in range(n_total):
        abstained = i < n_abstained
        records.append(
            EvalRecord(
                case_id=f"c{i}",
                program=ProgramId.SNAP,
                abstained=abstained,
                ground_truth_label="eligible",
                ambiguous=False,
                predicted_label=None if abstained else "eligible",
                tokens_input=100,
                tokens_output=10,
                cost_usd=0.01,
                turns=2,
                language=language,
            )
        )
    return LanguageSlice(
        language=language,
        records=records,
        report=compute_metrics(records),
        cost_report=compute_cost_report(records, scope=language),
    )


def test_equity_bug_fires_on_synthetic_bad_abstention_delta():
    thresholds = load_thresholds()
    en = _slice("en", n_abstained=0)
    es = _slice("es", n_abstained=4)  # 40% abstention gap: far past 5% threshold
    bugs = compare_language_slices(en, es, thresholds)
    metrics_flagged = {b.metric for b in bugs}
    assert "abstention_rate" in metrics_flagged and "over_refusal_rate" in metrics_flagged
    bug = next(b for b in bugs if b.metric == "abstention_rate")
    assert bug.label == EQUITY_BUG_LABEL
    assert "equity-bug" in bug.title() and "parity breach" in bug.title()
    # The report renders them as filed bugs and reports not-ok.
    report = ParityReport(en=en, es=es, thresholds=thresholds, bugs=bugs)
    assert not report.ok
    assert "equity bug(s)" in report.render()


def test_thresholds_are_configurable():
    en = _slice("en", n_abstained=0)
    es = _slice("es", n_abstained=1)  # 10% gap
    tight = dict(load_thresholds(), max_abstention_rate_delta=0.05)
    loose = dict(load_thresholds(), max_abstention_rate_delta=0.20)
    assert any(b.metric == "abstention_rate" for b in compare_language_slices(en, es, tight))
    assert not any(b.metric == "abstention_rate" for b in compare_language_slices(en, es, loose))
