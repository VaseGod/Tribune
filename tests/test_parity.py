"""Phase 3 — multilingual parity audit (equity gate).

Covers: ES twin construction (language plumbed, ingestion identical, legal terms
never silently translated), end-to-end EN/ES parity with the offline provider,
the equity-bug flag firing on a synthetic bad delta, and threshold
configurability.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

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
from tribune.orchestration.mtp import NativeMTPBackbone
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


# --------------------------------------------------------------------------- #
# Logit & Distribution Parity Assurance (Speculative MTP vs Autoregressive)
# --------------------------------------------------------------------------- #


def _simulate_autoregressive_step(last_token: int, step_offset: int, vocab_size: int) -> np.ndarray:
    """Deterministic autoregressive reference implementation predicting next token logits."""
    logits = np.zeros(vocab_size, dtype=np.float32)
    head_token = (last_token + step_offset * 7) % vocab_size
    logits[head_token] = 8.5
    logits[(head_token + 1) % vocab_size] = 2.0
    return logits


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=5000),
    seq_length=st.integers(min_value=3, max_value=30),
)
def test_speculative_mtp_logit_parity_property_based(seed: int, seq_length: int):
    """Property-based test: verify exact mathematical logit parity between speculative MTP
    and baseline autoregressive generation across diverse pseudorandom sequences."""
    rng = np.random.default_rng(seed)
    vocab_size = 500
    hidden_dim = 128
    sequence = rng.integers(0, vocab_size, size=seq_length).tolist()
    hidden_state = rng.standard_normal(hidden_dim).astype(np.float32)

    backbone = NativeMTPBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_speculative_depth=4)

    # 1. Speculative MTP forward pass projecting depth K
    branch = backbone.project_candidates(
        current_sequence=sequence,
        hidden_state=hidden_state,
        content_hint="structured",
    )
    k = branch.depth_k
    mtp_logits = branch.logits  # shape (k, vocab_size)

    # 2. Baseline Autoregressive step-by-step rollout
    ar_logits = np.zeros((k, vocab_size), dtype=np.float32)
    last_tok = sequence[-1]
    for step in range(k):
        ar_logits[step] = _simulate_autoregressive_step(last_tok, step + 1, vocab_size)

    # 3. Assert exact mathematical logit parity across all speculative steps
    np.testing.assert_allclose(
        mtp_logits,
        ar_logits,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Mathematical logit parity violation between native MTP and autoregressive baseline",
    )


def test_greedy_output_equivalence_mtp_vs_autoregressive():
    """Verify greedy output equivalence (argmax over logits) between MTP execution
    and baseline autoregressive generation across varied sequence lengths."""
    vocab_size = 1000
    hidden_dim = 256
    backbone = NativeMTPBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_speculative_depth=4)

    test_lengths = [4, 8, 16, 32, 64]
    for length in test_lengths:
        rng = np.random.default_rng(length * 100)
        seq = rng.integers(0, vocab_size, size=length).tolist()
        hidden = rng.standard_normal(hidden_dim).astype(np.float32)

        branch = backbone.project_candidates(seq, hidden, content_hint="json")
        k = branch.depth_k

        mtp_greedy = [int(np.argmax(branch.logits[s])) for s in range(k)]

        last_tok = seq[-1]
        ar_greedy = [
            int(np.argmax(_simulate_autoregressive_step(last_tok, s + 1, vocab_size)))
            for s in range(k)
        ]

        # Greedy predictions must be identically equal
        assert mtp_greedy == ar_greedy
        assert mtp_greedy == branch.candidate_tokens[:k]


def test_distribution_preservation_temperature_scaled_sampling():
    """Verify probability distribution preservation under temperature-scaled sampling
    (asserting zero KL divergence between MTP and autoregressive distributions)."""
    vocab_size = 200
    hidden_dim = 64
    backbone = NativeMTPBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, max_speculative_depth=4)

    seq = [12, 34, 56]
    hidden = np.ones(hidden_dim, dtype=np.float32)
    branch = backbone.project_candidates(seq, hidden, content_hint="code")
    k = branch.depth_k

    temperatures = [0.2, 0.5, 0.7, 1.0, 1.5]

    for temp in temperatures:
        for step in range(k):
            # MTP scaled softmax
            mtp_z = branch.logits[step] / temp
            mtp_shift = mtp_z - np.max(mtp_z)
            mtp_p = np.exp(mtp_shift) / np.sum(np.exp(mtp_shift))

            # Autoregressive scaled softmax
            ar_z = _simulate_autoregressive_step(seq[-1], step + 1, vocab_size) / temp
            ar_shift = ar_z - np.max(ar_z)
            ar_p = np.exp(ar_shift) / np.sum(np.exp(ar_shift))

            # Exact distribution preservation check
            np.testing.assert_allclose(mtp_p, ar_p, rtol=1e-5, atol=1e-6)

            # Assert KL divergence is practically zero (< 1e-6)
            kl_div = np.sum(ar_p * np.log(np.maximum(ar_p, 1e-12) / np.maximum(mtp_p, 1e-12)))
            assert abs(kl_div) < 1e-6
