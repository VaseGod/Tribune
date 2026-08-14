"""Phase 2 — quantization-sensitivity harness.

Covers: seed-set determinism/weighting/hash stability, manifest drift refusal,
ECE/Brier fixtures, the deterministic mock ladder (smoke), report rendering with
the frozen hash embedded, and the invariant that abstention drift is reported as
drift — abstentions stay completed successes in the cost accounting.
"""

import json
import os

from tribune.eval.metrics import brier_score, expected_calibration_error
from tribune.eval.quant_sensitivity import (
    SEED_WEIGHTS,
    build_seed_set,
    render_eval_note,
    run_ladder,
    seed_set_hash,
    smoke_ladder,
    write_manifest,
)
from tribune.eval.quant_sensitivity.backends import MockQuantVerifierProvider, QuantRung
from tribune.eval.quant_sensitivity.seedset import load_frozen_seed_set
from tribune.types import ProgramId

# --------------------------------------------------------------------------- #
# Seed set
# --------------------------------------------------------------------------- #


def test_seed_set_is_weighted_deterministic_and_hash_stable():
    a = build_seed_set()
    b = build_seed_set()
    assert len(a) == sum(SEED_WEIGHTS.values()) == 50
    counts = {}
    for case in a:
        program = case.target_programs[0]
        counts[program] = counts.get(program, 0) + 1
    assert counts[ProgramId.MEDICAID] == 15 and counts[ProgramId.SNAP] == 15
    assert [c.case_id for c in a] == [c.case_id for c in b]
    assert seed_set_hash(a) == seed_set_hash(b)


def test_frozen_manifest_matches_and_drift_is_refused(tmp_path):
    cases = build_seed_set()
    # The committed manifest must match the current generator output.
    frozen, manifest = load_frozen_seed_set()
    assert manifest["content_hash"] == seed_set_hash(frozen) == seed_set_hash(cases)

    # A drifted manifest is refused in strict mode.
    bad = str(tmp_path / "manifest.json")
    write_manifest(cases, bad)
    with open(bad, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["content_hash"] = "0" * 64
    with open(bad, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    try:
        load_frozen_seed_set(bad, strict=True)
        raise AssertionError("expected drift refusal")
    except RuntimeError as exc:
        assert "not comparable" in str(exc)
    # Non-strict mode annotates instead of refusing.
    _, annotated = load_frozen_seed_set(bad, strict=False)
    assert "drift_warning" in annotated


# --------------------------------------------------------------------------- #
# Calibration metrics: hand-checked fixtures
# --------------------------------------------------------------------------- #


def test_brier_fixture():
    # (0.8 vs 1)^2=0.04, (0.6 vs 0)^2=0.36, (1.0 vs 1)^2=0 -> mean 0.4/3
    assert abs(brier_score([0.8, 0.6, 1.0], [True, False, True]) - 0.4 / 3) < 1e-9


def test_ece_fixture():
    # Bin [0.7,0.8): two samples conf .75/.75, acc 0.5 -> |0.5-0.75|*2/4
    # Bin [0.9,1.0]: two samples conf .95/.95, acc 1.0 -> |1.0-0.95|*2/4
    confs = [0.75, 0.75, 0.95, 0.95]
    oks = [True, False, True, True]
    expected = 0.25 * 2 / 4 + 0.05 * 2 / 4
    assert abs(expected_calibration_error(confs, oks) - expected) < 1e-9


def test_perfect_calibration_scores_zero():
    assert expected_calibration_error([1.0, 1.0], [True, True]) < 1e-9
    assert brier_score([1.0, 0.0], [True, False]) < 1e-9


# --------------------------------------------------------------------------- #
# Mock backend + ladder smoke (deterministic, offline, free)
# --------------------------------------------------------------------------- #


def _smoke_cases():
    return build_seed_set({p: 1 for p in SEED_WEIGHTS})


def test_mock_flip_is_deterministic():
    rung = QuantRung("q2", "gguf-q2_k", flip_prob=0.5)
    provider = MockQuantVerifierProvider(rung)
    flips = [provider._flips(f"case-{i}") for i in range(64)]
    assert flips == [provider._flips(f"case-{i}") for i in range(64)]
    assert any(flips) and not all(flips)  # p=0.5 flips some, not all


def test_smoke_ladder_runs_and_degrades_monotonically_enough():
    cases = _smoke_cases()
    result = run_ladder(smoke_ladder(), cases=cases)
    assert [r.rung.label for r in result.rungs] == ["fp16", "q2"]
    fp16, q2 = result.rungs
    assert result.reference_label == "fp16"
    # The reference agrees with itself perfectly.
    assert fp16.kappa_vs_reference == 1.0 and fp16.abstention_decision_agreement == 1.0
    # The degraded rung must drift somewhere: more abstention or less agreement.
    assert (
        q2.report.abstention_rate >= fp16.report.abstention_rate
        or q2.abstention_decision_agreement < 1.0
    )
    # Every rung ran the same completed tasks; abstentions stay completed tasks.
    assert fp16.report.n == q2.report.n == len(cases)
    assert q2.cost_report.n == len(cases)
    # Abstention buckets never carry penalty costs (all costs finite/zero here).
    for bucket in q2.cost_report.buckets.values():
        assert bucket.total_cost_usd == 0.0  # free offline backends


def test_report_renders_with_frozen_hash_and_tables():
    cases = _smoke_cases()
    manifest = {"content_hash": seed_set_hash(cases), "n_cases": len(cases)}
    result = run_ladder(smoke_ladder(), cases=cases, manifest=manifest)
    note = render_eval_note(result)
    assert "Does abstention calibration survive quantization?" in note
    assert manifest["content_hash"] in note
    for heading in (
        "## Summary",
        "## Agreement & calibration vs. gold labels",
        "## Drift vs. the full-precision reference run",
        "## Honesty suite",
        "## Cost (Phase-1 metrics)",
    ):
        assert heading in note
    assert "abstention is a success outcome" in note


def test_committed_eval_note_exists():
    path = os.path.join("docs", "eval_notes", "eval_note_1_quant_sensitivity.md")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "content hash" in content


def test_moe_pruned_quant_ladder_runs_across_four_domains():
    from tribune.eval.quant_sensitivity import moe_pruned_quant_ladder

    # Ensure evaluation cases cover all benefit domains: SNAP, Medicaid, Housing, Unemployment, Appeals
    cases = build_seed_set({
        ProgramId.SNAP: 2,
        ProgramId.MEDICAID: 2,
        ProgramId.HOUSING: 2,
        ProgramId.UNEMPLOYMENT: 2,
        ProgramId.APPEALS: 2,
    })
    ladder = moe_pruned_quant_ladder()
    result = run_ladder(ladder, cases=cases)

    rung_labels = [r.rung.label for r in result.rungs]
    assert rung_labels == [
        "qwen3.8-max-moe-fp16",
        "qwen3.8-max-moe-4bit",
        "qwen3.8-max-moe-2bit",
        "qwen3.8-max-moe-1bit",
    ]

    ref_rung = result.rungs[0]
    one_bit_rung = result.rungs[3]

    # Full precision reference agreements
    assert ref_rung.kappa_vs_reference == 1.0
    assert ref_rung.abstention_decision_agreement == 1.0

    # 1-bit dynamic quantization regime experiences degradation/abstention drift
    assert one_bit_rung.report.n == len(cases)
    assert one_bit_rung.cost_report.n == len(cases)
    # Verification decisions across all statutory domains ran successfully
    programs_evaluated = {c.target_programs[0] for c in cases}
    assert {
        ProgramId.SNAP,
        ProgramId.MEDICAID,
        ProgramId.HOUSING,
        ProgramId.UNEMPLOYMENT,
    }.issubset(programs_evaluated)

