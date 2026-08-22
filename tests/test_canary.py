"""The drift sentinel detects rule-citation drift; the canary passes on a fresh baseline."""

from tribune.config import reset_settings_cache
from tribune.eval.canary import CanarySentinel, detect_drift, freeze_fingerprints


def test_detect_drift_flags_changed_rule():
    base = freeze_fingerprints()
    current = {k: dict(v) for k, v in base.items()}
    key = next(iter(current))
    cid = next(iter(current[key]))
    current[key][cid] = "MUTATED-HASH"
    drift = detect_drift(base, current)
    assert any("CHANGED" in d for d in drift)


def test_detect_drift_none_when_identical():
    base = freeze_fingerprints()
    assert detect_drift(base, base) == []


def test_canary_passes_on_fresh_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIBUNE_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    try:
        report = CanarySentinel().run(freeze=True)
        assert report.ok
        assert not report.confidently_wrong
    finally:
        reset_settings_cache()


def test_canary_evaluator_token_isolation():
    sentinel = CanarySentinel()
    token = sentinel.generate_canary_token()
    assert token.startswith("CANARY-")

    safe_output = {"status": "likely_eligible", "rationale": "Valid reasoning"}
    assert sentinel.verify_canary_isolation(token, safe_output) is True

    leaked_output = {"status": "likely_eligible", "extended_thinking": f"Note {token}"}
    assert sentinel.verify_canary_isolation(token, leaked_output) is False


def test_canary_accuracy_fallback_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIBUNE_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    sentinel = CanarySentinel()

    # Accuracy at 98.4% (< 98.5% floor) -> Triggers fallback
    report_fallback = sentinel.run(freeze=True, simulated_citation_accuracy=0.984)
    assert report_fallback.fallback_triggered is True
    assert report_fallback.ok is False
    assert "98.400%" in report_fallback.render()
    assert report_fallback.fallback_model == "grok-4.6"

    # Accuracy at 99.0% (>= 98.5% floor) -> Passes without fallback
    report_pass = sentinel.run(freeze=True, simulated_citation_accuracy=0.990)
    assert report_pass.fallback_triggered is False
    assert report_pass.ok is True


