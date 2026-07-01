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
