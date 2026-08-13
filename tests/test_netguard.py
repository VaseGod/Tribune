"""Phase 6 — sandboxed eval environment: network-egress guard.

Covers: deny-by-default blocks non-allowlisted connects, the allowlist permits
the configured endpoint, the guard cleanly uninstalls, allowlist derivation from
settings, and — the acceptance test — the appeals eval runs end-to-end with zero
network egress.
"""

import socket

import pytest

from tribune.config import get_settings
from tribune.eval import netguard
from tribune.eval.appeals_eval import run_appeals_eval


@pytest.fixture(autouse=True)
def _ensure_clean_guard():
    # Never leave a patched socket layer behind, even if a test raises.
    yield
    if netguard._active is not None:
        netguard.uninstall()


def test_deny_by_default_blocks_connect():
    with netguard.deny_egress(allowlist=set()):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(netguard.NetworkEgressBlocked, match="denied by the eval sandbox"):
                s.connect(("93.184.216.34", 80))  # example.com; never actually dialed
        finally:
            s.close()


def test_allowlist_permits_configured_endpoint():
    # An allowlisted (host, port) passes the guard check; a different one is denied.
    with netguard.deny_egress(allowlist={("gpu-host", 8001)}):
        netguard._check(("gpu-host", 8001))  # allowlisted -> no raise
        with pytest.raises(netguard.NetworkEgressBlocked):
            netguard._check(("gpu-host", 9999))  # same host, wrong port -> denied
        with pytest.raises(netguard.NetworkEgressBlocked):
            netguard._check(("other-host", 8001))  # wrong host -> denied


def test_guard_uninstalls_cleanly():
    original = socket.socket.connect
    with netguard.deny_egress(allowlist=set()):
        assert socket.socket.connect is not original
    assert socket.socket.connect is original
    assert netguard._active is None


def test_blocked_attempts_are_recorded():
    with netguard.deny_egress(allowlist=set()) as guard:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(netguard.NetworkEgressBlocked):
                s.connect(("10.0.0.1", 443))
        finally:
            s.close()
        assert "10.0.0.1:443" in guard.blocked


def test_allowlist_from_settings_offline_is_empty():
    settings = get_settings()  # defaults are fully local
    assert netguard.allowlist_from_settings(settings) == set()


def test_allowlist_from_settings_extracts_remote_endpoint():
    settings = get_settings().model_copy(
        update={"provider": "openai_compat", "openai_base_url": "http://gpu-host:8001/v1"}
    )
    assert netguard.allowlist_from_settings(settings) == {("gpu-host", 8001)}


def test_af_unix_is_not_treated_as_egress():
    # A non-(host, port) address must pass through untouched.
    with netguard.deny_egress(allowlist=set()):
        netguard._check("/tmp/some.sock")  # would raise if treated as egress


def test_inspect_outbound_payload_blocks_encrypted_reasoning_blocks():
    with netguard.deny_egress(allowlist=set()):
        dirty_payload = {"messages": [{"role": "user", "content": "hello"}], "encrypted_content": "secret_blob"}
        with pytest.raises(netguard.NetworkEgressBlocked, match="encrypted_content"):
            netguard.inspect_outbound_payload(dirty_payload)

        clean_payload = {"messages": [{"role": "user", "content": "hello"}]}
        netguard.inspect_outbound_payload(clean_payload)  # passes


# --- acceptance: the appeals eval touches no network ------------------------ #


def test_appeals_eval_runs_end_to_end_with_zero_egress():
    outcome = run_appeals_eval(n=8)
    assert outcome.blocked_egress == []
    assert outcome.allowlist == []  # fully offline
    # It actually ran the appeals program end to end.
    assert outcome.result.report.n == 8
    assert all(r.program.value == "appeals" for r in outcome.result.records)
    # Deterministic: a second run yields identical outcomes.
    again = run_appeals_eval(n=8)
    assert [r.abstained for r in again.result.records] == [
        r.abstained for r in outcome.result.records
    ]

