"""Comprehensive Unit Tests for Sentinel Security Daemon, Allowlist, and Token Broker.

Validates:
- Allowlist default-deny behavior
- Allowed host authorization
- Disallowed host denial
- Host-side authentication header injection
- Automatic secret scrubbing / redaction from stdout and stderr
- Deterministic surrogate tokens replacing real credentials
- Unix domain socket IPC request/response behavior
"""

from __future__ import annotations

import os
import tempfile
import time
import pytest

from tribune.security.allowlist import AllowlistPolicy, AllowlistRule, default_allowlist
from tribune.security.audit import SecurityAuditLogger, SecurityEventType
from tribune.security.sentinel import CommandDecision, SentinelBroker, SentinelClient, SentinelServer
from tribune.security.token_broker import TokenBroker


def test_allowlist_default_deny():
    """Verify that destinations not explicitly listed in allowlist are rejected."""
    policy = AllowlistPolicy(rules=[], default_policy="deny")
    allowed, reason, rule = policy.check_egress("malicious-exfil.com", port=443, scheme="https")
    assert not allowed
    assert "denied by default policy" in reason
    assert rule is None


def test_allowlist_allowed_host_passes():
    """Verify allowlisted destination passes with correct rule match."""
    rule = AllowlistRule(
        id="rule_internal_mock",
        destination_host="mock.internal.tribune",
        port=443,
        scheme="https",
        methods=("GET", "POST"),
        auth_profile="mock_internal_api",
        max_requests_per_task=5,
    )
    policy = AllowlistPolicy(rules=[rule], default_policy="deny")

    # Allowed GET
    allowed, reason, matched = policy.check_egress("mock.internal.tribune", port=443, scheme="https", method="GET")
    assert allowed
    assert matched is not None and matched.id == "rule_internal_mock"

    # Disallowed Method (DELETE)
    allowed_del, reason_del, _ = policy.check_egress("mock.internal.tribune", port=443, scheme="https", method="DELETE")
    assert not allowed_del
    assert "Method DELETE not allowed" in reason_del


def test_allowlist_per_task_rate_limiting():
    """Verify task request counter rejects excess egress attempts."""
    rule = AllowlistRule(
        id="rate_limited_api",
        destination_host="api.rate-limited.com",
        port=443,
        scheme="https",
        methods=("GET",),
        max_requests_per_task=2,
    )
    policy = AllowlistPolicy(rules=[rule], default_policy="deny")

    task_id = "task_001"
    ok1, _, _ = policy.check_egress("api.rate-limited.com", task_id=task_id)
    ok2, _, _ = policy.check_egress("api.rate-limited.com", task_id=task_id)
    assert ok1 and ok2

    # 3rd request exceeds limit
    ok3, reason, _ = policy.check_egress("api.rate-limited.com", task_id=task_id)
    assert not ok3
    assert "exceeded max_requests_per_task" in reason


def test_token_broker_surrogate_mapping_and_redaction():
    """Verify TokenBroker maps real secrets to MOCK_* handles and redacts output."""
    broker = TokenBroker(surrogate_prefix="MOCK_")
    real_secret = "sk-live-supersecretliveproductionapikey123456"

    # Register secret
    surrogate = broker.register_secret(real_secret=real_secret, secret_ref="OPENAI_API_KEY", session_id="test_sess")
    assert surrogate.startswith("MOCK_")
    assert surrogate != real_secret

    # Verify resolution stays host-only
    binding = broker.resolve_surrogate(surrogate)
    assert binding is not None
    assert binding.real_secret == real_secret

    # Environment sanitization
    dirty_env = {
        "OPENAI_API_KEY": real_secret,
        "PUBLIC_VAR": "hello_world",
        "DATABASE_URL": "postgres://user:super_secret_db_pass@db.internal:5432/db",
    }
    clean_env, modified_keys = broker.surrogate_environment(dirty_env, session_id="test_sess")
    assert clean_env["OPENAI_API_KEY"] == surrogate
    assert real_secret not in clean_env["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" in modified_keys

    # Output redaction
    leaked_output = f"Error communicating with upstream server: invalid key {real_secret} provided."
    redacted = broker.redact_text(leaked_output)
    assert real_secret not in redacted
    assert surrogate in redacted


def test_sentinel_host_side_auth_injection():
    """Verify Sentinel injects real authorization headers in host space without exposing to sandbox."""
    allowlist = AllowlistPolicy(
        rules=[
            AllowlistRule(
                id="internal_secure_api",
                destination_host="secure.internal.tribune",
                port=443,
                scheme="https",
                methods=("GET",),
                auth_profile="bearer_host_secret_ref",
            )
        ],
        default_policy="deny",
    )
    real_vault_secret = "sk-prod-vault-token-99999"
    broker = SentinelBroker(
        allowlist=allowlist,
        host_vault={"secure.internal.tribune": real_vault_secret},
    )

    decision = broker.evaluate_egress(
        destination_host="secure.internal.tribune",
        port=443,
        scheme="https",
        method="GET",
        task_id="t1",
    )
    assert decision.allowed
    assert decision.auth_profile == "bearer_host_secret_ref"
    assert decision.injected_headers.get("Authorization") == f"Bearer {real_vault_secret}"


def test_sentinel_command_inspection():
    """Verify Sentinel command inspection blocks privilege escalation and sensitive paths."""
    broker = SentinelBroker()

    # Blocked: sudo
    dec_sudo = broker.evaluate_command("sudo cat /etc/hosts")
    assert not dec_sudo.allowed
    assert "Disallowed security pattern matched" in dec_sudo.reason

    # Blocked: shadow file
    dec_shadow = broker.evaluate_command("cat /etc/shadow")
    assert not dec_shadow.allowed

    # Blocked: docker socket
    dec_docker = broker.evaluate_command("curl --unix-socket /var/run/docker.sock http://localhost/containers/json")
    assert not dec_docker.allowed

    # Allowed safe command
    dec_safe = broker.evaluate_command("python3 calculate_benefit.py --input data.json")
    assert dec_safe.allowed


def test_sentinel_uds_socket_communication():
    """Verify SentinelServer and SentinelClient communicate cleanly over Unix domain socket."""
    socket_path = f"/tmp/test_sentinel_{int(time.time() * 1000)}.sock"
    allowlist = default_allowlist()
    broker = SentinelBroker(allowlist=allowlist)
    server = SentinelServer(broker=broker, socket_path=socket_path)

    try:
        server.start()
        # Give thread short spin-up time
        time.sleep(0.1)

        client = SentinelClient(socket_path=socket_path)
        dec_safe = client.check_command("echo 'test safe command'")
        assert dec_safe.allowed
        assert "echo 'test safe command'" in dec_safe.redacted_command

        dec_blocked = client.check_command("sudo rm -rf /")
        assert not dec_blocked.allowed
    finally:
        server.stop()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
