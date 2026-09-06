"""Unit & integration tests for Phase 2: Out-of-Band Governance Hardening Against Deceptive Optimization."""

from __future__ import annotations

import os
import time

import pytest

from tribune.governance.action_gate import SecurityViolationError
from tribune.governance.gates import (
    DecoupledActionGateService,
    GatekeeperSupervisor,
    PolicyRejectionError,
)
from tribune.instrumentation import tracing
from tribune.security.sandbox import (
    ContainmentTerminationError,
    HardenedExecutionSandbox,
    HeartbeatWatchdog,
)
from tribune.types import ActionProposal


def test_supervisor_rejects_adversarial_proposals():
    """Verify GatekeeperSupervisor rejects proposals attempting specification gaming or restricted resource access."""
    supervisor = GatekeeperSupervisor()

    # 1. Prohibited resource in proposal
    bad_proposal = ActionProposal(
        proposal_id="prop_bad_1",
        case_id="case_001",
        agent_id="agent_adversary",
        action_type="file_read",
        target_resource="/etc/passwd",
        parameters={"path": "/etc/passwd"},
    )
    with pytest.raises(PolicyRejectionError) as exc:
        supervisor.submit_proposal(bad_proposal)
    assert "POL_RESTRICTED_RESOURCES" in str(exc.value)

    # 2. Specification gaming pattern
    gaming_proposal = ActionProposal(
        proposal_id="prop_bad_2",
        case_id="case_001",
        agent_id="agent_adversary",
        action_type="formal_filing",
        target_resource="benefits_db",
        parameters={"case_id": "c1", "program": "snap"},
        rationale="Let's override_rule to force eligibility",
    )
    with pytest.raises(PolicyRejectionError) as exc2:
        supervisor.submit_proposal(gaming_proposal)
    assert "POL_NO_SPEC_GAMING" in str(exc2.value)


def test_supervisor_mints_and_verifies_hmac_token():
    """Verify GatekeeperSupervisor mints valid HMAC-SHA256 tokens and verifies TTL and binding."""
    supervisor = GatekeeperSupervisor(default_ttl_seconds=2)

    valid_proposal = ActionProposal(
        proposal_id="prop_valid_1",
        case_id="case_100",
        agent_id="navigator",
        action_type="formal_filing",
        target_resource="state_portal_v1",
        parameters={"case_id": "case_100", "program": "snap"},
        rationale="Verified gross income satisfies 7 CFR 273.9",
    )

    token = supervisor.submit_proposal(valid_proposal)
    assert token.proposal_id == "prop_valid_1"
    assert token.action_type == "formal_filing"
    assert len(token.nonce) == 32
    assert len(token.signature) == 64  # SHA-256 hex digest length

    # Immediate verification succeeds
    assert supervisor.verify_token(
        token=token,
        proposal_id="prop_valid_1",
        action_type="formal_filing",
        target_resource="state_portal_v1",
    )

    # Verification fails on mismatched action or resource
    assert not supervisor.verify_token(
        token=token,
        proposal_id="prop_valid_1",
        action_type="other_action",
        target_resource="state_portal_v1",
    )
    assert not supervisor.verify_token(
        token=token,
        proposal_id="prop_valid_1",
        action_type="formal_filing",
        target_resource="tampered_portal",
    )

    # Expired token is rejected
    time.sleep(2.1)
    assert not supervisor.verify_token(
        token=token,
        proposal_id="prop_valid_1",
        action_type="formal_filing",
        target_resource="state_portal_v1",
    )


def test_sandbox_enforces_hmac_token_authorization():
    """Verify HardenedExecutionSandbox requires valid supervisor HMAC token before executing high-stakes tools."""
    supervisor = GatekeeperSupervisor(default_ttl_seconds=60)
    sandbox = HardenedExecutionSandbox(supervisor=supervisor, enable_watchdog=False)

    executed = False

    def dummy_filing_tool(case_id: str, payload: str) -> dict[str, str]:
        nonlocal executed
        executed = True
        return {"status": "submitted", "case_id": case_id}

    # 1. Reject execution when token is missing
    with pytest.raises(SecurityViolationError) as exc_missing:
        sandbox.execute_high_stakes_tool(
            tool_name="submit_filing",
            tool_fn=dummy_filing_tool,
            kwargs={"case_id": "c1", "payload": "doc"},
            authorization_token=None,
            proposal_id="p1",
            target_resource="portal",
        )
    assert "Missing HMAC authorization token" in str(exc_missing.value)
    assert not executed

    # 2. Mint token out-of-band and execute
    service = DecoupledActionGateService(supervisor=supervisor)
    proposal = ActionProposal(
        proposal_id="prop_legal_1",
        case_id="c1",
        agent_id="preparer",
        action_type="formal_filing",
        target_resource="portal",
        parameters={"case_id": "c1", "program": "medicaid"},
    )
    token = service.submit_action_proposal(proposal, sandbox=sandbox)

    res = sandbox.execute_high_stakes_tool(
        tool_name="submit_filing",
        tool_fn=dummy_filing_tool,
        kwargs={"case_id": "c1", "payload": "doc"},
        authorization_token=token,
        proposal_id="prop_legal_1",
        target_resource="portal",
        action_type="formal_filing",
    )
    assert res["status"] == "submitted"
    assert executed is True


def test_sandbox_host_defenses_mask_proc_and_sys():
    """Verify sandbox unconditionally blocks access to /proc and /sys virtual filesystems."""
    supervisor = GatekeeperSupervisor()
    sandbox = HardenedExecutionSandbox(supervisor=supervisor, enable_watchdog=False)

    proposal = ActionProposal(
        proposal_id="prop_fs_1",
        case_id="c1",
        agent_id="agent",
        action_type="formal_filing",
        target_resource="data_store",
        parameters={"case_id": "c1", "program": "snap"},
    )
    token = supervisor.submit_proposal(proposal)

    def dummy_file_reader(path: str) -> str:
        return f"read:{path}"

    # Test /proc access blocked
    with pytest.raises(SecurityViolationError) as exc_proc:
        sandbox.execute_high_stakes_tool(
            tool_name="file_reader",
            tool_fn=dummy_file_reader,
            kwargs={"path": "/proc/cpuinfo"},
            authorization_token=token,
            proposal_id="prop_fs_1",
            target_resource="data_store",
        )
    assert "strictly masked" in str(exc_proc.value)

    # Test /sys access blocked
    with pytest.raises(SecurityViolationError) as exc_sys:
        sandbox.execute_high_stakes_tool(
            tool_name="file_reader",
            tool_fn=dummy_file_reader,
            kwargs={"path": "/sys/class/net"},
            authorization_token=token,
            proposal_id="prop_fs_1",
            target_resource="data_store",
        )
    assert "strictly masked" in str(exc_sys.value)


def test_sandbox_host_defenses_block_process_enumeration():
    """Verify sandbox blocks process enumeration utilities (ps, top, pgrep, etc.)."""
    supervisor = GatekeeperSupervisor()
    sandbox = HardenedExecutionSandbox(supervisor=supervisor, enable_watchdog=False)

    proposal = ActionProposal(
        proposal_id="prop_proc_1",
        case_id="c1",
        agent_id="agent",
        action_type="formal_filing",
        target_resource="data_store",
        parameters={"case_id": "c1", "program": "snap"},
    )
    token = supervisor.submit_proposal(proposal)

    def dummy_exec(cmd: str) -> str:
        return f"ran:{cmd}"

    for blocked_cmd in ("ps aux", "pgrep python", "top -b -n 1", "pstree"):
        with pytest.raises(SecurityViolationError) as exc:
            sandbox.execute_high_stakes_tool(
                tool_name="shell_exec",
                tool_fn=dummy_exec,
                kwargs={"cmd": blocked_cmd},
                authorization_token=token,
                proposal_id="prop_proc_1",
                target_resource="data_store",
            )
        assert "Process enumeration utility" in str(exc.value)


def test_heartbeat_watchdog_detects_tampering_and_triggers_containment():
    """Verify HeartbeatWatchdog detects environment variable tampering and triggers containment termination."""
    telemetry_events = []

    def mock_sink(event_name: str, payload: dict) -> None:
        if "tamper" in event_name.lower():
            telemetry_events.append((event_name, payload))

    tracing.register_sink("test_watchdog_sink", mock_sink)

    watchdog = HeartbeatWatchdog(check_interval_s=0.05)

    # Baseline check passes cleanly
    watchdog.verify_integrity()

    # Simulate adversarial modification to PATH
    orig_path = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = "/tampered/adversarial/bin:" + orig_path
        with pytest.raises(ContainmentTerminationError) as exc_contain:
            watchdog.verify_integrity()
        assert "SIGKILL containment termination triggered" in str(exc_contain.value)
        assert "modified" in str(exc_contain.value)

        # Critical telemetry event was emitted
        assert len(telemetry_events) >= 1
        assert telemetry_events[0][0] == "critical_security_tamper_detected"
        assert telemetry_events[0][1]["severity"] == "CRITICAL"
    finally:
        os.environ["PATH"] = orig_path
        tracing.unregister_sink("test_watchdog_sink")
        watchdog.stop()
