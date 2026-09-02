"""Tests for Enterprise MCP Authentication Delegation, Action Gate Event Sourcing & Rollback Sandboxing."""

import time
from unittest.mock import MagicMock

import pytest

from tribune.governance.action_gate import (
    ActionGate,
    GateDecisionType,
    SandboxContext,
    SandboxSnapshot,
    SecurityViolationError,
    SupervisorSignature,
    TrajectoryEvent,
    TrajectoryEventLog,
    TrajectoryEventType,
)
from tribune.mcp import (
    DelegatedUserTokenContext,
    MCPAuthError,
    MCPHandler,
    validate_enterprise_token,
)


def test_delegated_user_token_context():
    """Verify DelegatedUserTokenContext fields, scope checks, expiration, and renewal."""
    ctx = DelegatedUserTokenContext(
        user_id="usr_123",
        subject="sub:usr_123",
        tenant_id="tenant_alpha",
        roles=["caseworker"],
        scopes={"rules:read", "cases:read", "cases:assess"},
        expires_at=time.time() + 100.0,
    )

    assert not ctx.is_expired()
    assert ctx.has_scope("rules:read")
    assert ctx.has_scope("cases:assess")
    assert not ctx.has_scope("appeals:submit")

    # Renewal
    renewed = ctx.renew_token(ttl=3600.0)
    assert renewed.expires_at > ctx.expires_at
    assert renewed.user_id == ctx.user_id


def test_validate_enterprise_token_and_rbac():
    """Verify validate_enterprise_token extracts delegated identity and capability scopes."""
    headers = {
        "Authorization": "Bearer enterprise_jwt_token_xyz",
        "X-Tribune-Role": "admin",
        "X-Tribune-User-ID": "admin_user_01",
        "X-Tribune-Tenant-ID": "alpha_corp",
        "X-Tribune-Scopes": "rules:read,cases:assess,appeals:prepare",
    }

    ctx = validate_enterprise_token(headers=headers)
    assert ctx.user_id == "admin_user_01"
    assert ctx.tenant_id == "alpha_corp"
    assert "admin" in ctx.roles
    assert ctx.has_scope("rules:read")
    assert ctx.has_scope("cases:assess")
    assert ctx.has_scope("appeals:prepare")


def test_mcp_scope_enforcement():
    """Verify MCPHandler enforces fine-grained capability scopes and blocks unauthorized tools."""
    handler = MCPHandler()

    # 1. Token with only rules:read attempting to run a case -> Blocked (missing cases:assess)
    req = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {
            "name": "tribune_run_case",
            "arguments": {"case_id": "test_case", "target_programs": ["snap"]},
        },
    }
    headers_restricted = {
        "X-Tribune-Role": "read_only",
        "X-Tribune-Scopes": "rules:read",
    }
    resp = handler.handle_request(req, headers=headers_restricted)
    assert "error" in resp
    assert resp["error"]["code"] == -32002
    assert "Forbidden" in resp["error"]["message"]

    # 2. Token with proper cases:assess scope -> Allowed
    headers_allowed = {
        "X-Tribune-Role": "user",
        "X-Tribune-Scopes": "rules:read,cases:assess",
    }
    resp_ok = handler.handle_request(req, headers=headers_allowed)
    assert "result" in resp_ok


def test_trajectory_event_log_integrity():
    """Verify append-only TrajectoryEventLog SHA-256 chaining and tamper detection."""
    log = TrajectoryEventLog()

    e1 = log.append_event(
        case_id="case_101",
        event_type=TrajectoryEventType.MUTATION_DRAFTED,
        agent_id="proposer",
        action_name="draft_assessment",
        payload={"income": 1200},
    )
    assert e1.prev_hash == "0" * 64
    assert len(e1.event_hash) == 64

    e2 = log.append_event(
        case_id="case_101",
        event_type=TrajectoryEventType.SPECULATIVE_EXECUTED,
        agent_id="speculator",
        action_name="eval_predicate",
        payload={"result": "likely_eligible"},
    )
    assert e2.prev_hash == e1.event_hash

    # Integrity verified
    assert log.verify_log_integrity() is True

    # Simulate malicious tampering
    log.events[0].payload["income"] = 999999
    assert log.verify_log_integrity() is False


def test_sandbox_context_and_rollback():
    """Verify SandboxContext state isolation, snapshotting, and rollback capabilities."""
    sb = SandboxContext(case_id="case_202", initial_state={"status": "initial", "count": 1})
    snap1 = sb.snapshot()

    sb.mutate("status", "drafted_mutation", agent_id="agent_1")
    sb.mutate("count", 2, agent_id="agent_1")
    assert sb.active_state["status"] == "drafted_mutation"
    assert sb.active_state["count"] == 2

    # Rollback to initial snapshot
    restored = sb.rollback_to_snapshot(snap1.snapshot_id)
    assert restored["status"] == "initial"
    assert restored["count"] == 1
    assert sb.active_state["status"] == "initial"


def test_action_gate_commit_with_judge_approval():
    """Verify ActionGate commit_sandbox requires judge approval and rolls back on rejection."""
    gate = ActionGate()
    case_id = "case_gov_303"
    sb = gate.create_sandbox(case_id, initial_state={"approved": False})
    sb.mutate("approved", True, agent_id="proposer")

    # 1. Judge rejection scenario
    mock_judge_fail = MagicMock()
    mock_judge_fail.passed = False
    mock_judge_fail.reasoning = "Violation of income threshold verification"
    mock_judge_fail.failing_rules = ["RULE_SNAP_01"]

    with pytest.raises(SecurityViolationError):
        gate.commit_sandbox(case_id, judge_result=mock_judge_fail)

    # Sandbox must be rolled back
    assert sb.active_state["approved"] is False

    # Check event log records rejection and rollback
    events = gate.event_log.get_case_events(case_id)
    types = [e.event_type for e in events]
    assert TrajectoryEventType.ROLLED_BACK in types
    assert TrajectoryEventType.JUDGE_REJECTED in types

    # 2. Judge approval scenario
    sb.mutate("approved", True, agent_id="proposer")
    mock_judge_pass = MagicMock()
    mock_judge_pass.passed = True

    sig = SupervisorSignature.issue("supervisor_alice", f"commit:{case_id}")
    committed = gate.commit_sandbox(case_id, supervisor_signature=sig, judge_result=mock_judge_pass)
    assert committed["approved"] is True
    assert gate.event_log.verify_log_integrity() is True


def test_mcp_tool_path_containment_and_env_tampering_blocking():
    """Verify tool governance intercepts path traversal and privilege/env tampering."""
    gate = ActionGate()

    # Path traversal block
    def mcp_file_op(path: str, case_id: str = "c1"):
        return {"read": path}

    sig = SupervisorSignature.issue("sup", "mcp_tool:c1")
    with pytest.raises(SecurityViolationError, match="blocked out-of-bounds file system traversal"):
        gate.execute_tool(
            tool_name="mcp_file_op",
            tool_fn=mcp_file_op,
            kwargs={"path": "../../../etc/shadow", "case_id": "c1"},
            sandbox_mode=False,
            supervisor_signature=sig,
        )

    # Env tampering pattern block
    dec = gate.evaluate_text_patterns("mcp_payload: os.environ['SECRET_KEY'] = '123'")
    assert dec.decision == GateDecisionType.BLOCK
    assert "ENVIRONMENT_TAMPERING" in dec.matched_rules

