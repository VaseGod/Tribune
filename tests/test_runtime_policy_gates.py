"""Tests for runtime policy gates and active guardrail enforcement."""

import pytest
from tribune.security.policies import (
    PolicyGateViolationError,
    RuntimePolicyGate,
)


def test_forbidden_tool_blocked():
    gate = RuntimePolicyGate()

    # Prohibited tool
    with pytest.raises(PolicyGateViolationError) as exc_info:
        gate.validate_tool_call("execute_shell", {"command": "ls"})

    assert "strictly prohibited" in str(exc_info.value)
    assert exc_info.value.violation_type == "FORBIDDEN_TOOL"


def test_path_traversal_and_sensitive_paths_blocked():
    gate = RuntimePolicyGate()

    # 1. Path traversal with ..
    with pytest.raises(PolicyGateViolationError) as exc_1:
        gate.validate_tool_call("read_document", {"path": "../../etc/passwd"})
    assert "Path traversal" in str(exc_1.value)

    # 2. Direct sensitive root directory
    with pytest.raises(PolicyGateViolationError) as exc_2:
        gate.validate_tool_call("read_document", {"file_path": "/etc/shadow"})
    assert "sensitive host path" in str(exc_2.value)


def test_dangerous_commands_blocked():
    gate = RuntimePolicyGate()

    with pytest.raises(PolicyGateViolationError) as exc:
        gate.validate_tool_call("run_script", {"command": "rm -rf /"})
    assert "Dangerous command" in str(exc.value)


def test_budget_caps_enforced():
    gate = RuntimePolicyGate(max_cost_cap_usd=0.50, max_token_ceiling=10_000)

    # Within limits: ok
    gate.validate_budget(cost_usd=0.20, tokens_consumed=5_000)

    # Cost cap breach
    with pytest.raises(PolicyGateViolationError) as exc_cost:
        gate.validate_budget(cost_usd=0.55, tokens_consumed=5_000)
    assert "Cost cap breached" in str(exc_cost.value)

    # Token ceiling breach
    with pytest.raises(PolicyGateViolationError) as exc_tokens:
        gate.validate_budget(cost_usd=0.10, tokens_consumed=12_000)
    assert "Token ceiling breached" in str(exc_tokens.value)


def test_kill_switch_blocks_operations():
    gate = RuntimePolicyGate()
    gate.activate_kill_switch("Host integrity violation")

    with pytest.raises(PolicyGateViolationError) as exc:
        gate.validate_tool_call("search_statute", {"query": "medicaid"})

    assert "Kill switch is active" in str(exc.value)
    assert exc.value.violation_type == "KILL_SWITCH_ACTIVE"


def test_network_isolation_gate():
    gate = RuntimePolicyGate(allow_external_network=False)

    # Localhost allowed
    gate.validate_network("http://localhost:8000/v1")

    # Legitimate .gov allowed
    gate.validate_network("https://fns.usda.gov/snap")

    # Unauthorized commercial domain blocked
    with pytest.raises(PolicyGateViolationError) as exc:
        gate.validate_network("http://evil-c2-server.com/data")
    assert "blocked by sandbox policy" in str(exc.value)
