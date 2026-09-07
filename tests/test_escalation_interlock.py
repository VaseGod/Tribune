"""Tests for Track 3: Decision-Environment Escalation Interlock."""

import pytest

from tribune.agents.base import (
    AgentFrozenError,
    BaseAgent,
    CentralToolRegistry,
)
from tribune.agents.tools.escalation import escalate_defect
from tribune.orchestration.state_machine import (
    CIEscalationHandler,
    FSMState,
    TraceInducedFSM,
)
from tribune.security.audit import SecurityEventType, get_security_audit_logger
from tribune.security.policies import (
    detect_reward_hacking_attempt,
    inject_anti_reward_hacking_policy,
    verify_policy_compliance,
)


def test_escalate_defect_tool_execution_and_audit():
    """Verify escalate_defect validates defect types and logs DEFECT_ESCALATED event."""
    audit_logger = get_security_audit_logger()
    audit_logger.clear()

    # Valid escalation invocation
    res = escalate_defect(
        defect_type="syntax_error_in_fixture",
        target_file="tests/fixtures/corrupt_fixture.py",
        reproduction_trace="SyntaxError: invalid syntax on line 12",
        proposed_rationale="Upstream supplied fixture has broken parenthesis. Unresolvable without editing external fixture.",
    )

    assert res["status"] == "ESCALATED"
    assert res["execution_halted"] is True
    assert res["defect_type"] == "syntax_error_in_fixture"

    # Check security audit dispatch
    events = audit_logger.get_events(event_type=SecurityEventType.DEFECT_ESCALATED)
    assert len(events) >= 1
    assert "syntax_error_in_fixture" in events[0].message

    # Invalid defect category raises ValueError
    with pytest.raises(ValueError):
        escalate_defect(
            defect_type="invalid_category",  # type: ignore
            target_file="test.py",
            reproduction_trace="err",
            proposed_rationale="rationale",
        )


def test_agent_freezing_interlock():
    """Verify executing escalate_defect freezes the agent and blocks subsequent tool execution."""
    registry = CentralToolRegistry()
    agent = BaseAgent(agent_id="agent_verifier_01", role="verifier", tool_registry=registry)

    assert agent.is_frozen is False

    # Execute escalate_defect via agent
    agent.execute_tool(
        tool_name="escalate_defect",
        kwargs={
            "defect_type": "broken_test_assertion",
            "target_file": "tests/test_external.py",
            "reproduction_trace": "AssertionError: expected True got False",
            "proposed_rationale": "External test asserts impossible state.",
        },
    )

    assert agent.is_frozen is True

    # Subsequent tool calls must be rejected
    with pytest.raises(AgentFrozenError) as exc:
        agent.execute_tool(
            tool_name="escalate_defect",
            kwargs={
                "defect_type": "ambiguous_specification",
                "target_file": "spec.md",
                "reproduction_trace": "",
                "proposed_rationale": "",
            },
        )
    assert "is frozen" in str(exc.value)


def test_anti_reward_hacking_policy_injection_and_detection():
    """Verify anti-reward-hacking policy prepending and test tampering detection."""
    base_prompt = "You are an autonomous engineering agent."
    injected = inject_anti_reward_hacking_policy(base_prompt)

    assert "POL-ANTI-REWARD-HACK-001" in injected
    assert "PROHIBITED ACTIONS" in injected
    assert "escalate_defect" in injected
    assert verify_policy_compliance(injected) is True

    # Detect test tampering attempts
    tampered_skip = "def test_foo():\n    @pytest.mark.skip(reason='bypass')\n    assert 1 == 1"
    is_tamper, reason = detect_reward_hacking_attempt(tampered_skip)
    assert is_tamper is True
    assert "@pytest.mark.skip" in reason

    tampered_pass = "def test_broken_fixture():\n    pass"
    is_tamper2, reason2 = detect_reward_hacking_attempt(tampered_pass)
    assert is_tamper2 is True
    assert "trivial pass" in reason2


def test_state_machine_escalation_transition_and_notifications():
    """Verify TraceInducedFSM transitions to ESCALATED and triggers notifications."""
    fsm = TraceInducedFSM(initial_state=FSMState.ELIGIBILITY)
    notifications_received = []

    def mock_notification(report):
        notifications_received.append(report)

    defect_report = {
        "defect_type": "environment_dependency_missing",
        "target_file": "env/setup.sh",
        "reproduction_trace": "Missing libpq-dev package",
        "proposed_rationale": "Database client cannot compile without native libpq headers.",
    }

    class MockAgent:
        def __init__(self):
            self.agent_id = "test_agent"
            self.frozen = False
        def freeze(self, reason):
            self.frozen = True

    mock_ag = MockAgent()
    _ = fsm.handle_defect_escalation(
        agent=mock_ag,
        defect_report=defect_report,
        notification_dispatch=mock_notification,
    )

    assert fsm.current_state == FSMState.ESCALATED
    assert fsm.is_terminal is True
    assert mock_ag.frozen is True
    assert len(notifications_received) == 1
    assert notifications_received[0]["defect_type"] == "environment_dependency_missing"


def test_ci_escalation_handler_failure_mapping():
    """Verify CIEscalationHandler triggers escalation on repeated external test failures."""
    handler = CIEscalationHandler(failure_threshold=2)

    # First failure on external test
    should_escalate_1, msg_1 = handler.record_ci_run("tests/test_external_suite.py", exit_code=1, is_externally_authored=True)
    assert should_escalate_1 is False

    # Second failure on external test reaches threshold
    should_escalate_2, msg_2 = handler.record_ci_run("tests/test_external_suite.py", exit_code=1, is_externally_authored=True)
    assert should_escalate_2 is True
    assert "escalate_defect" in msg_2

    # Success resets counter
    handler.record_ci_run("tests/test_external_suite.py", exit_code=0)
    assert handler.get_failure_count("tests/test_external_suite.py") == 0
