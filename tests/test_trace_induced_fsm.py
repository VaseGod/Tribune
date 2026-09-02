"""Tests for Trace-Induced Finite-State Machines (FSMs) and JIT Tool Harness Synthesis."""

from __future__ import annotations

import pytest

from tribune.orchestration.pipeline import JITToolHarnessSynthesizer
from tribune.orchestration.state_machine import (
    FSMState,
    InvalidStateTransitionError,
    TraceInducedFSM,
)
from tribune.governance.action_gate import SecurityViolationError


def test_fsm_valid_statutory_sequence() -> None:
    fsm = TraceInducedFSM(initial_state=FSMState.PREPARER)
    assert fsm.current_state == FSMState.PREPARER

    # 1. Preparer -> Eligibility
    s1 = fsm.transition(FSMState.ELIGIBILITY, agent="eligibility_proposer", action="assess")
    assert s1 == FSMState.ELIGIBILITY

    # 2. Eligibility -> Navigator
    s2 = fsm.transition(FSMState.NAVIGATOR, agent="navigator", action="plan & cross-reference")
    assert s2 == FSMState.NAVIGATOR

    # 3. Navigator -> Verifier
    s3 = fsm.transition(FSMState.VERIFIER, agent="verifier", action="verify")
    assert s3 == FSMState.VERIFIER

    # 4. Verifier -> ActionGate
    s4 = fsm.transition(FSMState.ACTION_GATE, agent="action_gate", action="gate")
    assert s4 == FSMState.ACTION_GATE

    # 5. ActionGate -> Done
    s5 = fsm.transition(FSMState.DONE, agent="navigator", action="conclude")
    assert s5 == FSMState.DONE
    assert fsm.is_terminal is True
    assert fsm.unscripted_attempts == 0


def test_fsm_rejection_of_unscripted_transitions() -> None:
    fsm = TraceInducedFSM(initial_state=FSMState.PREPARER)

    # Attempt illegal skip: PREPARER -> ACTION_GATE (bypassing ELIGIBILITY, NAVIGATOR, VERIFIER)
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        fsm.transition(FSMState.ACTION_GATE, agent="adversary", action="bypass")

    assert "Unscripted agent state transition attempted" in str(exc_info.value)
    assert fsm.unscripted_attempts == 1
    assert fsm.current_state == FSMState.ABSTAIN  # Fails closed to ABSTAIN


def test_fsm_replan_loop_transition() -> None:
    fsm = TraceInducedFSM(initial_state=FSMState.PREPARER)
    fsm.transition(FSMState.ELIGIBILITY, agent="proposer")
    fsm.transition(FSMState.NAVIGATOR, agent="navigator")
    fsm.transition(FSMState.VERIFIER, agent="verifier")

    # Verifier -> Replan -> Eligibility
    fsm.transition(FSMState.REPLAN, agent="navigator", action="replan")
    assert fsm.current_state == FSMState.REPLAN
    fsm.transition(FSMState.ELIGIBILITY, agent="proposer", action="re-assess")
    assert fsm.current_state == FSMState.ELIGIBILITY


def test_jit_tool_harness_authorized_execution() -> None:
    synthesizer = JITToolHarnessSynthesizer()
    
    # Authorized lookup_program_rules
    res = synthesizer.execute_tool(
        tool_name="lookup_program_rules",
        arguments={"program": "snap", "jurisdiction": "EX"},
    )
    assert res is not None
    assert res.get("program") == "snap"
    assert synthesizer.executed_tools_count == 1
    assert synthesizer.unscripted_blocks_count == 0


def test_jit_tool_harness_blocks_unauthorized_tool() -> None:
    synthesizer = JITToolHarnessSynthesizer()

    with pytest.raises(SecurityViolationError) as exc_info:
        synthesizer.execute_tool(
            tool_name="execute_arbitrary_shell_command",
            arguments={"cmd": "rm -rf /"},
        )

    assert "JIT Tool Harness Security Block" in str(exc_info.value)
    assert synthesizer.unscripted_blocks_count == 1
