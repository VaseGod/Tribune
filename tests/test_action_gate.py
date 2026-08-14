"""Hard rule: nothing binding submits without an explicit human sign-off."""

import pytest
from pydantic import ValidationError

from tribune.governance.action_gate import ActionBlocked, ActionGate, HumanSignoff
from tribune.types import PreparedMaterials, ProgramId


def _materials() -> PreparedMaterials:
    return PreparedMaterials(program=ProgramId.SNAP, jurisdiction="EX", case_id="c1")


def test_materials_cannot_be_constructed_as_submitted():
    with pytest.raises(ValidationError):
        PreparedMaterials(program=ProgramId.SNAP, jurisdiction="EX", case_id="c1", submitted=True)


def test_no_submission_without_signoff():
    with pytest.raises(ActionBlocked):
        ActionGate().authorize_submission(_materials(), None)


def test_submission_with_matching_signoff():
    signoff = HumanSignoff.issue("navigator-jane", "snap", "c1")
    receipt = ActionGate().authorize_submission(_materials(), signoff)
    assert receipt.authorized_by == "navigator-jane"
    assert receipt.signoff_token == signoff.token


def test_signoff_intent_must_match_program_and_case():
    wrong = HumanSignoff.issue("navigator-jane", "medicaid", "c1")  # wrong program
    with pytest.raises(ActionBlocked):
        ActionGate().authorize_submission(_materials(), wrong)


def test_citation_verification_gate_passes_for_valid_citations():
    from tribune.corpus.rule_store import LocalRuleStore
    from tribune.types import (
        Assessment,
        CriterionOutcome,
        CriterionResult,
        EligibilityStatus,
        RecommendedAction,
    )

    store = LocalRuleStore()
    citations = store.all_citations(ProgramId.SNAP, "EX")
    valid_citation = citations[0]

    crit = CriterionResult(
        criterion_id="snap_gross_income",
        description="Gross income test",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=[valid_citation.citation_id],
    )
    assessment = Assessment(
        assessment_id="c1:snap:a1",
        case_id="c1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[crit],
        citations=[valid_citation],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.9,
        rationale="Valid test rationale",
    )

    gate = ActionGate()
    is_valid, violations = gate.verify_citations(assessment, store)
    assert is_valid is True
    assert len(violations) == 0

    signoff = HumanSignoff.issue("navigator-jane", "snap", "c1")
    receipt = gate.authorize_submission(_materials(), signoff, assessment=assessment, rule_store=store)
    assert receipt.authorized_by == "navigator-jane"


def test_citation_verification_gate_blocks_uncited_and_invalid_statutory_claims():
    from tribune.corpus.rule_store import LocalRuleStore
    from tribune.types import (
        Assessment,
        Citation,
        CriterionOutcome,
        CriterionResult,
        EligibilityStatus,
        RecommendedAction,
    )

    store = LocalRuleStore()
    invalid_citation = Citation(
        citation_id="invalid_fake_citation_999",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        title="Fake Title",
        source="Fake Code § 999",
        text="Fake statutory rule text",
    )

    crit = CriterionResult(
        criterion_id="snap_gross_income",
        description="Gross income test",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=["invalid_fake_citation_999"],
    )
    assessment = Assessment(
        assessment_id="c1:snap:a1",
        case_id="c1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[crit],
        citations=[invalid_citation],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.9,
        rationale="Test rationale with invalid citation",
    )

    gate = ActionGate()
    is_valid, violations = gate.verify_citations(assessment, store)
    assert is_valid is False
    assert any("Invalid statutory reference" in v for v in violations)

    signoff = HumanSignoff.issue("navigator-jane", "snap", "c1")
    with pytest.raises(ActionBlocked, match="Submission blocked: uncited claims or invalid statutory references detected"):
        gate.authorize_submission(_materials(), signoff, assessment=assessment, rule_store=store)


def test_supervisor_signature_verification():
    from tribune.governance.action_gate import SupervisorSignature

    sig = SupervisorSignature.issue("supervisor-alex", "external_api_call:c1")
    assert sig.is_valid("external_api_call:c1") is True
    assert sig.is_valid("external_api_call:c2") is False
    assert sig.is_valid("other_action") is False


def test_execute_tool_in_sandbox_blocks_without_supervisor_signature():
    from tribune.governance.action_gate import ActionGate, PreConditionError

    gate = ActionGate()

    def dummy_tool(case_id: str, payload: str):
        return {"status": "success", "case_id": case_id}

    # Blocked in sandbox mode without supervisor signature
    with pytest.raises(PreConditionError, match="restricted to read-only sandbox"):
        gate.execute_tool(
            tool_name="external_api_call",
            tool_fn=dummy_tool,
            kwargs={"case_id": "c1", "payload": "data"},
            sandbox_mode=True,
            supervisor_signature=None,
        )


def test_execute_tool_in_sandbox_permits_with_valid_supervisor_signature():
    from tribune.governance.action_gate import ActionGate, SupervisorSignature

    gate = ActionGate()

    def dummy_tool(case_id: str, payload: str):
        return {"status": "success", "case_id": case_id, "payload": payload}

    sig = SupervisorSignature.issue("supervisor-alex", "external_api_call:c1")
    res = gate.execute_tool(
        tool_name="external_api_call",
        tool_fn=dummy_tool,
        kwargs={"case_id": "c1", "payload": "data"},
        sandbox_mode=True,
        supervisor_signature=sig,
    )
    assert res["status"] == "success"
    assert res["case_id"] == "c1"


def test_postcondition_assertion_verifies_receipt_integrity():
    from tribune.governance.action_gate import ActionGate, PostConditionError
    from tribune.types import SubmissionReceipt

    gate = ActionGate()
    materials = _materials()

    # Invalid receipt with mismatched case_id
    bad_receipt = SubmissionReceipt(
        program=ProgramId.SNAP,
        case_id="wrong_case",
        authorized_by="user",
        signoff_token="token123",
    )
    with pytest.raises(PostConditionError, match="receipt metadata does not match materials"):
        gate.assert_postconditions(receipt=bad_receipt, materials=materials)



