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
    from tribune.types import Assessment, Citation, CriterionResult, CriterionOutcome, EligibilityStatus, RecommendedAction

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
    from tribune.types import Assessment, Citation, CriterionResult, CriterionOutcome, EligibilityStatus, RecommendedAction

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


