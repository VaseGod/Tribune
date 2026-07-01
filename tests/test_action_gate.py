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
