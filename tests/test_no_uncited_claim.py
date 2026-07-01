"""Hard rule: an uncited eligibility claim must be impossible to construct."""

import pytest
from pydantic import ValidationError

from tribune.types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    ProgramId,
    RecommendedAction,
)


def test_resolved_criterion_requires_citation():
    with pytest.raises(ValidationError):
        CriterionResult(
            criterion_id="income",
            description="income test",
            outcome=CriterionOutcome.SATISFIED,
            required=True,
            citation_ids=[],
        )


def test_unknown_criterion_may_be_uncited():
    # An UNKNOWN criterion takes no position, so it need not be cited.
    crit = CriterionResult(
        criterion_id="income",
        description="income test",
        outcome=CriterionOutcome.UNKNOWN,
        required=True,
    )
    assert crit.outcome is CriterionOutcome.UNKNOWN


def test_eligible_assessment_without_citations_rejected():
    crit = CriterionResult(
        criterion_id="income",
        description="income test",
        outcome=CriterionOutcome.UNKNOWN,
        required=True,
    )
    with pytest.raises(ValidationError):
        Assessment(
            assessment_id="a1",
            case_id="c1",
            program=ProgramId.SNAP,
            jurisdiction="EX",
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=[crit],
            citations=[],
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.9,
            rationale="r",
        )


def test_assessment_referencing_unknown_citation_rejected():
    crit = CriterionResult(
        criterion_id="income",
        description="income test",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=["does-not-exist"],
    )
    with pytest.raises(ValidationError):
        Assessment(
            assessment_id="a1",
            case_id="c1",
            program=ProgramId.SNAP,
            jurisdiction="EX",
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=[crit],
            citations=[],
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.9,
            rationale="r",
        )


def test_properly_cited_eligible_assessment_ok():
    citation = Citation(
        citation_id="snap:EX:income",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        source="7 CFR 273.9(a)(1)",
        title="Gross income standard",
        text="130 percent of poverty",
    )
    crit = CriterionResult(
        criterion_id="income",
        description="income test",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=["snap:EX:income"],
    )
    a = Assessment(
        assessment_id="a1",
        case_id="c1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[crit],
        citations=[citation],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.9,
        rationale="r",
    )
    assert a.is_assertion and a.citations
