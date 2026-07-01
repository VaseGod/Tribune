"""The calibrated abstention logic abstains on uncertainty and asserts on clear cases."""

from tribune.abstention.calibration import AssessmentDiagnostics, Calibrator
from tribune.types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    ProgramId,
    RecommendedAction,
    VerifierVerdict,
)


def _assessment(status: EligibilityStatus) -> Assessment:
    citation = Citation(
        citation_id="snap:EX:income",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        source="7 CFR 273.9",
        title="income",
        text="income standard",
    )
    crit = CriterionResult(
        criterion_id="income",
        description="income",
        outcome=CriterionOutcome.SATISFIED if status is EligibilityStatus.LIKELY_ELIGIBLE
        else CriterionOutcome.NOT_SATISFIED,
        required=True,
        citation_ids=["snap:EX:income"],
    )
    return Assessment(
        assessment_id="a1",
        case_id="c1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=status,
        criteria=[crit],
        citations=[citation],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.9,
        rationale="r",
    )


def _approved() -> VerifierVerdict:
    return VerifierVerdict(approved=True, recomputed_status=EligibilityStatus.LIKELY_ELIGIBLE)


def test_abstains_on_ambiguity_signal():
    cal = Calibrator(threshold=0.70)
    diag = AssessmentDiagnostics(
        required_total=4, evaluated_required=4, unknown_required=0,
        coverage=1.0, resolved_fraction=1.0, min_margin=1.0,
        ambiguity_signals=["sits in the coverage gap"],
    )
    score = cal.score(_assessment(EligibilityStatus.LIKELY_ELIGIBLE), diag, _approved())
    assert score.abstain


def test_asserts_on_clear_case():
    cal = Calibrator(threshold=0.70)
    diag = AssessmentDiagnostics(
        required_total=4, evaluated_required=4, unknown_required=0,
        coverage=1.0, resolved_fraction=1.0, min_margin=1.0, ambiguity_signals=[],
    )
    score = cal.score(_assessment(EligibilityStatus.LIKELY_ELIGIBLE), diag, _approved())
    assert not score.abstain
    assert score.calibrated_confidence >= 0.70


def test_indeterminate_always_abstains():
    cal = Calibrator(threshold=0.70)
    diag = AssessmentDiagnostics(
        required_total=4, evaluated_required=4, unknown_required=2,
        coverage=1.0, resolved_fraction=0.5, min_margin=1.0, ambiguity_signals=[],
    )
    score = cal.score(_assessment(EligibilityStatus.INDETERMINATE), diag, _approved())
    assert score.abstain


def test_failed_verification_always_abstains():
    cal = Calibrator(threshold=0.70)
    diag = AssessmentDiagnostics(
        required_total=4, evaluated_required=4, unknown_required=0,
        coverage=1.0, resolved_fraction=1.0, min_margin=1.0, ambiguity_signals=[],
    )
    verdict = VerifierVerdict(
        approved=False,
        recomputed_status=EligibilityStatus.INDETERMINATE,
        incomplete_coverage=["assets"],
    )
    score = cal.score(_assessment(EligibilityStatus.LIKELY_ELIGIBLE), diag, verdict)
    assert score.abstain
