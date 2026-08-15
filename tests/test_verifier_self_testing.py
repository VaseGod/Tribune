"""Tests for Verifier Multi-Step Self-Testing Trajectories and Abstention Calibration."""


from tribune.abstention.calibration import AssessmentDiagnostics, Calibrator
from tribune.agents.verifier import Verifier
from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.corpus import programs as program_registry
from tribune.corpus.rule_store import LocalRuleStore
from tribune.providers.local_rules import LocalRulesProvider
from tribune.types import (
    Assessment,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    ProgramId,
    RecommendedAction,
    VerifierVerdict,
)


def _make_sample_assessment() -> tuple[Assessment, LocalRuleStore]:
    store = LocalRuleStore()
    citations = store.all_citations(ProgramId.SNAP, "EX")
    ruleset = program_registry.get_ruleset(ProgramId.SNAP)
    criteria = [
        CriterionResult(
            criterion_id=rule.criterion_id,
            description=rule.description,
            outcome=CriterionOutcome.SATISFIED,
            required=rule.required,
            citation_ids=[rule.citation(ProgramId.SNAP, "EX").citation_id],
        )
        for rule in ruleset.rules
    ]
    assessment = Assessment(
        assessment_id="c1:snap:a1",
        case_id="c1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=criteria,
        citations=citations,
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.95,
        rationale="Clean SNAP application satisfying all statutory rules.",
    )
    return assessment, store


def test_verifier_generate_self_testing_prompt():
    assessment, store = _make_sample_assessment()
    verifier = Verifier(provider=LocalRulesProvider(role="verifier"), rule_store=store)

    prompt = verifier.generate_self_testing_prompt(assessment, "EX")
    assert "gpt-5.6-sol-ultrafast" in prompt
    assert "multi-step self-testing trajectory" in prompt
    assert "Citation Integrity Check" in prompt
    assert "Cross-Statute Coherence" in prompt


def test_verifier_executes_self_testing_trajectory():
    case = SyntheticCaseGenerator().build_case(
        case_id="c1",
        jurisdiction="EX",
        overrides={"monthly_income": 1000.0, "household_size": 3},
        target_programs=[ProgramId.SNAP],
    )
    assessment, store = _make_sample_assessment()
    verifier = Verifier(provider=LocalRulesProvider(role="verifier"), rule_store=store)

    verdict = verifier.verify(assessment, case.evidence, "EX")
    assert verdict.approved is True
    assert verdict.self_testing_score >= 0.85
    assert len(verdict.trajectory_steps) == 4
    assert all(step["passed"] for step in verdict.trajectory_steps)


def test_calibrator_abstains_when_self_validation_confidence_below_parity_threshold():
    # Parity threshold configured at 0.85
    calibrator = Calibrator(threshold=0.60, min_self_validation_confidence=0.85)

    diag = AssessmentDiagnostics(
        required_total=4,
        evaluated_required=4,
        unknown_required=0,
        coverage=1.0,
        resolved_fraction=1.0,
        min_margin=0.5,
    )

    # Low self testing score: 0.70 < 0.85
    low_confidence_verdict = VerifierVerdict(
        approved=True,
        recomputed_status=EligibilityStatus.LIKELY_ELIGIBLE,
        self_testing_score=0.70,
        trajectory_steps=[{"step": 1, "milestone": "Statutory Citation Mapping", "score": 0.70}],
    )

    assessment, _ = _make_sample_assessment()

    score = calibrator.score(assessment, diag, low_confidence_verdict)
    assert score.abstain is True
    assert "intermediate self-validation confidence" in score.reason
    assert "manual administrative review" in score.reason


def test_calibrator_passes_when_self_validation_confidence_meets_parity_threshold():
    calibrator = Calibrator(threshold=0.60, min_self_validation_confidence=0.85)

    diag = AssessmentDiagnostics(
        required_total=4,
        evaluated_required=4,
        unknown_required=0,
        coverage=1.0,
        resolved_fraction=1.0,
        min_margin=0.5,
    )

    high_confidence_verdict = VerifierVerdict(
        approved=True,
        recomputed_status=EligibilityStatus.LIKELY_ELIGIBLE,
        self_testing_score=0.95,
        trajectory_steps=[{"step": 1, "milestone": "Statutory Citation Mapping", "score": 0.95}],
    )

    assessment, _ = _make_sample_assessment()

    score = calibrator.score(assessment, diag, high_confidence_verdict)
    assert score.abstain is False
    assert "meets the assertion threshold" in score.reason
