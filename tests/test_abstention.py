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


def test_calculate_top5_entropy_and_gating():
    import pytest

    from tribune.abstention.calibration import (
        MissingLogprobsError,
        calculate_top5_entropy,
        evaluate_decision_entropy,
    )

    # 1. Deterministic / confident top token: e.g. logprob 0.0, rest very negative
    sharp_logprobs = [-0.001, -7.0, -8.0, -9.0, -10.0]
    h_sharp = calculate_top5_entropy(sharp_logprobs)
    assert h_sharp < 0.05

    # 2. Highly ambiguous / uniform top 5 tokens: e.g. all equal logprobs
    uniform_logprobs = [-1.6, -1.6, -1.6, -1.6, -1.6]
    h_uniform = calculate_top5_entropy(uniform_logprobs)
    assert h_uniform > 2.3  # log2(5) ~= 2.32

    # 3. Dict / JSON format support
    dict_logprobs = {"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.5}, {"logprob": -3.0}, {"logprob": -4.0}, {"logprob": -5.0}]}
    h_dict = calculate_top5_entropy(dict_logprobs)
    assert 0.0 < h_dict < 1.5

    # 4. Evaluate decision entropy with tau=0.35
    h_low, gated_low = evaluate_decision_entropy(sharp_logprobs, tau=0.35)
    assert not gated_low

    h_high, gated_high = evaluate_decision_entropy(uniform_logprobs, tau=0.35)
    assert gated_high

    # 5. Missing logprobs guard
    with pytest.raises(MissingLogprobsError):
        calculate_top5_entropy(None)

    with pytest.raises(MissingLogprobsError):
        calculate_top5_entropy([])


def test_eligibility_proposer_entropy_bifurcation():
    from tribune.agents.eligibility import EligibilityProposer
    from tribune.corpus.rule_store import LocalRuleStore
    from tribune.providers.base import ModelProvider, SynthesisRequest, SynthesisResult
    from tribune.types import (
        EligibilityStatus,
        Evidence,
        EvidenceType,
        IngestMethod,
        ProgramId,
        Provenance,
        RecommendedAction,
    )

    class MockCountingProvider(ModelProvider):
        name = "mock_counting"
        version = "1.0"
        calls: int = 0

        def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
            self.calls += 1
            return SynthesisResult(
                status=EligibilityStatus.LIKELY_ELIGIBLE,
                recommended_action=RecommendedAction.PREPARE_APPLICATION,
                self_confidence=0.92,
                rationale="Model synthesis executed.",
            )

        def review_assessment(self, req):
            pass

    rule_store = LocalRuleStore()
    provider = MockCountingProvider()
    proposer = EligibilityProposer(provider, rule_store)

    prov = Provenance(source_doc_id="doc1", ingest_method=IngestMethod.MANUAL, anonymized=True, content_hash="hash123")
    evidence = [
        Evidence(evidence_id="e1", type=EvidenceType.MONTHLY_INCOME, value=1200.0, provenance=prov),
        Evidence(evidence_id="e2", type=EvidenceType.HOUSEHOLD_SIZE, value=3, provenance=prov),
        Evidence(evidence_id="e3", type=EvidenceType.CITIZENSHIP_STATUS, value="citizen", provenance=prov),
        Evidence(evidence_id="e4", type=EvidenceType.RESIDENT, value=True, provenance=prov),
    ]

    # Case A: Low Entropy H < tau -> Deterministic rule lookup (Bypasses generative LLM call)
    sharp_logprobs = [-0.001, -7.0, -8.0, -9.0, -10.0]
    assessment_a, diag_a = proposer.assess(
        case_id="case_low_h",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        logprobs=sharp_logprobs,
        tau=0.35,
    )
    assert diag_a.entropy is not None
    assert diag_a.entropy < 0.35
    assert not diag_a.entropy_gated
    assert "Deterministic Rule Lookup" in assessment_a.rationale
    assert provider.calls == 0  # 0 LLM calls!

    # Case B: High Entropy H >= tau -> Model Rollout Triggered
    ambiguous_logprobs = [-1.6, -1.6, -1.6, -1.6, -1.6]
    assessment_b, diag_b = proposer.assess(
        case_id="case_high_h",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        logprobs=ambiguous_logprobs,
        tau=0.35,
    )
    assert diag_b.entropy is not None
    assert diag_b.entropy >= 0.35
    assert diag_b.entropy_gated
    assert provider.calls == 1  # 1 LLM rollout triggered

