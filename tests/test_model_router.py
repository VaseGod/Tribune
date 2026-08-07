"""Tests for Tiered Model Router."""

import os
import pytest
from tribune.config import TribuneSettings, reset_settings_cache
from tribune.providers.base import (
    CriterionResult,
    EligibilityStatus,
    ProgramId,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
    get_provider_for_role,
)
from tribune.providers.local_rules import LocalRulesProvider
from tribune.providers.router import ModelRouter


class MockFailingProvider:
    """Mock provider that fails or returns low confidence."""

    def __init__(self, fail_with_exception: bool = False, self_confidence: float = 0.3):
        self.name = "mock_failing"
        self.version = "1.0"
        self.fail_with_exception = fail_with_exception
        self.self_confidence = self_confidence

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        if self.fail_with_exception:
            raise RuntimeError("Tier 1 connection timeout")
        return SynthesisResult(
            status=EligibilityStatus.INDETERMINATE,
            recommended_action=None,
            self_confidence=self.self_confidence,
            rationale="Low confidence output",
        )

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        if self.fail_with_exception:
            raise RuntimeError("Tier 1 review failure")
        return ReviewResult(supported=False, concerns=["unparseable output"])


class MockSuccessProvider:
    """Mock provider that succeeds with high confidence."""

    def __init__(self, name: str = "mock_success"):
        self.name = name
        self.version = "1.0"

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        return SynthesisResult(
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            recommended_action=None,
            self_confidence=0.95,
            rationale="All criteria met cleanly",
        )

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        return ReviewResult(supported=True, concerns=[])


def test_router_classification_heuristics():
    router = ModelRouter()

    # Intent based
    assert router.classify_task(intent="parsing") == 1
    assert router.classify_task(intent="classification") == 1
    assert router.classify_task(intent="reasoning") == 2
    assert router.classify_task(intent="complex_synthesis") == 2

    # Role based
    assert router.classify_task(role="proposer") == 1
    assert router.classify_task(role="verifier") == 2

    # Context length based
    assert router.classify_task(context_length=100) == 1
    assert router.classify_task(context_length=5000) == 2

    # Synthesis request based
    req_snap = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[],
        required_total=2,
        coverage_complete=True,
        evidence_summary="test",
        citations=[],
    )
    assert router.classify_task(req=req_snap) == 1

    req_medicaid = SynthesisRequest(
        program=ProgramId.MEDICAID,
        jurisdiction="EX",
        criteria=[],
        required_total=6,
        coverage_complete=True,
        evidence_summary="test",
        citations=[],
    )
    assert router.classify_task(req=req_medicaid) == 2


def test_router_environment_variable_configuration(monkeypatch):
    monkeypatch.setenv("DEFAULT_TIER1_MODEL", "Custom-Luna-7B")
    monkeypatch.setenv("DEFAULT_TIER2_MODEL", "Custom-Sol-70B")
    reset_settings_cache()

    router = ModelRouter()
    assert router.tier1_model == "Custom-Luna-7B"
    assert router.tier2_model == "Custom-Sol-70B"


def test_router_automatic_fallback_on_exception():
    tier1 = MockFailingProvider(fail_with_exception=True)
    tier2 = MockSuccessProvider("tier2_sol")

    router = ModelRouter(tier1_provider=tier1, tier2_provider=tier2)

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[],
        required_total=2,
        coverage_complete=True,
        evidence_summary="test",
        citations=[],
    )

    res = router.synthesize_assessment(req)
    assert res.self_confidence == 0.95
    assert "Fallback" in res.rationale
    assert router.stats["fallbacks"] == 1
    assert router.stats["tier2_calls"] == 1


def test_router_automatic_fallback_on_low_confidence():
    tier1 = MockFailingProvider(fail_with_exception=False, self_confidence=0.30)
    tier2 = MockSuccessProvider("tier2_sol")

    router = ModelRouter(tier1_provider=tier1, tier2_provider=tier2)

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[],
        required_total=2,
        coverage_complete=True,
        evidence_summary="test",
        citations=[],
    )

    res = router.synthesize_assessment(req)
    assert res.self_confidence == 0.95
    assert "Fallback" in res.rationale
    assert router.stats["fallbacks"] == 1


def test_router_generic_execute_with_routing():
    router = ModelRouter()

    def t1_fn():
        return "tier1_result"

    def t2_fn():
        return "tier2_result"

    res_t1 = router.route_and_execute(t1_fn, t2_fn, intent="parsing")
    assert res_t1 == "tier1_result"

    res_t2 = router.route_and_execute(t1_fn, t2_fn, intent="reasoning")
    assert res_t2 == "tier2_result"


def test_factory_get_provider_for_role_router():
    provider = get_provider_for_role("router")
    assert isinstance(provider, ModelRouter)
