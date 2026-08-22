"""Tests for Tiered Model Router."""

from tribune.config import reset_settings_cache
from tribune.providers.base import (
    EligibilityStatus,
    ProgramId,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
    get_provider_for_role,
)
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


def test_orchestration_router_task_aware_allocation():
    from tribune.orchestration.router import Router

    orch_router = Router()

    # Document parsing, intake, and synthetic generation route to DeepSeek V4 Pro ($0.435/M)
    d1 = orch_router.route_task("document_parsing")
    assert d1.target_model == "DeepSeek V4 Pro"
    assert d1.cost_per_m_input == 0.435
    assert d1.tier == 1

    d2 = orch_router.route_task("multimodal_intake")
    assert d2.target_model == "DeepSeek V4 Pro"
    assert d2.tier == 1

    d3 = orch_router.route_task("synthetic_casegen")
    assert d3.target_model == "DeepSeek V4 Pro"
    assert d3.tier == 1

    # Complex statutory eligibility and verifications route to Grok 4.6 ($2.00/M)
    d4 = orch_router.route_task("statutory_determination", program=ProgramId.MEDICAID)
    assert d4.target_model == "Grok 4.6"
    assert d4.cost_per_m_input == 2.00
    assert d4.tier == 2

    d5 = orch_router.route_task("multi_step_verification")
    assert d5.target_model == "Grok 4.6"
    assert d5.cost_per_m_input == 2.00
    assert d5.tier == 2

    # Escalation routes to Grok 4.6
    init_route = orch_router.initial(ProgramId.MEDICAID)
    esc_route = orch_router.escalate(ProgramId.MEDICAID, init_route)
    assert esc_route.target_model == "Grok 4.6"
    assert esc_route.cost_per_m_input == 2.00


def test_reasonmaxxer_routing_complex_extraction():
    router = ModelRouter()
    # Complex extraction maps to Tier 0 (local dense Qwen3.8-27B)
    assert router.classify_task(intent="complex_extraction") == 0
    assert router.classify_task(intent="multi_step_extraction") == 0
    assert router.classify_task(intent="complex_document_extraction") == 0
    assert router.classify_task(intent="extraction", context_length=3000) == 0

    # Standard tasks follow latency / cost rules
    assert router.classify_task(intent="parsing") == 1
    assert router.classify_task(intent="classification") == 1
    assert router.classify_task(intent="reasoning") == 2


def test_reasonmaxxer_route_document_extraction_execution():
    router = ModelRouter()
    res = router.route_document_extraction(
        prompt="Extract all income sources and medical expense items from 73k token appeal document.",
        context="Appeal Case #10293",
        temperature=1.0,
        top_p=0.95,
    )
    assert res["status"] == "success"
    assert res["model"] == "qwen3.8-27b"
    assert res["temperature"] == 1.0
    assert res["top_p"] == 0.95
    assert router.stats["local_dense_calls"] == 1


def test_reasonmaxxer_local_dense_fallback():
    local_failing = MockFailingProvider(fail_with_exception=True)
    tier2 = MockSuccessProvider("tier2_sol")
    router = ModelRouter(local_dense_provider=local_failing, tier2_provider=tier2)

    def fn_local():
        raise RuntimeError("Local llama.cpp OOM error")

    def fn_tier2():
        return "tier2_commercial_success"

    res = router.route_and_execute(
        fn_tier1=lambda: "tier1",
        fn_tier2=fn_tier2,
        intent="complex_extraction",
        fn_local=fn_local,
    )
    assert res == "tier2_commercial_success"
    assert router.stats["local_dense_fallbacks"] == 1
    assert router.stats["tier2_calls"] == 1


def test_reasonmaxxer_rollback_toggle(monkeypatch):
    monkeypatch.setenv("TRIBUNE_ENABLE_REASONMAXXER", "false")
    reset_settings_cache()
    router = ModelRouter()
    assert not router.enable_reasonmaxxer

    # With ReasonMaxxer disabled, extraction does not go to tier 0
    assert router.classify_task(intent="complex_extraction") == 1
    assert router.classify_task(intent="reasoning") == 2


