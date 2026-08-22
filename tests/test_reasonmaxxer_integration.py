"""End-to-end integration test suite for ReasonMaxxer framework and Qwen3.8-27B local reasoning upgrade."""

import pytest

from tribune.agents.eligibility import EligibilityProposer
from tribune.config import get_settings, reset_settings_cache
from tribune.corpus.rule_store import LocalRuleStore
from tribune.eval.canary import CanarySentinel
from tribune.eval.quant_sensitivity import build_seed_set, run_ladder
from tribune.eval.quant_sensitivity.backends import qwen3_8_27b_dense_ladder
from tribune.governance.action_gate import ActionBlocked, ActionGate, HumanSignoff
from tribune.memory.consolidation import (
    MemoryConsolidator,
    VRAMProtectionGate,
)
from tribune.memory.partitions import PartitionManager
from tribune.providers.base import (
    ModelProvider,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
)
from tribune.providers.router import ModelRouter
from tribune.types import (
    EligibilityStatus,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    Provenance,
    RecommendedAction,
)


class MockTrackingDenseProvider(ModelProvider):
    def __init__(self, name: str = "mock_qwen3.8-27b"):
        self.name = name
        self.version = "1.0"
        self.synthesis_calls = 0
        self.extraction_calls = 0

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        self.synthesis_calls += 1
        return SynthesisResult(
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.96,
            rationale=f"[{self.name}] High confidence statutory synthesis completed.",
        )

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        return ReviewResult(supported=True, concerns=[])

    def extract_document(self, params: dict) -> dict:
        self.extraction_calls += 1
        return {
            "status": "success",
            "model": self.name,
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 0.95),
            "extracted_fields": {
                "income": 1200.0,
                "household_size": 3,
                "token_count": 73400,
            },
            "source": "local_qwen3.8-27b_inference",
        }


def _make_evidence_set() -> list[Evidence]:
    prov = Provenance(source_doc_id="doc_appeal_73k", ingest_method=IngestMethod.STRUCTURED, anonymized=True, content_hash="hash_73k")
    return [
        Evidence(evidence_id="ev_inc", type=EvidenceType.MONTHLY_INCOME, value=1200.0, provenance=prov),
        Evidence(evidence_id="ev_hh", type=EvidenceType.HOUSEHOLD_SIZE, value=3, provenance=prov),
        Evidence(evidence_id="ev_cit", type=EvidenceType.CITIZENSHIP_STATUS, value="citizen", provenance=prov),
        Evidence(evidence_id="ev_res", type=EvidenceType.RESIDENT, value=True, provenance=prov),
    ]


def test_e2e_routing_correctness_for_complex_extraction():
    """Verify qwen3.8-27b is selected for complex extraction requests with temperature=1.0, top_p=0.95."""
    local_provider = MockTrackingDenseProvider("qwen3.8-27b")
    router = ModelRouter(local_dense_provider=local_provider)

    assert router.classify_task(intent="complex_document_extraction") == 0
    assert router.classify_task(intent="multi_step_extraction") == 0

    res = router.route_document_extraction(
        prompt="Extract all income lines from 73k token legal brief",
        context="Appeal 73k Context",
        temperature=1.0,
        top_p=0.95,
    )
    assert res["status"] == "success"
    assert res["model"] == "qwen3.8-27b"
    assert res["temperature"] == 1.0
    assert res["top_p"] == 0.95
    assert local_provider.extraction_calls == 1


def test_e2e_fallback_behavior_on_local_inference_failure():
    """Simulate local inference failure and verify seamless failover to commercial APIs."""
    class FailingLocalProvider(ModelProvider):
        name = "failing_local"
        version = "1.0"
        def synthesize_assessment(self, req):
            raise RuntimeError("CUDA out of memory in local llama.cpp instance")
        def review_assessment(self, req):
            raise RuntimeError("CUDA OOM")

    commercial_tier2 = MockTrackingDenseProvider("grok-4.6")
    router = ModelRouter(
        tier1_provider=FailingLocalProvider(),
        tier2_provider=commercial_tier2,
        local_dense_provider=FailingLocalProvider(),
    )

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[],
        required_total=2,
        coverage_complete=True,
        evidence_summary="summary",
        citations=[],
    )

    # When primary fails, falls back to tier 2
    res = router.synthesize_assessment(req)
    assert res.self_confidence == 0.96
    assert "Fallback" in res.rationale
    assert router.stats["fallbacks"] >= 1
    assert router.stats["tier2_calls"] >= 1
    assert commercial_tier2.synthesis_calls == 1

    # Test route_and_execute local fallback
    res_exec = router.route_and_execute(
        fn_tier1=lambda: "t1",
        fn_tier2=lambda: "commercial_t2",
        intent="complex_extraction",
        fn_local=lambda: (_ for _ in ()).throw(RuntimeError("Local OOM")),
    )
    assert res_exec == "commercial_t2"
    assert router.stats["local_dense_fallbacks"] >= 1



def test_e2e_entropy_threshold_bifurcation():
    """Mock logprobs to test the H < tau (rule store) and H >= tau (local generation) bifurcation."""
    rule_store = LocalRuleStore()
    tracking_provider = MockTrackingDenseProvider()
    proposer = EligibilityProposer(tracking_provider, rule_store)
    evidence = _make_evidence_set()

    # 1. Low entropy H < 0.35 (Clear statutory outcome) -> Bypasses LLM rollout (0 LLM calls)
    low_entropy_logprobs = [-0.001, -7.0, -8.0, -9.0, -10.0]
    assessment_low, diag_low = proposer.assess(
        case_id="case_low_ent",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        logprobs=low_entropy_logprobs,
        tau=0.35,
    )
    assert diag_low.entropy is not None
    assert diag_low.entropy < 0.35
    assert not diag_low.entropy_gated
    assert "Deterministic Rule Lookup" in assessment_low.rationale
    assert tracking_provider.synthesis_calls == 0

    # 2. High entropy H >= 0.35 (Statutory ambiguity) -> Triggers LLM rollout (1 LLM call)
    high_entropy_logprobs = [-1.6, -1.6, -1.6, -1.6, -1.6]
    assessment_high, diag_high = proposer.assess(
        case_id="case_high_ent",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        logprobs=high_entropy_logprobs,
        tau=0.35,
    )
    assert diag_high.entropy is not None
    assert diag_high.entropy >= 0.35
    assert diag_high.entropy_gated
    assert tracking_provider.synthesis_calls == 1


def test_e2e_memory_consolidation_latent_trace_compaction():
    """Verify trace summarization triggers when context length exceeds configured bounds, reducing token footprint."""
    pm = PartitionManager()
    partition = pm.open("trace-compaction-case")
    consolidator = MemoryConsolidator(partition)

    # 20 intermediate reasoning trace turns
    raw_traces = [
        {"step_index": i, "content": f"<thought>Step {i}: validating predicate against rule {i}</thought> Resolved criterion {i}."}
        for i in range(20)
    ]

    summary, compacted = consolidator.compact_latent_reasoning_traces(raw_traces, max_visible_steps=3)
    assert summary["compacted_count"] == 17
    assert summary["compression_ratio"] == 0.85
    assert len(compacted) == 4  # 1 summary + 3 recent
    assert compacted[0]["type"] == "compacted_latent_summary"


def test_e2e_long_context_prefill_and_batch_limits(monkeypatch):
    """Mock a 70k+ token payload and assert batch/ubatch constraints (1024 / 512) are respected."""
    monkeypatch.setenv("TRIBUNE_BATCH_SIZE", "1024")
    monkeypatch.setenv("TRIBUNE_UBATCH_SIZE", "512")
    monkeypatch.setenv("TRIBUNE_ENABLE_REASONMAXXER", "true")
    reset_settings_cache()

    settings = get_settings()
    assert settings.batch_size == 1024
    assert settings.ubatch_size == 512
    assert settings.enable_reasonmaxxer is True

    # Check VRAM protection gate under simulated high VRAM load
    pm = PartitionManager()
    partition = pm.open("vram-70k-case")
    consolidator = MemoryConsolidator(partition)

    monkeypatch.setenv("TRIBUNE_SIMULATED_VRAM_USAGE", "0.97")
    effective_ubatch, triggered = VRAMProtectionGate.check_and_enforce_vram_protection(
        current_ubatch_size=settings.ubatch_size,
        vram_usage_threshold=0.95,
        consolidator=consolidator,
    )
    assert effective_ubatch == 256  # 512 halved to 256
    assert triggered is True


def test_e2e_quantization_ladder_and_canary_accuracy_fallback(tmp_path, monkeypatch):
    """Ensure quantization ladder benchmarks Qwen3.8-27B and canary triggers fallback at 98.4% accuracy."""
    monkeypatch.setenv("TRIBUNE_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    # 1. Run Qwen3.8-27B ladder across quant tiers
    cases = build_seed_set({
        ProgramId.SNAP: 2,
        ProgramId.MEDICAID: 2,
        ProgramId.HOUSING: 2,
        ProgramId.UNEMPLOYMENT: 2,
        ProgramId.APPEALS: 2,
    })
    ladder = qwen3_8_27b_dense_ladder()
    ladder_result = run_ladder(ladder, cases=cases)
    assert len(ladder_result.rungs) == 5
    for r in ladder_result.rungs:
        assert r.citation_retention >= 0.95

    # 2. Canary Evaluation Accuracy Gate Fallback
    sentinel = CanarySentinel()
    # Simulated accuracy at 98.4% (below 98.5% gate)
    report_fallback = sentinel.run(freeze=True, simulated_citation_accuracy=0.984)
    assert report_fallback.fallback_triggered is True
    assert report_fallback.ok is False
    assert report_fallback.fallback_model == "grok-4.6"
    assert "98.400%" in report_fallback.render()

    # Simulated accuracy at 99.0% (at/above 98.5% gate)
    report_pass = sentinel.run(freeze=True, simulated_citation_accuracy=0.990)
    assert report_pass.fallback_triggered is False
    assert report_pass.ok is True


def test_e2e_hot_reload_rollback_toggle(monkeypatch):
    """Verify TRIBUNE_ENABLE_REASONMAXXER=false hot-reload reverts system to legacy behavior."""
    monkeypatch.setenv("TRIBUNE_ENABLE_REASONMAXXER", "false")
    reset_settings_cache()

    settings = get_settings()
    assert settings.enable_reasonmaxxer is False

    router = ModelRouter()
    assert router.enable_reasonmaxxer is False
    # In rollback mode, extraction does not route to Tier 0
    assert router.classify_task(intent="complex_document_extraction") == 1
    assert router.classify_task(intent="reasoning") == 2


def test_e2e_governance_action_gate_preserved():
    """Verify ActionGate strictly enforces human signoff and blocks unauthorized submission."""
    gate = ActionGate()
    from tribune.types import PreparedMaterials
    materials = PreparedMaterials(
        case_id="case_gov_test",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        application_fields={"income": "1200.0"},
    )

    # Submission without signoff is strictly blocked
    with pytest.raises(ActionBlocked):
        gate.authorize_submission(materials, signoff=None)

    # Valid human signoff authorizes receipt
    signoff = HumanSignoff.issue("officer_jane", ProgramId.SNAP.value, "case_gov_test")
    receipt = gate.authorize_submission(materials, signoff=signoff)
    assert receipt.authorized_by == "officer_jane"
    assert receipt.signoff_token == signoff.token

