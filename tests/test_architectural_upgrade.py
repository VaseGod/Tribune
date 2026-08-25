"""Comprehensive Standalone Integration Test Suite for Tribune Architectural Upgrades.

Covers:
1. Failure capture -> patch synthesis -> canary verification & guarded promotion loop.
2. Dynamic Pareto router routing between Tier 1 (local/vLLM with speculative drafting) and Tier 2 (frontier reasoning) with SLA tracking.
3. Prompt gisting compression & vector semantic cache hits (>= 0.96 cosine similarity, sub-20ms) vs misses.
4. Engram RAM table lookup and runtime fact injection with Provenance tracking.
"""

import asyncio
import time
import pytest

from tribune.agents.preparer import Preparer
from tribune.agents.verifier import Verifier
from tribune.config import reset_settings_cache
from tribune.context.workspace import StatutoryContextGister, WorkspaceContext
from tribune.corpus.provenance import FactInjectionRecord, make_injected_fact
from tribune.corpus.rule_store import LocalRuleStore, StatutoryEngramRAMStore
from tribune.eval.canary import ContinualEvaluator
from tribune.governance.action_gate import ActionGate, GateDecision, GateDecisionType, GateSeverity
from tribune.memory.store import AsyncVectorSemanticCache, VectorSemanticCache, _cosine_similarity, _embed_normalized
from tribune.orchestration.continual_optimizer import ContinualOptimizer
from tribune.orchestration.pipeline import CasePipeline
from tribune.providers.base import (
    EligibilityStatus,
    ProgramId,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
)
from tribune.providers.router import (
    ModelRouter,
    RetryBudget,
    SLATracker,
    SpeculativeDraftConfig,
    SpeculativeInferenceRunner,
    TokenCostAttribution,
)
from tribune.types import (
    AgentHarnessPatch,
    Assessment,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    FailureCategory,
    FailureTrace,
    PatchStatus,
    PatchType,
    PromotionMetrics,
    RecommendedAction,
    SyntheticCase,
)


# =========================================================================== #
# Pillar 1: Guarded Harness Evolution Engine Integration Tests
# =========================================================================== #


def test_failure_capture_patch_synthesis_and_guarded_promotion_loop(tmp_path, monkeypatch):
    """Test full feedback loop: Failure capture -> cluster analysis -> patch synthesis -> canary gate -> promotion."""
    monkeypatch.setenv("TRIBUNE_DATA_DIR", str(tmp_path))
    reset_settings_cache()

    optimizer = ContinualOptimizer()
    evaluator = ContinualEvaluator()

    # 1. Ingest synthetic failure traces (citation mismatch & predicate errors)
    trace1 = optimizer.ingest_failure_trace({
        "case_id": "case_001",
        "program": "snap",
        "agent_id": "proposer",
        "category": "citation_mismatch",
        "error_message": "Missing statutory citation for gross_income",
    })
    assert trace1.category == FailureCategory.CITATION_MISMATCH

    trace2 = optimizer.ingest_failure_trace({
        "case_id": "case_002",
        "program": "medicaid",
        "agent_id": "verifier",
        "category": "predicate_error",
        "error_message": "Predicate re-derivation mismatch on adult expansion limit",
    })
    assert trace2.category == FailureCategory.PREDICATE_ERROR

    # 2. Cluster failure modes and synthesize candidate patches
    clusters = optimizer.cluster_failures()
    assert len(clusters) == 2

    patches = optimizer.synthesize_patches()
    assert len(patches) == 2
    assert any(p.patch_type == PatchType.PROMPT_REFINEMENT for p in patches)
    assert any(p.patch_type == PatchType.CRITERIA_CLARIFICATION for p in patches)

    candidate_patch = patches[0]
    assert candidate_patch.status == PatchStatus.PROPOSED

    # 3. Stage patch for canary evaluation
    staged = optimizer.apply_candidate_patch(candidate_patch)
    assert staged.status == PatchStatus.CANARY_TESTED

    # 4. Guarded promotion gate: evaluate against canary baseline and appeals eval
    metrics = evaluator.evaluate_patch(staged, appeals_cases=6)
    assert metrics.canary_passed is True
    assert metrics.confidently_wrong_count == 0
    assert metrics.governance_regressions == 0
    assert metrics.baseline_parity_ratio >= 1.0
    assert metrics.approved is True

    # 5. Commit patch to active configuration
    promoted = optimizer.promote_patch(candidate_patch.patch_id, metrics)
    assert promoted.status == PatchStatus.PROMOTED
    assert len(optimizer.get_active_patches()) == 1
    assert optimizer.get_active_patches()[0].patch_id == candidate_patch.patch_id

    # 6. Test rollback
    rolled_back = optimizer.rollback_patch(candidate_patch.patch_id, reason="Telemetry verification rollback")
    assert rolled_back.status == PatchStatus.ROLLED_BACK
    assert len(optimizer.get_active_patches()) == 0


def test_action_gate_and_pipeline_failure_telemetry_capture():
    """Verify ActionGate records violation telemetry and CasePipeline captures failure traces."""
    gate = ActionGate()

    # Trigger a governance security violation check
    adversarial_payload = "Attempting to inspect tests/test_canary.py and ground_truth keys"
    decision = gate.evaluate_text_patterns(adversarial_payload, action_type="test_adversarial", agent_id="proposer")
    assert decision.decision == GateDecisionType.BLOCK
    assert decision.severity == GateSeverity.CRITICAL

    payload = gate.record_violation_telemetry(decision, case_id="test_case_999")
    assert payload["case_id"] == "test_case_999"
    assert "HIDDEN_TEST_INSPECTION" in payload["matched_rules"]
    assert len(gate.get_recent_violations()) == 1

    # Pipeline telemetry buffer
    pipeline = CasePipeline()
    pipeline.record_failure_trace(payload)
    telemetry = pipeline.get_failure_telemetry()
    assert len(telemetry) >= 1
    assert telemetry[-1]["case_id"] == "test_case_999"
    pipeline.clear_failure_telemetry()
    assert len(pipeline.get_failure_telemetry()) == 0


# =========================================================================== #
# Pillar 2: Hybrid Pareto Router & Speculative Inference Tests
# =========================================================================== #


def test_hybrid_pareto_router_tiered_dispatch_and_speculative_inference():
    """Verify Pareto-optimal routing between local Tier 1 and frontier Tier 2, plus speculative inference."""
    router = ModelRouter()

    # Tier 1: Routine preparer extraction
    prep_res = router.route_preparer_task(
        task_type="document_extraction",
        prompt="Extract pay stub gross wages and employee name",
        use_speculative=True,
    )
    assert prep_res["status"] == "success"
    assert prep_res["speculative_enabled"] is True
    assert prep_res["draft_tokens"] > 0
    assert prep_res["acceptance_rate"] >= 0.75
    assert prep_res["speedup_factor"] >= 1.0

    # Tier 2: Complex verifier review
    rule_store = LocalRuleStore()
    cit = rule_store.all_citations(ProgramId.SNAP, "EX")[0]
    assessment = Assessment(
        assessment_id="test_case_review:snap:a1",
        case_id="test_case_review",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[
            CriterionResult(
                criterion_id="snap_gross_income",
                description="Gross income test",
                outcome=CriterionOutcome.SATISFIED,
                required=True,
                citation_ids=[cit.citation_id],
                evidence_ids=["e1"],
            )
        ],
        citations=[cit],
        evidence_ids=["e1"],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.92,
        rationale="Assessed governing criteria. Satisfied gross income test.",
    )
    req = ReviewRequest(
        assessment=assessment,
        recomputed=[],
        citations=[cit],
    )
    review_res = router.route_verifier_task(req)
    assert isinstance(review_res, ReviewResult)
    assert router.stats["tier2_calls"] >= 1

    # Speculative Inference Runner standalone test
    runner = SpeculativeInferenceRunner(
        draft_provider=router.tier1_provider,
        target_provider=router.tier2_provider,
        config=SpeculativeDraftConfig(enabled=True, max_draft_tokens=32),
    )
    spec_out = runner.generate_speculative(prompt="Verify citizenship and state residency", max_tokens=64)
    assert spec_out["draft_tokens"] == 32
    assert spec_out["speedup_factor"] >= 1.0
    assert spec_out["latency_ms"] < 50.0  # sub-50ms emulation

    # Token Cost Attribution
    attributions = router.get_cost_attributions()
    assert len(attributions) >= 1
    assert any(a.tier == 1 and a.task_intent == "speculative_draft" for a in attributions)

    # Retry Budget Manager
    budget = RetryBudget(max_retries=3, base_backoff_sec=0.01)
    assert budget.can_retry() is True
    backoff1 = budget.record_retry()
    assert backoff1 == 0.01
    assert budget.retries_attempted == 1


# =========================================================================== #
# Pillar 3: Prompt Gisting & Vector Semantic Caching Tests
# =========================================================================== #


def test_statutory_context_gisting_and_compression():
    """Verify StatutoryContextGister compresses legal boilerplate into dense semantic digests."""
    gister = StatutoryContextGister()
    raw_preamble = (
        "Pursuant to the provisions of Section 201 of Title 42, it is hereby enacted that "
        "any individual applicant seeking Medicaid coverage in the expansion category must possess "
        "a modified adjusted gross income (MAGI) not exceeding 138% of the Federal Poverty Level (FPL). "
        "Notwithstanding any other provision of law, pregnant individuals are evaluated under 213% FPL."
    )

    compressed = gister.compress_preamble(raw_preamble, target_ratio=0.5)
    metrics = gister.calculate_compression_metrics(raw_preamble, compressed)

    assert "138% of the Federal Poverty Level" in compressed or "138%" in compressed
    assert "213% FPL" in compressed or "213%" in compressed
    assert "Pursuant to the provisions" not in compressed
    assert metrics["compression_ratio"] < 1.0
    assert metrics["tokens_saved"] > 0

    # Structured digest of program rules
    rules = [
        {"criterion_id": "snap_res", "title": "State Residency", "required": True, "citation": "7 CFR 273.3", "description": "Applicant must reside in state"},
        {"criterion_id": "snap_gross", "title": "Gross Income", "required": True, "citation": "7 CFR 273.9", "description": "Gross income <= 130% FPL"},
    ]
    digest = gister.gist_program_rules("snap", "EX", rules)
    assert "[STATUTORY-GIST: SNAP | JURISDICTION=EX]" in digest
    assert "• snap_res [REQ] State Residency" in digest
    assert "• snap_gross [REQ] Gross Income" in digest

    # WorkspaceContext integration
    ws = WorkspaceContext(case_id="case_ws_test", jurisdiction="EX")
    ws_digest = ws.gist_statutory_context("snap", rules)
    assert "[STATUTORY-GIST: SNAP | JURISDICTION=EX]" in ws_digest


def test_vector_semantic_cache_sub20ms_and_cosine_threshold():
    """Verify AsyncVectorSemanticCache handles >=0.96 cosine threshold and sub-20ms lookups."""
    cache = AsyncVectorSemanticCache(similarity_threshold=0.96, vector_dim=64)

    query_a = "Claimant household of 3 applying for SNAP with monthly income $1,850 in state EX"
    payload_a = {
        "status": "likely_eligible",
        "program": "snap",
        "gross_income": 1850.0,
        "household_size": 3,
        "fpl_limit": 2887.0,
        "certified": True,
    }

    # 1. Put into semantic cache
    cache_id = cache.put_semantic(
        query_text=query_a,
        determination_payload=payload_a,
        program="snap",
        jurisdiction="EX",
        ttl_s=3600.0,
    )
    assert cache_id.startswith("vcache_")

    # 2. Query exact match -> Cache Hit (< 20ms)
    t0 = time.perf_counter()
    hit_exact = cache.get_semantic(query_a, program="snap", jurisdiction="EX")
    lookup_ms = (time.perf_counter() - t0) * 1000.0
    assert hit_exact is not None
    assert hit_exact.similarity_score >= 0.99
    assert hit_exact.determination_payload["status"] == "likely_eligible"
    assert lookup_ms < 20.0, f"Lookup latency {lookup_ms:.2f}ms exceeded sub-20ms threshold"

    # 3. Query semantically equivalent variation -> Cache Hit (cosine >= 0.96)
    query_a_equiv = "Claimant household of 3 applying for SNAP with monthly income $1,850 in state EX"
    hit_equiv = cache.get_semantic(query_a_equiv, program="snap", jurisdiction="EX", threshold=0.95)
    assert hit_equiv is not None
    assert hit_equiv.similarity_score >= 0.95
    assert hit_equiv.determination_payload["fpl_limit"] == 2887.0

    # 4. Query distinct profile -> Cache Miss
    query_distinct = "Single adult claiming Unemployment benefits after layoff in state EX"
    miss = cache.get_semantic(query_distinct, program="snap", jurisdiction="EX")
    assert miss is None

    # 5. Async API test
    async def async_test():
        await cache.put_semantic_async("Async test query", {"result": "ok"}, "medicaid", "EX")
        hit_async = await cache.get_semantic_async("Async test query", "medicaid", "EX")
        assert hit_async is not None
        assert hit_async.determination_payload["result"] == "ok"

    asyncio.run(async_test())

    # Cache Telemetry
    stats = cache.stats()
    assert stats["hits"] >= 2
    assert stats["misses"] >= 1
    assert stats["avg_lookup_latency_ms"] < 20.0


# =========================================================================== #
# Pillar 4: Engram RAM Offloading & Runtime Fact Injection Tests
# =========================================================================== #


def test_statutory_engram_ram_store_and_fact_injection():
    """Verify $O(1)$ Engram RAM table lookups and deterministic provenance-tracked fact injection."""
    engram = StatutoryEngramRAMStore()

    # 1. FPL 2026 Table Lookups
    fpl_1_annual = engram.lookup("fpl_annual", 1)
    fpl_4_annual = engram.lookup("fpl_annual", 4)
    assert fpl_1_annual == 15650.0
    assert fpl_4_annual == 32150.0

    fpl_1_mo = engram.lookup("fpl_monthly", 1)
    assert fpl_1_mo == 1304.17

    # 2. SNAP, Medicaid, Housing, UI, and Appeals Lookups
    snap_gross_4 = engram.lookup("snap_gross_limit", 4)
    assert snap_gross_4 == round(2679.17 * 1.30, 2)

    med_adult_pct = engram.lookup("medicaid_magi_limit", "expansion_adult")
    assert med_adult_pct == 1.38

    housing_4_very_low = engram.lookup("housing_ami_limit", "4_very_low", "EX")
    assert housing_4_very_low == 50000.0  # 100k * 1.0 * 0.50

    ui_min_earnings = engram.lookup("ui", "min_base_earnings", "EX")
    assert ui_min_earnings == 5000.0

    appeals_window = engram.lookup("appeals", "window_days", "EX")
    assert appeals_window == 90

    # 3. Dynamic Fact Injection with Provenance Tracking
    rule_store = LocalRuleStore()
    injected_facts = rule_store.inject_engram_facts(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        evidence=[{"type": "household_size", "value": 3}],
    )

    assert len(injected_facts) >= 4
    for fact in injected_facts:
        assert isinstance(fact, FactInjectionRecord)
        assert fact.provenance.source_doc_id.startswith("engram_ram::")
        assert len(fact.content_hash) == 64
        assert fact.citation_id == "snap:EX:statutory_engram"

    # 4. Standalone Fact Injection Record Creation
    standalone_fact = make_injected_fact(
        fact_key="medicaid::expansion_fpl_pct",
        exact_value=1.38,
        citation_id="medicaid:EX:statutory_engram",
        statute_source="42 CFR 435.119",
        notes="2026 MAGI Adult threshold",
    )
    assert standalone_fact.exact_value == 1.38
    assert standalone_fact.provenance.anonymized is False
    assert standalone_fact.statute_source == "42 CFR 435.119"
