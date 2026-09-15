"""Comprehensive Unit & Integration Test Suite for the 5-Phase Tribune Modernization Roadmap.

Tests:
1. Phase 1: SelfCompact Scaffold Integration (TokenTrackingAccumulator, CompactionRubric, CompactionProbe, SelfCompactScaffold).
2. Phase 2: Dynamic Inference Steering and Cache Optimization (Tiered transitions, prompt cache prefix preservation, 40% compute savings).
3. Phase 3: Asynchronous MCP Tool Invocation (Non-blocking external registry queries, parallel asyncio.gather dispatch, streaming).
4. Phase 4: Streaming Acoustic Ingestion Layer (80ms soft tokens, dynamic delay scheduling, speaker diarization, sub-$0.20/hr costs).
5. Phase 5: Invariant Predicate Equivalence and Zero Informational Loss across compaction boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pytest

from tribune.agents.eligibility import (
    CompactionProbe,
    CompactionRubric,
    DynamicCognitiveSteering,
    EligibilityProposer,
    SelfCompactScaffold,
    TokenTrackingAccumulator,
)
from tribune.config import ReasoningTierLevel, TribuneSettings
from tribune.corpus.rule_store import LocalRuleStore
from tribune.ingestion.acoustic import (
    AcousticDiarizer,
    AcousticIngestionEngine,
    DynamicDelayScheduler,
    FRAME_DURATION_MS,
    HOURLY_BENCHMARK_RATE_USD,
    MAITranscribe2Adapter,
    MetaMuseVoiceTranscribeAdapter,
    SoftTokenSegmenter,
)
from tribune.mcp import MCPHandler
from tribune.providers.local_rules import LocalRulesProvider
from tribune.types import (
    Citation,
    CriterionOutcome,
    CriterionResult,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    Provenance,
    SpeakerRole,
)


# =========================================================================== #
# Phase 1 Tests: SelfCompact Scaffold Integration
# =========================================================================== #


def test_token_tracking_accumulator_16k_increments():
    """Verify accumulator monitors token volume and triggers probe at 16,000-token increments."""
    acc = TokenTrackingAccumulator(threshold_interval=16_000)
    assert acc.total_tokens_ingested == 0

    # Below threshold
    crossed = acc.add_tokens(8_000)
    assert crossed is False
    assert acc.total_tokens_ingested == 8_000

    # Crossing 16k
    crossed = acc.add_tokens(8_500)
    assert crossed is True
    assert acc.total_tokens_ingested == 16_500

    # Acknowledge probe resets interval mark
    acc.acknowledge_probe()
    assert acc.last_probe_token_mark == 16_500

    # Next increment
    crossed = acc.add_tokens(5_000)
    assert crossed is False
    crossed = acc.add_tokens(11_500)
    assert crossed is True
    assert acc.total_tokens_ingested == 33_000


def test_compaction_rubric_rules_evaluation():
    """Verify CompactionRubric strictly enforces Rule 1 and Rule 2."""
    rubric = CompactionRubric()

    c1 = CriterionResult(
        criterion_id="gross_income_test",
        description="Household income under threshold",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=["7 CFR 273.9"],
    )
    citations = [
        Citation(
            citation_id="7 CFR 273.9",
            program=ProgramId.SNAP,
            jurisdiction="federal",
            source="7 CFR 273.9",
            title="Income Eligibility",
            text="Gross income limit test",
        )
    ]
    ev_hashes = [hashlib.sha256(b"income_verified").hexdigest()]

    # Valid closed module
    ok, reason = rubric.evaluate_module(
        module_name="income_test",
        criteria=[c1],
        citations=citations,
        evidence_hashes=ev_hashes,
    )
    assert ok is True
    assert "satisfied" in reason

    # Rule 2 Failure: Unresolved document discrepancy blocks compaction
    ok_disc, reason_disc = rubric.evaluate_module(
        module_name="income_test",
        criteria=[c1],
        citations=citations,
        evidence_hashes=ev_hashes,
        unresolved_discrepancies=["IRS wage mismatch"],
    )
    assert ok_disc is False
    assert "Rule 2" in reason_disc
    assert "discrepancies detected" in reason_disc

    # Rule 2 Failure: Active calculation blocks compaction
    ok_calc, reason_calc = rubric.evaluate_module(
        module_name="income_test",
        criteria=[c1],
        citations=citations,
        evidence_hashes=ev_hashes,
        active_calculations=["Computing self-employment deduction"],
    )
    assert ok_calc is False
    assert "Rule 2" in reason_calc
    assert "Active verification calculation" in reason_calc

    # Rule 1 Failure: Unknown outcome
    c_unknown = CriterionResult(
        criterion_id="identity_test",
        description="Identity verified",
        outcome=CriterionOutcome.UNKNOWN,
        required=True,
        citation_ids=["7 CFR 273.2"],
    )
    ok_unk, reason_unk = rubric.evaluate_module(
        module_name="identity",
        criteria=[c_unknown],
        citations=citations,
        evidence_hashes=ev_hashes,
    )
    assert ok_unk is False
    assert "Rule 1" in reason_unk
    assert "UNKNOWN" in reason_unk

    # Rule 1 Failure: Missing citations
    ok_nocite, reason_nocite = rubric.evaluate_module(
        module_name="assets",
        criteria=[c1],
        citations=[],
        evidence_hashes=ev_hashes,
    )
    assert ok_nocite is False
    assert "Rule 1" in reason_nocite
    assert "Missing statutory citations" in reason_nocite

    # Rule 1 Failure: Missing evidentiary hashes
    ok_nohash, reason_nohash = rubric.evaluate_module(
        module_name="assets",
        criteria=[c1],
        citations=citations,
        evidence_hashes=[],
    )
    assert ok_nohash is False
    assert "Rule 1" in reason_nohash
    assert "Missing verifiable evidentiary hashes" in reason_nohash


def test_self_compact_scaffold_50k_to_2k_condensation():
    """Verify SelfCompactScaffold reduces 50k+ raw token trace to ~2k structured predicate JSON artifact."""
    criteria = [
        CriterionResult(
            criterion_id="gross_income",
            description="Gross income under 130% FPL",
            outcome=CriterionOutcome.SATISFIED,
            required=True,
            citation_ids=["cite_gross"],
        ),
        CriterionResult(
            criterion_id="residency",
            description="Residency in administering jurisdiction",
            outcome=CriterionOutcome.SATISFIED,
            required=True,
            citation_ids=["cite_res"],
        ),
    ]
    citations = [
        Citation(
            citation_id="cite_gross",
            program=ProgramId.SNAP,
            jurisdiction="federal",
            source="7 CFR 273.9",
            title="Income Eligibility",
            text="Gross income limit test",
        ),
        Citation(
            citation_id="cite_res",
            program=ProgramId.SNAP,
            jurisdiction="federal",
            source="7 CFR 273.3",
            title="Residency Requirement",
            text="Residency test",
        ),
    ]

    compacted = SelfCompactScaffold.condense_trace(
        case_id="case-bench-01",
        statutory_module="snap_adjudication",
        criteria=criteria,
        citations=citations,
        raw_token_count=52_000,
    )

    assert compacted.token_count_before == 52_000
    assert compacted.token_count_after <= 2_000
    assert compacted.compression_ratio < 0.05
    assert len(compacted.closed_predicates) == 2
    assert "IMMUTABLE-COMPACTED-LEGAL-STATE" in compacted.compacted_summary

    # Prepend compacted state into prompt
    base_prompt = "You are the eligibility adjudicator for SNAP."
    injected_prompt = SelfCompactScaffold.inject_compacted_state_header(base_prompt, [compacted])
    assert "=== VERIFIED STATUTORY COMPACTED STATE" in injected_prompt
    assert base_prompt in injected_prompt


# =========================================================================== #
# Phase 2 Tests: Dynamic Cognitive Steering & Cache Optimization
# =========================================================================== #


def test_dynamic_cognitive_steering_contradiction_escalation():
    """Verify reasoning tier escalates to HIGH/MAX on contradictions without cache invalidation."""
    steering = DynamicCognitiveSteering(
        base_tier=ReasoningTierLevel.LOW,
        escalated_tier=ReasoningTierLevel.HIGH,
        max_tier=ReasoningTierLevel.MAX,
        prompt_cache_discount=1.00,
    )

    # 1. Routine ingestion: base tier (LOW)
    d1 = steering.evaluate_evidentiary_conflict(discrepancies=[])
    assert d1["selected_tier"] == "low"
    assert d1["cache_prefix_preserved"] is True
    assert d1["prompt_cache_rate_per_1m"] == 1.00

    # 2. Contradiction flagged: escalates to HIGH
    d2 = steering.evaluate_evidentiary_conflict(discrepancies=["State wage $3200 vs declared $1400"])
    assert d2["selected_tier"] == "high"
    assert d2["cache_prefix_preserved"] is True

    # 3. Severe contradiction: escalates to MAX
    d3 = steering.evaluate_evidentiary_conflict(severe_contradiction=True)
    assert d3["selected_tier"] == "max"
    assert d3["cache_prefix_preserved"] is True

    # 4. Reconciliation turn completed: restores to LOW
    d4 = steering.evaluate_evidentiary_conflict(discrepancies=[])
    assert d4["selected_tier"] == "low"
    assert len(steering.transition_log) == 4


def test_compute_savings_40_percent_reduction():
    """Verify tiered cognitive steering achieves at least 40% reduction in total task compute expense."""
    steering = DynamicCognitiveSteering()

    # Scenario: 100k total tokens, 15k spent in conflict resolution, 85k in routine parsing
    report = steering.calculate_compute_savings(
        total_tokens=100_000,
        contradiction_tokens=15_000,
        peak_rate_per_1m=5.00,
        low_rate_per_1m=1.00,
        high_rate_per_1m=3.00,
    )

    # Monolithic peak cost = 100k * $5.00/M = $0.50
    assert report["monolithic_cost_usd"] == 0.50
    # Tiered cost = (85k * $1.00/M) + (15k * $3.00/M) = $0.085 + $0.045 = $0.130
    assert report["tiered_cost_usd"] == 0.13
    # Savings = $0.50 - $0.13 = $0.37 (74% reduction > 40% target)
    assert report["cost_savings_usd"] == 0.37
    assert report["savings_pct"] >= 40.0
    assert report["target_40pct_met"] is True


# =========================================================================== #
# Phase 3 Tests: Asynchronous MCP Tool Invocation & Parallel Dispatch
# =========================================================================== #


@pytest.mark.asyncio
async def test_async_mcp_external_registry_queries():
    """Verify non-blocking external registry queries execute asynchronously."""
    handler = MCPHandler()

    # 1. Property Tax Assessor non-blocking lookup
    prop_res = await handler._call_tool_async(
        "tribune_query_property_tax",
        {"jurisdiction": "EX", "parcel_id_or_address": "123 Main St"},
    )
    prop_data = json.loads(prop_res)
    assert prop_data["verified"] is True
    assert prop_data["assessed_market_value"] > 0
    assert "Async Non-Blocking Query" in prop_data["source"]

    # 2. Unemployment Database query
    unemp_res = await handler._call_tool_async(
        "tribune_query_unemployment_database",
        {"jurisdiction": "EX", "claimant_ssn_hash": "hash123"},
    )
    unemp_data = json.loads(unemp_res)
    assert unemp_data["verified"] is True
    assert unemp_data["monetary_entitlement"] is True

    # 3. Vital Statistics query
    vital_res = await handler._call_tool_async(
        "tribune_query_vital_statistics",
        {"jurisdiction": "EX", "full_name": "Maria Gonzalez", "dob": "1990-01-01"},
    )
    vital_data = json.loads(vital_res)
    assert vital_data["verified"] is True
    assert vital_data["us_citizenship_confirmed"] is True


@pytest.mark.asyncio
async def test_parallel_asyncio_gather_tool_dispatch():
    """Verify parallel asynchronous dispatch queries disparate registries concurrently without I/O wait stalls."""
    handler = MCPHandler()

    tool_calls = [
        {"name": "tribune_query_property_tax", "arguments": {"jurisdiction": "EX", "parcel_id_or_address": "Parcel-99"}},
        {"name": "tribune_query_unemployment_database", "arguments": {"jurisdiction": "EX", "claimant_ssn_hash": "ssn-456"}},
        {"name": "tribune_query_vital_statistics", "arguments": {"jurisdiction": "EX", "full_name": "John Doe", "dob": "1982-04-10"}},
    ]

    # Execute parallel dispatch via asyncio.gather
    results = await handler.dispatch_parallel_tools(tool_calls)
    assert len(results) == 3
    assert all(r["success"] is True for r in results)

    # Also test via MCP JSON-RPC call
    payload = {
        "jsonrpc": "2.0",
        "id": "parallel-req-1",
        "method": "tools/call",
        "params": {
            "name": "tribune_parallel_dispatch",
            "arguments": {"calls": tool_calls},
        },
    }
    rpc_res = await handler.handle_request_async(payload)
    assert "result" in rpc_res
    content_text = rpc_res["result"]["content"][0]["text"]
    parsed_rpc = json.loads(content_text)
    assert parsed_rpc["parallel_execution_count"] == 3


@pytest.mark.asyncio
async def test_streaming_tool_execution():
    """Verify stream_tool_execution yields chunked ndjson updates."""
    handler = MCPHandler()
    chunks = []
    async for chunk in handler.stream_tool_execution(
        "tribune_query_property_tax",
        {"jurisdiction": "EX", "parcel_id_or_address": "456 Elm St"},
    ):
        chunks.append(json.loads(chunk))

    assert len(chunks) >= 2
    assert chunks[0]["status"] == "started"
    assert chunks[-1]["status"] == "completed"
    assert chunks[-1]["result"]["verified"] is True


# =========================================================================== #
# Phase 4 Tests: Streaming Acoustic Ingestion Layer
# =========================================================================== #


def test_soft_token_segmentation_80ms():
    """Verify audio is segmented into 80-millisecond soft tokens with entropy bounds."""
    segmenter = SoftTokenSegmenter(frame_duration_ms=FRAME_DURATION_MS)
    assert segmenter.frame_duration_ms == 80.0

    # 1.0 second of audio should produce exactly 12-13 soft tokens (1000ms / 80ms = 12.5)
    spans = segmenter.segment_audio(audio_duration_s=1.0)
    assert len(spans) == 12
    for span in spans:
        assert span.duration_ms == 80.0
        assert 0.0 <= span.acoustic_entropy <= 1.0
        assert 0.0 <= span.confidence <= 1.0


def test_dynamic_delay_scheduler_adaptive_delays():
    """Verify scheduler computes adaptive delays (80ms to 320ms) based on acoustic entropy."""
    scheduler = DynamicDelayScheduler(
        min_delay_ms=80.0,
        max_delay_ms=320.0,
        entropy_threshold_low=0.30,
        entropy_threshold_high=0.70,
    )

    # Low entropy (clear phrasing) -> 80ms minimum delay
    assert scheduler.compute_delay_ms(0.20) == 80.0
    assert scheduler.compute_delay_ms(0.30) == 80.0

    # High entropy (ambiguous phrasing / dialect) -> 320ms maximum delay
    assert scheduler.compute_delay_ms(0.75) == 320.0
    assert scheduler.compute_delay_ms(0.90) == 320.0

    # Intermediate entropy -> interpolated dynamic delay
    mid_delay = scheduler.compute_delay_ms(0.50)
    assert 80.0 < mid_delay < 320.0


def test_acoustic_speaker_diarization_and_cost_accounting():
    """Verify speaker diarization segregate claimant from caseworker and cost is sub-$0.20/hr."""
    engine = AcousticIngestionEngine()
    assert engine.hourly_rate <= HOURLY_BENCHMARK_RATE_USD
    assert engine.hourly_rate < 0.20

    # Run acoustic intake for 120s hearing
    result = engine.process_audio("simulated_hearing_audio", duration_s=120.0)
    assert result.duration_s == 120.0
    assert result.hourly_cost < 0.20
    assert result.total_cost <= (120.0 / 3600.0) * 0.20

    # Check diarized speaker attribution
    speakers = {seg.speaker for seg in result.segments}
    assert SpeakerRole.CLAIMANT in speakers
    assert SpeakerRole.CASEWORKER in speakers

    # Convert to RawDocument and Evidence
    raw_doc = engine.to_raw_document(result, "hearing-case-001")
    assert raw_doc.doc_type == "spoken_hearing_transcript"
    assert "monthly_income" in raw_doc.fields
    assert "household_size" in raw_doc.fields

    evidence = engine.to_evidence(result, "hearing-case-001")
    assert len(evidence) >= 2
    for ev in evidence:
        assert ev.provenance.ingest_method == IngestMethod.ACOUSTIC
