"""Tests for Hybrid-Attention Provider Routing and Engram / PLE Offloading."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from tribune.config import TribuneSettings
from tribune.providers.local_rules import (
    LocalRuntimeConfig,
    benchmark_hybrid_attention_inference,
    benchmark_local_inference,
    estimate_memory_footprint,
)
from tribune.providers.openai_compat import OpenAICompatProvider
from tribune.providers.router import ModelRouter
from tribune.types import Citation, CriterionOutcome, CriterionResult, ProgramId
from tribune.providers.base import SynthesisRequest


def test_hybrid_attention_config_and_defaults() -> None:
    settings = TribuneSettings()
    assert hasattr(settings, "vllm_ple_cpu_offload")
    assert hasattr(settings, "num_speculative_tokens")
    assert hasattr(settings, "confidence_fallback_threshold")
    assert hasattr(settings, "hybrid_attention_backend")
    assert hasattr(settings, "qwen_flash_model")
    assert settings.confidence_fallback_threshold == 0.85
    assert settings.num_speculative_tokens >= 1


def test_openai_compat_pricing_and_speculative_params() -> None:
    settings = TribuneSettings()
    provider = OpenAICompatProvider(
        model="Qwen3.8-Flash-Next",
        settings=settings,
        role="proposer",
    )
    pricing = provider.pricing_parameters
    assert pricing["input_cost_per_1m"] == 0.12
    assert pricing["output_cost_per_1m"] == 0.35
    assert provider.extra_body.get("num_speculative_tokens") == settings.num_speculative_tokens


def test_local_rules_hybrid_benchmark_throughput() -> None:
    # Test hybrid attention throughput target > 120 tokens/sec
    res = benchmark_hybrid_attention_inference(
        model_family="Qwen3.8-Flash-Next",
        cpu_offload=True,
        num_speculative_tokens=1,
        gen_tokens=128,
    )
    assert res["status"] == "success"
    assert res["tokens_per_second"] > 120.0, f"Expected >120 tok/sec, got {res['tokens_per_second']}"


def test_memory_estimation_with_ple_cpu_offload() -> None:
    cfg_without = LocalRuntimeConfig(vllm_ple_cpu_offload=False, quantization="FP16")
    cfg_with = LocalRuntimeConfig(vllm_ple_cpu_offload=True, quantization="FP16")
    mem_without = estimate_memory_footprint(cfg_without)
    mem_with = estimate_memory_footprint(cfg_with)
    assert mem_with["vram_weights_gb"] < mem_without["vram_weights_gb"]


def test_router_hybrid_attention_routing_and_confidence_fallback() -> None:
    settings = TribuneSettings(confidence_fallback_threshold=0.85)
    router = ModelRouter(settings=settings)

    citation = Citation(
        citation_id="cit_1",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        title="Gross Income Standard",
        source="7 CFR 273.9",
        text="Gross income rule",
    )
    criterion = CriterionResult(
        criterion_id="snap_gross_income",
        description="Gross income <= 130% FPL",
        outcome=CriterionOutcome.SATISFIED,
        citation_ids=["cit_1"],
        required=True,
    )

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        evidence_summary="Income: $1200, Assets: $500",
        criteria=[criterion],
        required_total=1,
        coverage_complete=True,
        citations=[citation],
    )

    # Route hybrid attention task
    res = router.route_hybrid_attention_task(req)
    assert res is not None
    assert router.stats.get("hybrid_attention_calls", 0) > 0


def test_router_confidence_fallback_trigger() -> None:
    settings = TribuneSettings(confidence_fallback_threshold=0.90)
    router = ModelRouter(settings=settings)

    # Mock low confidence response from hybrid provider
    mock_low_conf = MagicMock()
    mock_low_conf.self_confidence = 0.65
    mock_low_conf.status = "eligible"
    mock_low_conf.recommended_action = MagicMock(value="prepare_application")
    mock_low_conf.rationale = "Uncertain statutory interpretation"
    router.hybrid_provider.synthesize_assessment = MagicMock(return_value=mock_low_conf)

    citation = Citation(
        citation_id="cit_med",
        program=ProgramId.MEDICAID,
        jurisdiction="EX",
        title="Medicaid Income Standard",
        source="42 CFR 435.110",
        text="Medicaid MAGI rule",
    )
    criterion = CriterionResult(
        criterion_id="med_magi",
        description="Income <= 138% FPL",
        outcome=CriterionOutcome.SATISFIED,
        citation_ids=["cit_med"],
        required=True,
    )

    req = SynthesisRequest(
        program=ProgramId.MEDICAID,
        jurisdiction="EX",
        evidence_summary="Income: $800",
        criteria=[criterion],
        required_total=1,
        coverage_complete=True,
        citations=[citation],
    )

    res = router.route_hybrid_attention_task(req)
    assert "CALIBRATED CONFIDENCE FALLBACK" in res.rationale
    assert router.stats.get("confidence_fallbacks", 0) > 0


def test_kv_memory_monitor_and_cache_affinity_routing():
    """Verify KVMemoryMonitor tracks KV memory footprints and KVCacheAffinityRouter routes prompt-heavy tasks."""
    from tribune.providers.router import KVCacheAffinityRouter, KVMemoryMonitor, ModelRouter

    monitor = KVMemoryMonitor(num_layers=32, hidden_dim=4096)

    # 1. Memory estimation across quantization formats
    fp16_bytes = monitor.estimate_kv_memory_bytes(token_count=10_000, quant_type="FP16")
    q8_bytes = monitor.estimate_kv_memory_bytes(token_count=10_000, quant_type="q8_0")
    int4_bytes = monitor.estimate_kv_memory_bytes(token_count=10_000, quant_type="INT4")

    assert q8_bytes == fp16_bytes // 2
    assert int4_bytes == fp16_bytes // 4

    # 2. KV Affinity evaluation for short vs prompt-heavy contexts
    short_affinity = monitor.evaluate_kv_affinity(prompt_tokens=500, context_tokens=500)
    assert short_affinity["is_prompt_heavy"] is False
    assert short_affinity["recommended_kv_cache"] == "FP16"

    heavy_affinity = monitor.evaluate_kv_affinity(prompt_tokens=5000, context_tokens=1000)
    assert heavy_affinity["is_prompt_heavy"] is True
    assert heavy_affinity["recommended_kv_cache"] == "q8_0"

    # 3. Router dispatch based on KV cache affinity
    kv_router = KVCacheAffinityRouter(kv_monitor=monitor, kv_compressed_model="Qwen3.8-Flash-Next")
    res_short = kv_router.route_by_kv_affinity(prompt_tokens=500, context_tokens=200)
    assert res_short["kv_compressed_attention"] is False

    res_heavy = kv_router.route_by_kv_affinity(prompt_tokens=8000, context_tokens=1000)
    assert res_heavy["kv_compressed_attention"] is True
    assert res_heavy["selected_model"] == "Qwen3.8-Flash-Next"
    assert res_heavy["sliding_window_attention"] is True

    # 4. ModelRouter integration
    router = ModelRouter()
    r_dec = router.route_by_kv_cache_affinity(prompt_tokens=6000, is_prompt_heavy=True)
    assert r_dec["kv_compressed_attention"] is True
    assert router.stats["kv_affinity_routed_calls"] >= 1


def test_session_sticky_router_prefix_affinity_and_lifecycle():
    """Verify SessionStickyRouter maximizes warm KV-cache hits and manages concurrency lifecycle."""
    from tribune.routing.session_sticky_router import (
        CapacityExceededError,
        SessionStickyRouter,
        WorkerNodeState,
    )

    node1 = WorkerNodeState(node_id="worker_gpu_0", total_vram_gb=24.0)
    node2 = WorkerNodeState(node_id="worker_gpu_1", total_vram_gb=24.0)

    router = SessionStickyRouter(
        nodes=[node1, node2],
        max_concurrency_per_node=2,
        prefix_window=1024,
        cold_prefill_ms=100.0,
        warm_hit_ms=10.0,
    )

    # 1. First turn: Cold cache dispatch with standardized 1,024-char system prefix
    system_prompt = ("You are an administrative adjudication assistant for Title 7 CFR SNAP benefits verification and case law analysis. " * 15)[:1024]
    assert len(system_prompt) == 1024

    turn1_prompt = f"{system_prompt}\nUser: Evaluate income $1200 for family of 3."

    d1 = router.route(turn1_prompt)
    assert d1.is_cache_hit is False
    assert d1.estimated_prefill_latency_ms == 100.0
    first_node_id = d1.node_id
    assert first_node_id in ("worker_gpu_0", "worker_gpu_1")

    # 2. Multi-turn conversation continuation: Warm KV cache affinity hit (shares exact 1024 prefix)
    turn2_prompt = f"{turn1_prompt}\nAssistant: Gross income is satisfied.\nUser: Now check assets."
    d2 = router.route(turn2_prompt)
    assert d2.is_cache_hit is True
    assert d2.node_id == first_node_id  # Session affinity maintained
    assert d2.estimated_prefill_latency_ms == 10.0
    assert d2.active_concurrency == 2

    # 3. Warm node reaches max concurrency (2): Next request falls back to node2
    turn3_prompt = f"{turn1_prompt}\nAssistant: Inquire about vehicles.\nUser: 1 car valued at $4000."
    d3 = router.route(turn3_prompt)
    assert d3.is_cache_hit is False  # Cold fallback because node1 is saturated
    other_node_id = "worker_gpu_1" if first_node_id == "worker_gpu_0" else "worker_gpu_0"
    assert d3.node_id == other_node_id
    assert d3.active_concurrency == 1

    # 4. Concurrency release: release turn on first_node_id
    router.release_turn(first_node_id)
    first_node = router.get_node(first_node_id)
    assert first_node is not None
    assert first_node.active_concurrency == 1

    # 5. Subsequent request can hit first_node_id again as a warm hit
    turn4_prompt = f"{turn1_prompt}\nAssistant: Verified assets.\nUser: Proceed to filing."
    d4 = router.route(turn4_prompt)
    assert d4.is_cache_hit is True
    assert d4.node_id == first_node_id
    assert d4.active_concurrency == 2

    # Check metrics
    stats = router.get_stats()
    assert stats["total_requests"] == 4
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 2
    assert stats["hit_ratio"] == 0.5

    # 6. Saturated cluster raises CapacityExceededError
    # Fill remaining capacity on other_node
    router.route("Completely independent prompt #1" * 40)
    # Both nodes now have concurrency 2 (max_concurrency_per_node=2)
    with pytest.raises(CapacityExceededError):
        router.route("Saturated prompt that exceeds cluster capacity" * 40)

