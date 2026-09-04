"""Tests for KV-Cache Integrity Enforcement, Precision Isolation, and 128k Long-Context Evaluation.

Validates:
1. Strict enforcement of unquantized BF16 KV cache across all quantization rungs.
2. Barrier rejection of weight-only quantization settings contaminating KV-cache precision.
3. 128,000 token context stress testing without statutory context degradation or precision underflow.
4. benchmark_long_context_kv_integrity verification.
"""

from __future__ import annotations

import pytest

from tribune.eval.quant_sensitivity import (
    KVCacheIntegrityBarrier,
    KVCacheIntegrityError,
    QuantRung,
    benchmark_long_context_kv_integrity,
    multi_format_quant_ladder,
)


def test_quant_rungs_default_to_bf16_kv_cache():
    ladder = multi_format_quant_ladder()
    for rung in ladder:
        # Every rung must enforce BF16 unquantized KV-cache regardless of weight quantization
        assert rung.kv_cache_precision == "BF16"
        assert rung.max_context_length >= 128000


def test_kv_cache_integrity_barrier_passes_clean_bf16():
    rung = QuantRung(
        label="awq-4bit-test",
        quant_format="awq-4bit",
        kv_cache_precision="BF16",
        max_context_length=131072,
    )
    result = KVCacheIntegrityBarrier.assert_kv_cache_integrity(
        rung=rung,
        context_tokens=128000,
        enforce_bf16=True,
    )
    assert result["status"] == "passed_barrier"
    assert result["kv_cache_precision"] == "BF16"
    assert result["attenuation_factor"] >= 0.95


def test_kv_cache_integrity_barrier_rejects_quantized_kv_contamination():
    # Attempting to set contaminated 4-bit or 8-bit KV-cache on quantized weights
    contaminated_rung = QuantRung(
        label="contaminated-q4-kv",
        quant_format="gguf-q4_k_m",
        kv_cache_precision="INT4",  # Contamination!
    )
    with pytest.raises(KVCacheIntegrityError) as exc_info:
        KVCacheIntegrityBarrier.assert_kv_cache_integrity(
            rung=contaminated_rung,
            context_tokens=1024,
            enforce_bf16=True,
        )
    assert "KV-cache precision violation" in str(exc_info.value)
    assert "INT4" in str(exc_info.value)


def test_kv_cache_integrity_barrier_rejects_exceeded_context():
    rung = QuantRung(
        label="test-ctx-limit",
        quant_format="fp16",
        kv_cache_precision="BF16",
        max_context_length=32768,
    )
    with pytest.raises(KVCacheIntegrityError) as exc_info:
        KVCacheIntegrityBarrier.assert_kv_cache_integrity(
            rung=rung,
            context_tokens=65536,
        )
    assert "exceeds maximum supported capacity" in str(exc_info.value)


def test_long_context_128k_stress_evaluation():
    results = benchmark_long_context_kv_integrity(context_tokens=128000)
    assert len(results) > 0
    for res in results:
        assert res["context_tokens"] == 128000
        assert res["kv_cache_precision"] == "BF16"
        assert res["context_retention"] >= 0.99
        assert res["passed_barrier"] is True
