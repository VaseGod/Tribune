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


def test_transactional_pointer_resets_and_o1_rollback():
    """Verify TransactionalKVCache captures baseline sequence length S,
    allocates K candidate slots, and on partial acceptance m < K resets pointer
    to S + m in O(1) without tensor memory copies, allocations, or relocations."""
    import numpy as np

    from tribune.orchestration.cache import TransactionalKVCache

    cache = TransactionalKVCache(max_seq_len=2048, num_heads=4, head_dim=64, precision="BF16")

    # 1. Populate baseline sequence S = 120
    s_keys = np.random.randn(120, 4, 64).astype(np.float32)
    s_values = np.random.randn(120, 4, 64).astype(np.float32)
    s_len = cache.append_tokens(s_keys, s_values)
    assert s_len == 120
    assert cache.seq_len == 120

    # 2. Begin transaction for speculative depth K = 4
    k = 4
    tx_id = cache.begin_transaction(speculative_depth_k=k)

    # 3. Dispatch speculative MTP branch of length K
    spec_keys = np.random.randn(k, 4, 64).astype(np.float32)
    spec_values = np.random.randn(k, 4, 64).astype(np.float32)
    tentative_len = cache.allocate_speculative_branch(tx_id, spec_keys, spec_values)
    assert tentative_len == 120 + k
    assert cache.seq_len == 124

    # 4. Target model verification accepts m = 2 < K tokens
    m = 2
    # Baseline pointer resets to S + m = 122 in O(1)
    # Zero re-allocations and zero tensor copies occurred
    committed_len = cache.commit_transaction(tx_id, accepted_m=m)
    assert committed_len == 120 + m
    assert cache.seq_len == 122
    assert cache.allocation_count == 0  # No runtime memory re-allocations
    assert cache.copy_count == 0  # No tensor copies
    assert cache.rollback_count == 1

    # 5. Verify full rollback on another transaction
    tx_id2 = cache.begin_transaction(speculative_depth_k=3)
    spec_keys2 = np.random.randn(3, 4, 64).astype(np.float32)
    spec_values2 = np.random.randn(3, 4, 64).astype(np.float32)
    cache.allocate_speculative_branch(tx_id2, spec_keys2, spec_values2)
    assert cache.seq_len == 125

    rolled_back_len = cache.rollback_transaction(tx_id2)
    assert rolled_back_len == 122
    assert cache.seq_len == 122


def test_hybrid_sliding_window_transient_and_pinned_persistent_tiers():
    """Verify HybridTieredKVCache separates pinned uncompressed BF16 persistent context
    from cyclical sliding-window transient reasoning."""
    from tribune.orchestration.cache import HybridTieredKVCache

    cache = HybridTieredKVCache(transient_window_size=100, deep_horizon_threshold=500)

    # 1. Pin persistent system instructions and repository context
    sys_block = cache.pin_persistent_context(
        block_id="sys_instructions",
        context_text_or_tokens="System Prompt: Statutory Adjudication Guidelines 7 CFR 273",
        tokens_count=50,
    )
    assert sys_block.is_pinned is True
    assert sys_block.precision == "BF16"
    assert sys_block.tier == "persistent"
    assert "sys_instructions" in cache.persistent_blocks

    # 2. Append intermediate reasoning chunks in sliding window (total > 100 limit)
    # Reasoning block 1: 40 tokens
    _b1 = cache.append_transient_reasoning("reason_1", "Thought step 1", 40)
    # Reasoning block 2: 40 tokens
    _b2 = cache.append_transient_reasoning("reason_2", "Thought step 2", 40)
    # Reasoning block 3: 40 tokens -> Total transient = 120 > 100 -> b1 rolled over cyclically
    _b3 = cache.append_transient_reasoning("reason_3", "Thought step 3", 40)

    # b1 evicted from sliding window, b2 and b3 remain
    assert cache.cyclical_evictions_count >= 1
    assert [b.block_id for b in cache.transient_blocks] == ["reason_2", "reason_3"]

    # Crucially, persistent block is NEVER evicted
    assert "sys_instructions" in cache.persistent_blocks
    assert cache.persistent_blocks["sys_instructions"].is_pinned is True
    assert cache.total_persistent_tokens == 50


def test_selective_quantization_deep_horizon_cache_bloat_defense():
    """Verify selective quantization schemes on long-context blocks prevent cache bloat
    during deep-horizon execution while preserving uncompressed BF16 for pinned context."""
    from tribune.orchestration.cache import HybridTieredKVCache

    cache = HybridTieredKVCache(transient_window_size=10000, deep_horizon_threshold=1000)

    # Persistent pinned system context in BF16
    cache.pin_persistent_context("pinned_constitution", "Constitutional statutes", 300)

    # Inactive historical reasoning blocks
    cache.append_transient_reasoning("hist_step_1", "Initial historical facts", 400)
    cache.append_transient_reasoning("hist_step_2", "Intermediate evidentiary notes", 400)
    cache.append_transient_reasoning("active_scratchpad", "Active live reasoning", 200)

    # Trigger selective quantization defense
    res = cache.apply_selective_quantization(target_quant_format="INT8")

    assert res["quantized_blocks_count"] >= 1
    assert res["memory_saved_bytes"] > 0
    assert res["persistent_blocks_uncompressed_bf16"] == 1

    # Check that pinned context stayed uncompressed BF16
    assert cache.persistent_blocks["pinned_constitution"].precision == "BF16"
    assert cache.persistent_blocks["pinned_constitution"].is_quantized is False

    # Check that historical blocks were selectively quantized to INT8 to prevent bloat
    hist_block_1 = next(b for b in cache.transient_blocks if b.block_id == "hist_step_1")
    assert hist_block_1.is_quantized is True
    assert hist_block_1.precision == "INT8"
