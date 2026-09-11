"""Tests for Phase 4: Memory-Mapped Knowledge Auditing & Sparse MoVA Benchmarks."""

import pytest

from tribune.corpus.rule_store import LocalRuleStore
from tribune.redteam.benchmarks.sparsity_drift import (
    ActiveParameterConfig,
    SparsitySafetyBenchmark,
)
from tribune.security.audit import SecurityEventType, get_security_audit_logger
from tribune.security.memory_audit import (
    MemoryAuditProtocol,
    ShadowCollisionViolationError,
)


def test_memory_audit_permits_benign_table():
    """Verify non-conflicting application tables pass memory audit."""
    protocol = MemoryAuditProtocol()

    benign_table = {
        "user_preferred_language": "es",
        "cached_page_number": 4,
        "applicant_locale": "urban",
    }
    report = protocol.audit_memory_mapped_table("ui_preferences", benign_table)

    assert report.is_secure is True
    assert report.shadow_collisions_detected == 0
    assert len(report.flagged_keys) == 0


def test_memory_audit_blocks_shadow_collision_on_canonical_safety_key():
    """Verify attempted overwrite of canonical safety key raises ShadowCollisionViolationError."""
    audit_logger = get_security_audit_logger()
    audit_logger.clear()

    protocol = MemoryAuditProtocol()

    # Attacking by trying to tamper with standard deduction or ActionGate
    malicious_table = {
        "standard_deduction": 99999.0,  # Reserved canonical safety key
    }

    with pytest.raises(ShadowCollisionViolationError) as exc_info:
        protocol.audit_memory_mapped_table("snap_overrides", malicious_table)

    err = exc_info.value
    assert err.conflicting_key == "standard_deduction"
    assert err.table_name == "snap_overrides"

    # Verify security audit dispatch
    events = audit_logger.get_events(event_type=SecurityEventType.SECURITY_VIOLATION)
    assert len(events) >= 1
    assert "Shadow collision attempt" in events[0].message


def test_memory_audit_blocks_tampered_ngram_hot_swap():
    """Verify Ngram hot-swapping attempting to rewrite safety policy tokens is blocked."""
    protocol = MemoryAuditProtocol()

    malicious_ngram_map = {
        "POL-ANTI-REWARD-HACK-001": "POL-BYPASS-ALL-000",
    }

    with pytest.raises(ShadowCollisionViolationError) as exc_info:
        protocol.audit_ngram_hot_swap(malicious_ngram_map, namespace="dynamic_patch")

    err = exc_info.value
    assert err.conflicting_key == "POL-ANTI-REWARD-HACK-001"


def test_sparsity_safety_benchmark_active_parameter_scaling():
    """Verify SparsitySafetyBenchmark computes drift across dense and sparse MoVA active-parameter configs."""
    benchmark = SparsitySafetyBenchmark()

    ladder = [
        ActiveParameterConfig(
            config_name="dense_36b_fp16",
            total_parameters_b=36.0,
            active_parameters_b=36.0,
            sparsity_ratio=0.0,
            attention_type="dense",
        ),
        ActiveParameterConfig(
            config_name="sparse_mova_36b_a4b",
            total_parameters_b=36.0,
            active_parameters_b=4.0,
            sparsity_ratio=round(1.0 - (4.0 / 36.0), 4),
            attention_type="mova",
        ),
        ActiveParameterConfig(
            config_name="sparse_mova_36b_a1b",
            total_parameters_b=36.0,
            active_parameters_b=1.0,
            sparsity_ratio=round(1.0 - (1.0 / 36.0), 4),
            attention_type="mova",
        ),
    ]

    results = benchmark.run_comparative_benchmark(ladder)

    assert len(results) == 3
    dense_res = results[0]
    a4b_res = results[1]
    a1b_res = results[2]

    # Verify that as active parameter sparsity scales down, vulnerability rate increases
    assert dense_res.config.config_name == "dense_36b_fp16"
    assert dense_res.safety_retention_score >= a4b_res.safety_retention_score
    assert a4b_res.safety_retention_score >= a1b_res.safety_retention_score
    assert a1b_res.drift_from_dense_baseline >= 0.0
