"""Quantization-sensitivity harness.

Runs the verifier across a ladder of quantization levels and measures whether
abstention calibration survives: agreement (kappa) against both the
full-precision reference run and gold labels, abstention/over-refusal rates,
calibration drift (ECE / Brier), honesty metrics, and the Phase-1 cost metrics.
Produces "TRIBUNE Eval Note #1: Does abstention calibration survive
quantization?" as a markdown report.
"""

from .backends import (
    KVCacheIntegrityBarrier,
    KVCacheIntegrityError,
    OffloadBenchmarkMetrics,
    QuantRung,
    benchmark_qwen3_8_flash_next_offload,
    default_mock_ladder,
    high_throughput_local_benchmark_ladder,
    moe_pruned_quant_ladder,
    multi_format_quant_ladder,
    qwen3_8_flash_next_offload_ladder,
    smoke_ladder,
)
from .ladder import (
    LadderResult,
    RungResult,
    benchmark_long_context_kv_integrity,
    run_ladder,
)
from .report import render_eval_note
from .seedset import SEED_WEIGHTS, build_seed_set, load_manifest, seed_set_hash, write_manifest

__all__ = [
    "QuantRung",
    "KVCacheIntegrityBarrier",
    "KVCacheIntegrityError",
    "OffloadBenchmarkMetrics",
    "default_mock_ladder",
    "high_throughput_local_benchmark_ladder",
    "moe_pruned_quant_ladder",
    "multi_format_quant_ladder",
    "qwen3_8_flash_next_offload_ladder",
    "benchmark_qwen3_8_flash_next_offload",
    "benchmark_long_context_kv_integrity",
    "smoke_ladder",
    "LadderResult",
    "RungResult",
    "run_ladder",
    "render_eval_note",
    "SEED_WEIGHTS",
    "build_seed_set",
    "load_manifest",
    "seed_set_hash",
    "write_manifest",
]

