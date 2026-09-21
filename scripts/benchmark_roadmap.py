#!/usr/bin/env python3
"""Tribune Architectural Implementation Roadmap Benchmark Suite.

Executes comprehensive performance measurements across all 5 modernized subsystems:
1. Context Graph edge classification latency (local decision path target <20ms)
2. Calibrated memory management & sliding-window eviction memory proxy
3. Turn-taking latency simulation (<300ms target)
4. Audio interruption flush latency (<50ms target)
5. Dynamic MoVA adapter activation count and estimated KV-cache memory overhead
6. Sandbox shell-first vs catalog-only token consumption proxy and task completion proxy
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sandbox.container_policy import SandboxPolicy
from sandbox.sandbox_runtime import RuntimeMode, ShellSandboxExecutor
from tribune.adapters.gating import InputContextFeatures
from tribune.adapters.router import DynamicMoVARouter
from tribune.context.decision_router import (
    DeterministicHeuristicClassifier,
    EdgeDecisionRouter,
    ModelBackedRLCDClassifier,
)
from tribune.context.entities import EdgeCandidate, EdgeFeatures
from tribune.context.manager import ProactiveContextManager
from tribune.context.scoring import ContextNode
from tribune.ingestion.acoustic import (
    AudioPacket,
    FullDuplexAcousticPipeline,
    LocalLoopbackTransport,
)


def benchmark_edge_classification(iterations: int = 1000) -> dict[str, Any]:
    """Benchmark local non-autoregressive decision classification latency."""
    router_heuristic = EdgeDecisionRouter(classifier=DeterministicHeuristicClassifier())
    router_model = EdgeDecisionRouter(classifier=ModelBackedRLCDClassifier())

    cand = EdgeCandidate(
        source_id="ent_doc_wage_1",
        target_id="ent_snap_limit",
        features=EdgeFeatures(
            source_entity_id="ent_doc_wage_1",
            target_entity_id="ent_snap_limit",
            source_text="Wage pay stub $1,450",
            target_text="SNAP statutory income threshold",
            subsumption_detected=True,
            semantic_similarity=0.86,
        ),
    )

    # 1. Warm-up
    for _ in range(50):
        router_heuristic.route_candidate(cand)
        router_model.route_candidate(cand)

    # 2. Benchmark Heuristic Classifier
    start_t = time.perf_counter()
    for _ in range(iterations):
        router_heuristic.route_candidate(cand)
    total_heur_s = time.perf_counter() - start_t
    mean_heur_ms = (total_heur_s / iterations) * 1000.0

    # 3. Benchmark Model-Backed Classifier
    start_t = time.perf_counter()
    for _ in range(iterations):
        router_model.route_candidate(cand)
    total_model_s = time.perf_counter() - start_t
    mean_model_ms = (total_model_s / iterations) * 1000.0

    target_achieved = mean_heur_ms < 20.0 and mean_model_ms < 20.0

    return {
        "iterations": iterations,
        "heuristic_mean_latency_ms": round(mean_heur_ms, 4),
        "model_backed_mean_latency_ms": round(mean_model_ms, 4),
        "target_ms": 20.0,
        "target_achieved": target_achieved,
    }


def benchmark_memory_management(num_nodes: int = 100) -> dict[str, Any]:
    """Benchmark sliding-window continuous utility scoring and eviction memory proxy."""
    window_size = 2048
    manager = ProactiveContextManager(
        total_budget=32000,
        window_size=window_size,
        eviction_policy="calibrated_kernel",
    )

    start_t = time.perf_counter()
    for i in range(num_nodes):
        manager.advance_step()
        is_relevant = (i % 3 == 0)
        node = ContextNode(
            node_id=f"node_{i}",
            text=f"Sample evidence entry {i} discussing statutory eligibility limits",
            token_count=50,
            created_step=i + 1,
            last_accessed_step=i + 1,
            task_relevance=0.95 if is_relevant else 0.10,
            semantic_relevance=0.90 if is_relevant else 0.10,
        )
        manager.register_node(node)
        if is_relevant and i >= 70:
            manager.touch_node(f"node_{i}")

    # Trigger eviction to window budget
    evicted = manager.evict_to_budget(target_budget=window_size)
    duration_ms = (time.perf_counter() - start_t) * 1000.0

    telemetry = manager.get_memory_telemetry()

    return {
        "num_nodes_ingested": num_nodes,
        "active_node_count": telemetry["node_count"],
        "active_tokens": telemetry["token_estimate"],
        "eviction_count": len(evicted),
        "retrieval_precision_proxy": telemetry["retrieval_precision_proxy"],
        "average_utility": telemetry["average_node_utility_score"],
        "memory_footprint_bytes": telemetry["memory_footprint_estimate"],
        "total_processing_ms": round(duration_ms, 2),
    }


async def benchmark_acoustic_pipeline() -> dict[str, Any]:
    """Benchmark full-duplex turn-taking latency and interruption flush latency."""
    transport = LocalLoopbackTransport()
    pipeline = FullDuplexAcousticPipeline(transport=transport)
    await pipeline.start()

    # Pre-populate 50 outbound packets
    for i in range(50):
        await transport.send_packet(AudioPacket(payload=b"\x00\x01", timestamp_ms=i * 20.0, seq=i))

    pipeline.is_system_playing = True
    pipeline.state.last_system_utterance = "Verifying statutory criteria under 7 CFR 273."

    # Simulate barge-in packet
    barge_in = AudioPacket(
        payload=b"\xff\xff",
        timestamp_ms=200.0,
        seq=1,
        is_speech=True,
        volume=0.90,
    )
    start_flush = time.perf_counter()
    await pipeline.process_inbound_packet(barge_in)
    await pipeline.process_inbound_packet(barge_in)  # 2 consecutive frames trigger barge-in
    interruption_flush_ms = (time.perf_counter() - start_flush) * 1000.0

    # Measure simulated turn latency
    packets = [AudioPacket(payload=b"\x12\x34", timestamp_ms=i * 20.0, seq=i) for i in range(5)]
    start_turn = time.perf_counter()
    await pipeline.play_outbound_audio(packets, system_text="Acknowledged update.")
    turn_latency_ms = (time.perf_counter() - start_turn) * 1000.0

    await pipeline.stop()

    return {
        "turn_taking_latency_ms": round(turn_latency_ms, 2),
        "turn_latency_target_ms": 300.0,
        "interruption_flush_latency_ms": round(interruption_flush_ms, 2),
        "flush_target_ms": 50.0,
        "interruption_target_met": interruption_flush_ms < 50.0,
        "turn_target_met": turn_latency_ms < 300.0,
    }


def benchmark_mova_adapters() -> dict[str, Any]:
    """Benchmark dynamic adapter activation count and KV-cache overhead."""
    router = DynamicMoVARouter(max_active_experts=2, cache_budget_mb=1024.0)

    # 1. Pure text query
    text_ctx = InputContextFeatures(text="What are the income thresholds?")
    start_t = time.perf_counter()
    dec_text = router.route(text_ctx)
    text_latency_ms = (time.perf_counter() - start_t) * 1000.0

    # 2. Multimodal mixed query (audio + image layout)
    mm_ctx = InputContextFeatures(
        text="Examine wage pay stub layout table",
        has_audio=True,
        audio_duration_s=30.0,
        has_image=True,
    )
    start_t = time.perf_counter()
    dec_mm = router.route(mm_ctx)
    mm_latency_ms = (time.perf_counter() - start_t) * 1000.0

    # 3. Static fallback comparison
    dec_static = router.route(mm_ctx, force_static=True)

    return {
        "pure_text": {
            "active_experts": len(dec_text.active_experts),
            "estimated_kv_cache_mb": dec_text.estimated_kv_cache_mb,
            "routing_latency_ms": round(text_latency_ms, 3),
            "interference_proxy": dec_text.cross_modal_interference_proxy,
        },
        "dynamic_multimodal": {
            "active_experts": len(dec_mm.active_experts),
            "expert_names": [e.value for e in dec_mm.active_experts],
            "estimated_kv_cache_mb": dec_mm.estimated_kv_cache_mb,
            "routing_latency_ms": round(mm_latency_ms, 3),
            "interference_proxy": dec_mm.cross_modal_interference_proxy,
        },
        "static_fallback": {
            "active_experts": len(dec_static.active_experts),
            "estimated_kv_cache_mb": dec_static.estimated_kv_cache_mb,
            "interference_proxy": dec_static.cross_modal_interference_proxy,
        },
        "kv_cache_savings_mb": round(
            dec_static.estimated_kv_cache_mb - dec_mm.estimated_kv_cache_mb, 2
        ),
    }


def benchmark_sandbox_token_usage() -> dict[str, Any]:
    """Benchmark token consumption and task success proxy of shell-first vs catalog-only."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy = SandboxPolicy(allow_shell=True, allow_network=False)

        # Shell-first executor
        shell_executor = ShellSandboxExecutor(
            policy=policy,
            mode=RuntimeMode.SHELL_FIRST,
            workspace_dir=tmp_dir,
        )

        # 1. Shell-first task execution
        cmd1 = "python3 -c 'income=1450; size=3; print(f\"ELIGIBLE:{income <= 1450 + (size-1)*514}\")'"
        res_shell = shell_executor.execute_shell(cmd1)

        # 2. Catalog-only task execution
        res_catalog = shell_executor.execute_catalog_tool(
            "query_statutory_table",
            {"program": "snap", "household_size": 3, "year": 2026},
        )

        token_stats = shell_executor.get_token_comparison_stats()

        return {
            "shell_first": {
                "exit_code": res_shell.exit_code,
                "output": res_shell.stdout.strip(),
                "tokens_consumed": res_shell.metadata.get("tokens_consumed", 0),
                "duration_s": round(res_shell.duration_s, 4),
            },
            "catalog_typed": {
                "exit_code": res_catalog.exit_code,
                "tokens_consumed": res_catalog.metadata.get("tokens_consumed", 0),
                "duration_s": round(res_catalog.duration_s, 4),
            },
            "token_reduction_ratio": token_stats["token_reduction_ratio"],
            "task_completion_proxy": (
                res_shell.exit_code == 0 and res_catalog.exit_code == 0
            ),
        }


def main() -> None:
    print("================================================================================")
    print("          TRIBUNE ARCHITECTURAL IMPLEMENTATION ROADMAP BENCHMARK                ")
    print("================================================================================")

    print("\n[1/5] Benchmarking Context Graph Edge Classification Latency...")
    edge_bench = benchmark_edge_classification(iterations=1000)
    print(f"  • Heuristic Classifier Latency: {edge_bench['heuristic_mean_latency_ms']:.4f} ms")
    print(f"  • Model-Backed Classifier Latency: {edge_bench['model_backed_mean_latency_ms']:.4f} ms")
    print(f"  • Target (<20.0 ms): {'PASSED' if edge_bench['target_achieved'] else 'OPTIMIZATION NEEDED'}")

    print("\n[2/5] Benchmarking Calibrated Memory Management & Node Eviction...")
    mem_bench = benchmark_memory_management(num_nodes=100)
    print(f"  • Ingested Nodes: {mem_bench['num_nodes_ingested']}")
    print(f"  • Active Nodes Post-Eviction: {mem_bench['active_node_count']}")
    print(f"  • Eviction Count: {mem_bench['eviction_count']}")
    print(f"  • Retrieval Precision Proxy: {mem_bench['retrieval_precision_proxy'] * 100:.1f}%")
    print(f"  • Active Tokens: {mem_bench['active_tokens']}")
    print(f"  • Estimated Memory Footprint: {mem_bench['memory_footprint_bytes']} bytes")

    print("\n[3/5] Benchmarking Streaming Acoustic Pipeline...")
    audio_bench = asyncio.run(benchmark_acoustic_pipeline())
    print(f"  • Turn-Taking Latency: {audio_bench['turn_taking_latency_ms']:.2f} ms (Target <300 ms: {'PASSED' if audio_bench['turn_target_met'] else 'MISSED'})")
    print(f"  • Interruption Flush Latency: {audio_bench['interruption_flush_latency_ms']:.2f} ms (Target <50 ms: {'PASSED' if audio_bench['interruption_target_met'] else 'MISSED'})")

    print("\n[4/5] Benchmarking Dynamic MoVA Adapter Gating...")
    mova_bench = benchmark_mova_adapters()
    print(f"  • Pure Text Query Active Experts: {mova_bench['pure_text']['active_experts']} (KV-Cache: {mova_bench['pure_text']['estimated_kv_cache_mb']} MB)")
    print(f"  • Multimodal Query Active Experts: {mova_bench['dynamic_multimodal']['active_experts']} ({', '.join(mova_bench['dynamic_multimodal']['expert_names'])})")
    print(f"  • Multimodal KV-Cache Allocation: {mova_bench['dynamic_multimodal']['estimated_kv_cache_mb']} MB")
    print(f"  • Static Fallback Cache Overhead: {mova_bench['static_fallback']['estimated_kv_cache_mb']} MB")
    print(f"  • Dynamic Routing KV-Cache Savings: {mova_bench['kv_cache_savings_mb']} MB")

    print("\n[5/5] Benchmarking Sandboxed Shell vs Catalog Token Consumption...")
    sandbox_bench = benchmark_sandbox_token_usage()
    print(f"  • Shell-First Tokens: {sandbox_bench['shell_first']['tokens_consumed']}")
    print(f"  • Catalog-Typed Tokens: {sandbox_bench['catalog_typed']['tokens_consumed']}")
    print(f"  • Token Reduction Ratio: {sandbox_bench['token_reduction_ratio'] * 100:.1f}%")
    print(f"  • Task Completion Proxy: {'SUCCESS' if sandbox_bench['task_completion_proxy'] else 'FAILED'}")

    print("\n================================================================================")
    print("                    ROADMAP BENCHMARK COMPLETED SUCCESSFULLY                    ")
    print("================================================================================")

    results = {
        "timestamp": time.time(),
        "edge_classification": edge_bench,
        "memory_management": mem_bench,
        "acoustic_pipeline": audio_bench,
        "mova_adapters": mova_bench,
        "sandbox_execution": sandbox_bench,
    }

    # Save to artifacts / results
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "eval_notes", "benchmark_roadmap_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results recorded to {out_path}")


if __name__ == "__main__":
    main()
