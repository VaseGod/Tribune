"""End-to-End Integration Test for the Tribune Architectural Implementation Roadmap.

Ties together all five modern subsystems:
1. Context Graph Modernization via Calibrated Decision Routing
2. Calibrated Memory Management & Node Eviction
3. Full-Duplex Streaming Acoustic Pipeline
4. Dynamic Coarse-to-Fine MoVA Adapter Gating
5. Hardened Shell-First Sandbox Execution
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

import pytest

from sandbox.container_policy import SandboxPolicy
from sandbox.sandbox_runtime import RuntimeMode, ShellSandboxExecutor
from tribune.adapters.gating import InputContextFeatures, InputModality
from tribune.adapters.router import DynamicMoVARouter
from tribune.context.entities import EdgeCandidate, EdgeClass, EdgeFeatures, EntityMention
from tribune.context.graph import EntityEventGraph
from tribune.context.graph_builder import CalibratedGraphBuilder
from tribune.context.manager import ProactiveContextManager
from tribune.context.scoring import ContextNode
from tribune.ingestion.acoustic import AudioPacket, FullDuplexAcousticPipeline
from tribune.metrics import get_roadmap_metrics, reset_roadmap_metrics


class TestRoadmapEndToEndIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_roadmap_metrics()
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @pytest.mark.asyncio
    async def test_full_roadmap_pipeline_flow(self) -> None:
        metrics = get_roadmap_metrics()

        # Step 1: Ingest Spoken Testimony through Acoustic Pipeline
        acoustic_pipeline = FullDuplexAcousticPipeline(session_id="e2e_hearing_session")
        await acoustic_pipeline.start()

        spoken_packet = AudioPacket(
            payload=b"\x11\x22\x33\x44",
            timestamp_ms=100.0,
            seq=1,
            is_speech=True,
            volume=0.72,
            metadata={"partial_text": "I earn $1,450 monthly and pay $750 in rent", "is_final": True},
        )
        markers, transcript, is_final = await acoustic_pipeline.process_inbound_packet(spoken_packet)
        self.assertTrue(is_final)
        self.assertIn("1,450", transcript)
        await acoustic_pipeline.stop()

        metrics.record_acoustic_turn(
            turn_latency_ms=acoustic_pipeline.telemetry.turn_taking_latency_ms,
            flush_latency_ms=acoustic_pipeline.telemetry.interruption_flush_latency_ms,
            partial_count=1,
            tool_call_count=0,
            correlation_id="corr_e2e_1",
        )

        # Step 2: Context Graph Entity Normalization & Calibrated Decision Routing
        graph = EntityEventGraph(graph_id="e2e_graph")
        graph_builder = CalibratedGraphBuilder(graph=graph, confidence_threshold=0.85)

        mention_income = EntityMention(
            text="Monthly Gross Wages",
            entity_type="income",
            source_doc_id="hearing_transcript",
            attributes={"amount": 1450},
        )
        mention_statute = EntityMention(
            text="SNAP Maximum Gross Income Limit",
            entity_type="statute",
            source_doc_id="statutory_db",
            attributes={"limit": 1580},
        )
        res_income = graph_builder.normalize_entity(mention_income)
        res_statute = graph_builder.normalize_entity(mention_statute)

        # Propose relationship edge
        cand = EdgeCandidate(
            source_id=res_income.resolved_id,
            target_id=res_statute.resolved_id,
            features=EdgeFeatures(
                source_entity_id=res_income.resolved_id,
                target_entity_id=res_statute.resolved_id,
                source_text=res_income.canonical_name,
                target_text=res_statute.canonical_name,
                subsumption_detected=True,
                semantic_similarity=0.88,
            ),
        )
        tx = graph_builder.propose_and_route_edge(cand)
        self.assertTrue(tx.committed)
        self.assertEqual(tx.edge_decision.edge_class, EdgeClass.Extends)

        metrics.record_context_edge_decision(
            latency_ms=tx.latency_ms,
            confidence=tx.edge_decision.confidence,
            escalated=tx.escalated,
            edge_class=tx.edge_decision.edge_class.value,
            correlation_id="corr_e2e_1",
        )

        # Step 3: Calibrated Working Memory Management & Sliding Window Eviction
        context_mgr = ProactiveContextManager(
            total_budget=500,
            window_size=300,
            eviction_policy="calibrated_kernel",
        )
        # Register nodes into memory
        node_wage = ContextNode(
            node_id=res_income.resolved_id,
            text="Monthly wages $1,450 verified",
            token_count=80,
            created_step=1,
            last_accessed_step=1,
            task_relevance=0.95,
        )
        node_statute = ContextNode(
            node_id=res_statute.resolved_id,
            text="SNAP 2026 gross limit $1,580",
            token_count=100,
            created_step=1,
            last_accessed_step=1,
            task_relevance=0.90,
        )
        node_aside = ContextNode(
            node_id="casual_weather_aside",
            text="Casual conversation regarding public transportation commute",
            token_count=180,
            created_step=1,
            last_accessed_step=1,
            task_relevance=0.05,
        )

        context_mgr.register_node(node_wage)
        context_mgr.register_node(node_statute)
        context_mgr.register_node(node_aside)
        context_mgr.advance_step()

        # Total tokens = 80 + 100 + 180 = 360 > 300 target window
        evictions = context_mgr.evict_to_budget(target_budget=300)
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].node_id, "casual_weather_aside")
        self.assertFalse(node_wage.is_evicted)
        self.assertFalse(node_statute.is_evicted)

        mem_tel = context_mgr.get_memory_telemetry()
        metrics.record_memory_eviction(
            node_count=mem_tel["node_count"],
            token_estimate=mem_tel["token_estimate"],
            eviction_count=mem_tel["eviction_count"],
            average_utility=mem_tel["average_node_utility_score"],
            correlation_id="corr_e2e_1",
        )

        # Step 4: Dynamic Adapter Routing (MoVA)
        adapter_router = DynamicMoVARouter()
        # Query is pure text -> direct trunk route
        text_query = InputContextFeatures(text="Check statutory eligibility formula")
        routing_dec = adapter_router.route(text_query)
        self.assertEqual(routing_dec.modality, InputModality.UNIMODAL_TEXT)
        self.assertTrue(routing_dec.bypass_mova)
        self.assertEqual(len(routing_dec.active_experts), 0)

        metrics.record_adapter_routing(
            active_experts=len(routing_dec.active_experts),
            routing_latency_ms=routing_dec.total_routing_latency_ms,
            estimated_kv_cache_mb=routing_dec.estimated_kv_cache_mb,
            cross_modal_interference=routing_dec.cross_modal_interference_proxy,
            correlation_id="corr_e2e_1",
        )

        # Step 5: Sandboxed Shell Execution
        policy = SandboxPolicy(allow_shell=True, allow_network=False)
        sandbox = ShellSandboxExecutor(
            policy=policy,
            mode=RuntimeMode.SHELL_FIRST,
            workspace_dir=self.tmp_dir.name,
        )

        # Execute deterministic Python calculation in shell
        calc_cmd = "python3 -c 'income=1450; limit=1580; print(f\"ELIGIBLE:{income <= limit}\")'"
        sand_res = sandbox.execute_shell(calc_cmd)

        self.assertEqual(sand_res.exit_code, 0)
        self.assertIn("ELIGIBLE:True", sand_res.stdout)
        self.assertEqual(sand_res.isolation_mode, "shell_first")

        tokens_consumed = sand_res.metadata.get("tokens_consumed", 20)
        metrics.record_sandbox_execution(
            tokens_consumed=tokens_consumed,
            exploit_trapped=False,
            correlation_id="corr_e2e_1",
        )

        # Verify final consolidated telemetry snapshot
        snapshot = metrics.get_snapshot()
        self.assertGreater(snapshot.edge_decision_confidence, 0.80)
        self.assertEqual(snapshot.memory_eviction_count, 1)
        self.assertGreater(snapshot.sandbox_shell_tokens, 0)


def test_async_end_to_end_runner() -> None:
    test_case = TestRoadmapEndToEndIntegration()
    test_case.setUp()
    try:
        asyncio.run(test_case.test_full_roadmap_pipeline_flow())
    finally:
        test_case.tearDown()


if __name__ == "__main__":
    unittest.main()
