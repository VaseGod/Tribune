"""Unit tests for DynamicMoVARouter coarse-to-fine gating and modality-specific activation."""

from __future__ import annotations

import unittest

from tribune.adapters.experts import ExpertType
from tribune.adapters.gating import InputContextFeatures, InputModality
from tribune.adapters.router import DynamicMoVARouter


class TestDynamicMoVARouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = DynamicMoVARouter(
            max_active_experts=2,
            cache_budget_mb=1024.0,
            enable_dynamic_routing=True,
        )

    def test_pure_text_bypasses_all_adapters(self) -> None:
        """Unimodal text queries must bypass all adapters directly to the LLM trunk (0 experts, 0MB cache)."""
        context = InputContextFeatures(
            text="What is the gross income limit for SNAP in 2026?",
            has_audio=False,
            has_image=False,
            has_structured_graph=False,
        )
        decision = self.router.route(context)

        self.assertEqual(decision.modality, InputModality.UNIMODAL_TEXT)
        self.assertTrue(decision.bypass_mova)
        self.assertEqual(len(decision.active_experts), 0)
        self.assertEqual(decision.estimated_kv_cache_mb, 0.0)
        self.assertEqual(decision.cross_modal_interference_proxy, 0.0)
        self.assertIsNone(decision.fine_decision)

    def test_audio_only_activates_acoustic_expert(self) -> None:
        """Audio-only testimony activates only the acoustic expert."""
        context = InputContextFeatures(
            text="",
            has_audio=True,
            audio_duration_s=45.0,
            has_image=False,
        )
        decision = self.router.route(context)

        self.assertEqual(decision.modality, InputModality.ACOUSTIC_ONLY)
        self.assertFalse(decision.bypass_mova)
        self.assertEqual(decision.active_experts, [ExpertType.ACOUSTIC_STREAM])
        self.assertGreater(decision.estimated_kv_cache_mb, 0.0)
        self.assertEqual(decision.cross_modal_interference_proxy, 0.0)

    def test_vision_layout_activates_dinov2_expert(self) -> None:
        """Image inputs with document layout cues activate Vision Expert A (DINOv2)."""
        context = InputContextFeatures(
            text="Examine wage pay stub layout table and boxes",
            has_audio=False,
            has_image=True,
            image_count=1,
        )
        decision = self.router.route(context)

        self.assertEqual(decision.modality, InputModality.MULTIMODAL_MIXED)
        self.assertIn(ExpertType.VISION_DINOV2, decision.active_experts)
        self.assertNotIn(ExpertType.ACOUSTIC_STREAM, decision.active_experts)

    def test_static_fallback_mode(self) -> None:
        """Fallback static mode activates default adapter set."""
        context = InputContextFeatures(text="Check case eligibility", has_audio=False, has_image=False)
        decision = self.router.route(context, force_static=True)

        self.assertFalse(decision.bypass_mova)
        self.assertGreater(len(decision.active_experts), 1)
        self.assertGreater(decision.cross_modal_interference_proxy, 0.3)
        self.assertTrue(decision.fine_decision.static_fallback_used)

    def test_telemetry_recording(self) -> None:
        """Verify router telemetry tracks active expert counts, latency, and distribution."""
        context = InputContextFeatures(text="Hello", has_audio=False, has_image=False)
        self.router.route(context)
        tel = self.router.get_telemetry()
        self.assertEqual(tel["active_expert_count"], 0)
        self.assertGreaterEqual(tel["routing_latency_ms"], 0.0)
        self.assertEqual(tel["estimated_kv_cache_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
