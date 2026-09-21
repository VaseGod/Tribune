"""Unit tests for MoVAdapterPipeline dynamic selection, KV-cache constraints, and pruning."""

from __future__ import annotations

import unittest

from tribune.adapters.experts import ExpertType
from tribune.adapters.gating import InputContextFeatures
from tribune.adapters.mova import MoVAdapterPipeline


class TestMoVAdapterPipeline(unittest.TestCase):
    def test_dynamic_selection_within_budget(self) -> None:
        """Pipeline selects up to max_active_experts based on highest affinity."""
        pipeline = MoVAdapterPipeline(
            max_active_experts=2,
            cache_budget_mb=1024.0,
        )
        context = InputContextFeatures(
            text="Verify claimant oral testimony and attached pay stub layout",
            has_audio=True,
            audio_duration_s=30.0,
            has_image=True,
            image_count=1,
        )
        decision = pipeline.select_experts(context)

        self.assertLessEqual(len(decision.active_experts), 2)
        self.assertIn(ExpertType.ACOUSTIC_STREAM, decision.active_experts)
        self.assertIn(ExpertType.VISION_DINOV2, decision.active_experts)
        self.assertLessEqual(decision.estimated_kv_cache_mb, 1024.0)

    def test_kv_cache_budget_pruning(self) -> None:
        """When cache budget is constrained, lower-affinity experts are pruned."""
        # Budget only enough for one expert (~200MB)
        pipeline = MoVAdapterPipeline(
            max_active_experts=3,
            cache_budget_mb=200.0,
        )
        context = InputContextFeatures(
            text="Review image document and audio",
            has_audio=True,
            audio_duration_s=20.0,
            has_image=True,
            image_count=1,
        )
        decision = pipeline.select_experts(context)

        # Must have stayed strictly within the 200MB budget
        self.assertLessEqual(decision.estimated_kv_cache_mb, 200.0)
        self.assertLessEqual(len(decision.active_experts), 1)

    def test_cross_modal_interference_proxy(self) -> None:
        """Interference is 0.0 for unimodal/single-expert and moderate for dual-expert."""
        pipeline = MoVAdapterPipeline(max_active_experts=2)

        # Audio-only -> 1 expert -> 0.0 interference
        ctx_single = InputContextFeatures(text="", has_audio=True, audio_duration_s=10.0)
        dec_single = pipeline.select_experts(ctx_single)
        self.assertEqual(dec_single.cross_modal_interference_proxy, 0.0)

        # Mixed multimodal -> 2 experts -> low calibrated interference (~0.12)
        ctx_mixed = InputContextFeatures(
            text="Examine wage document table",
            has_audio=True,
            audio_duration_s=10.0,
            has_image=True,
        )
        dec_mixed = pipeline.select_experts(ctx_mixed)
        if len(dec_mixed.active_experts) == 2:
            self.assertEqual(dec_mixed.cross_modal_interference_proxy, 0.12)


if __name__ == "__main__":
    unittest.main()
