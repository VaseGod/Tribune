"""Unit tests for Local Provider & Runtime Engine (llama.cpp GGUF backends)."""

from __future__ import annotations

import os
import unittest
import pytest

from tribune.providers.base import SynthesisRequest
from tribune.providers.local_rules import (
    LocalGGUFProvider,
    LocalRulesProvider,
    LocalRuntimeConfig,
    benchmark_local_inference,
    detect_runtime_capabilities,
    estimate_memory_footprint,
)
from tribune.types import CriterionOutcome, CriterionResult, ProgramId


class TestLocalRuntime(unittest.TestCase):
    def test_local_runtime_config_defaults(self) -> None:
        cfg = LocalRuntimeConfig()
        self.assertEqual(cfg.model_family, "qwen3.8-27b")
        self.assertEqual(cfg.quantization, "IQ4_XS")
        self.assertTrue(cfg.flash_attention)
        self.assertEqual(cfg.kv_cache_type, "q4_1")
        self.assertTrue(cfg.speculative_decoding)
        self.assertEqual(cfg.spec_type, "ngram-mod,draft-mtp")
        self.assertEqual(cfg.spec_draft_n_max, 2)
        self.assertEqual(cfg.memory_budget_gb, 16.0)

    def test_detect_runtime_capabilities(self) -> None:
        caps = detect_runtime_capabilities()
        self.assertIn("llama_cpp_installed", caps)
        self.assertIn("gpu_available", caps)
        self.assertIn("flash_attention_supported", caps)
        self.assertIn("kv_quant_supported", caps)
        self.assertIn("speculative_decoding_supported", caps)
        self.assertTrue(caps["kv_quant_supported"])
        self.assertTrue(caps["speculative_decoding_supported"])

    def test_estimate_memory_footprint_iq4_xs_and_q4_k_m(self) -> None:
        # IQ4_XS footprint on 16GB GPU
        cfg_iq4 = LocalRuntimeConfig(quantization="IQ4_XS", kv_cache_type="q4_1", flash_attention=True)
        mem_iq4 = estimate_memory_footprint(cfg_iq4)
        self.assertEqual(mem_iq4["weights_gb"], 13.8)
        self.assertLessEqual(mem_iq4["total_estimated_gb"], 16.0)
        self.assertTrue(mem_iq4["fits_budget"])

        # Q4_K_M footprint on 16GB GPU with q4_1 KV cache
        cfg_q4 = LocalRuntimeConfig(quantization="Q4_K_M", kv_cache_type="q4_1", flash_attention=True)
        mem_q4 = estimate_memory_footprint(cfg_q4)
        self.assertEqual(mem_q4["weights_gb"], 14.8)
        self.assertLessEqual(mem_q4["total_estimated_gb"], 16.8)

    def test_benchmark_local_inference_simulation(self) -> None:
        cfg = LocalRuntimeConfig(
            quantization="IQ4_XS",
            flash_attention=True,
            kv_cache_type="q4_1",
            speculative_decoding=True,
        )
        res = benchmark_local_inference(config=cfg, prompt_tokens=512, gen_tokens=128, mock=True)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["tokens_per_second"], 70.0)  # Targets 70+ tokens/sec
        self.assertIn("memory", res)
        self.assertTrue(res["memory"]["fits_budget"])

    def test_local_gguf_provider_fallback_execution(self) -> None:
        provider = LocalGGUFProvider(quantization="IQ4_XS", role="proposer")
        self.assertEqual(provider.role, "proposer")
        self.assertIn("local_gguf:qwen3.8-27b-iq4_xs", provider.name)

        req = SynthesisRequest(
            program=ProgramId.SNAP,
            jurisdiction="EX",
            criteria=[
                CriterionResult(
                    criterion_id="snap_gross_income",
                    description="Gross income below 130% FPL",
                    outcome=CriterionOutcome.SATISFIED,
                    required=True,
                    citation_ids=["7_CFR_273_9"],
                )
            ],
            required_total=1,
            coverage_complete=True,
            evidence_summary="Income $1200",
            citations=[],
        )
        result = provider.synthesize_assessment(req)
        self.assertEqual(result.status.value, "likely_eligible")
        self.assertGreater(result.self_confidence, 0.5)


@pytest.mark.skipif(
    not os.getenv("TRIBUNE_LOCAL_GGUF_PATH"),
    reason="Skipping heavyweight local GGUF execution test (set TRIBUNE_LOCAL_GGUF_PATH to run)",
)
def test_live_gguf_execution():
    model_path = os.getenv("TRIBUNE_LOCAL_GGUF_PATH", "")
    cfg = LocalRuntimeConfig(model_path=model_path, quantization="IQ4_XS")
    res = benchmark_local_inference(config=cfg, prompt_tokens=128, gen_tokens=32, mock=False)
    assert res["status"] == "success"


if __name__ == "__main__":
    unittest.main()
