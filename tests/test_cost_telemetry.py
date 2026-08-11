"""Unit tests for Real-Time Cost & Token Telemetry Middleware."""

from __future__ import annotations

import unittest

from tribune.config import TribuneSettings
from tribune.instrumentation.telemetry import (
    CostTelemetryMiddleware,
    TelemetryMetricsStore,
    TelemetryRecord,
)
from tribune.providers.llm_client import LLMCompletionRequest, LocalRulesLLMAdapter


class TestCostTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TelemetryMetricsStore()
        self.store.clear()

    def test_record_call_and_pricing_calculation(self) -> None:
        rec = self.store.record_call(
            provider_name="deepseek:deepseek-v4-flash",
            model="deepseek-v4-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached_tokens=0,
            latency_ms=450.0,
            role="coder",
        )
        self.assertIsInstance(rec, TelemetryRecord)
        # DeepSeek-V4-Flash rates in pricing.json: input 0.14/M, output 0.28/M -> Total 0.42 USD
        self.assertAlmostEqual(rec.cost_usd, 0.42, places=4)
        self.assertEqual(rec.input_tokens, 1_000_000)
        self.assertEqual(rec.output_tokens, 1_000_000)
        self.assertEqual(rec.latency_ms, 450.0)

    def test_telemetry_middleware_interception(self) -> None:
        settings = TribuneSettings()
        raw_adapter = LocalRulesLLMAdapter(settings)
        wrapped_provider = CostTelemetryMiddleware(raw_adapter, store=self.store, role="verifier")

        req = LLMCompletionRequest(
            system_prompt="Verifier assessment",
            messages=[{"role": "user", "content": "Verify rules"}],
        )
        resp = wrapped_provider.complete(req)

        self.assertEqual(len(self.store.records), 1)
        rec = self.store.records[0]
        self.assertEqual(rec.provider_name, resp.provider_name)
        self.assertEqual(rec.role, "verifier")
        self.assertGreater(rec.input_tokens, 0)

    def test_get_summary_metrics(self) -> None:
        self.store.record_call("anthropic:claude-3-5-sonnet", "claude-3-5-sonnet", 100, 200, latency_ms=150.0)
        self.store.record_call("deepseek:v4-flash", "deepseek-v4-flash", 500, 300, latency_ms=250.0)

        summary = self.store.get_summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_input_tokens"], 600)
        self.assertEqual(summary["total_output_tokens"], 500)
        self.assertEqual(summary["avg_latency_ms"], 200.0)
        self.assertIn("anthropic:claude-3-5-sonnet", summary["by_provider"])
        self.assertIn("deepseek:v4-flash", summary["by_provider"])


if __name__ == "__main__":
    unittest.main()
