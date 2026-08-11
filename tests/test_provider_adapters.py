"""Unit tests for Provider Adapters (OpenAI, Anthropic, DeepSeek, vLLM, LocalRules)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tribune.config import TribuneSettings
from tribune.providers.llm_client import (
    AnthropicProviderAdapter,
    DeepSeekProviderAdapter,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LocalRulesLLMAdapter,
    OpenAIProviderAdapter,
    ProviderAPIError,
    get_llm_provider,
    vLLMProviderAdapter,
)


class TestProviderAdapters(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TribuneSettings(
            provider="local_rules",
            openai_api_key="mock-key",
            anthropic_api_key="mock-key",
            deepseek_api_key="mock-key",
            deepseek_reasoning_effort="high",
        )

    def test_factory_lookup(self) -> None:
        p_local = get_llm_provider("local_rules", self.settings)
        self.assertIsInstance(p_local, LocalRulesLLMAdapter)

        p_openai = get_llm_provider("openai", self.settings)
        self.assertIsInstance(p_openai, OpenAIProviderAdapter)

        p_anthropic = get_llm_provider("anthropic", self.settings)
        self.assertIsInstance(p_anthropic, AnthropicProviderAdapter)

        p_deepseek = get_llm_provider("deepseek", self.settings)
        self.assertIsInstance(p_deepseek, DeepSeekProviderAdapter)

        p_vllm = get_llm_provider("vllm", self.settings)
        self.assertIsInstance(p_vllm, vLLMProviderAdapter)

    def test_local_rules_adapter_completion(self) -> None:
        adapter = LocalRulesLLMAdapter(self.settings)
        req = LLMCompletionRequest(
            system_prompt="Calculate status",
            messages=[{"role": "user", "content": "Check eligibility"}],
        )
        resp = adapter.complete(req)
        self.assertIsInstance(resp, LLMCompletionResponse)
        self.assertIn("status", resp.content)
        self.assertGreater(resp.input_tokens, 0)
        self.assertGreater(resp.output_tokens, 0)

    @patch("tribune.providers.llm_client.BaseHTTPProviderAdapter._post_json")
    def test_deepseek_adapter_reasoning_effort(self, mock_post: MagicMock) -> None:
        mock_post.return_value = (
            {
                "choices": [{"message": {"content": "DeepSeek response"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 25},
            },
            120.5,
        )
        adapter = DeepSeekProviderAdapter(self.settings, model="deepseek-v4-flash")
        req = LLMCompletionRequest(
            messages=[{"role": "user", "content": "Refactor code"}],
            reasoning_effort="low",
        )
        resp = adapter.complete(req)

        mock_post.assert_called_once()
        url, payload, headers = mock_post.call_args[0]
        self.assertIn("chat/completions", url)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(resp.content, "DeepSeek response")
        self.assertEqual(resp.input_tokens, 15)
        self.assertEqual(resp.output_tokens, 25)

    @patch("tribune.providers.llm_client.BaseHTTPProviderAdapter._post_json")
    def test_anthropic_adapter_formatting(self, mock_post: MagicMock) -> None:
        mock_post.return_value = (
            {
                "content": [{"type": "text", "text": "Claude response"}],
                "usage": {"input_tokens": 20, "output_tokens": 30},
            },
            250.0,
        )
        adapter = AnthropicProviderAdapter(self.settings, model="claude-3-5-sonnet-20241022")
        req = LLMCompletionRequest(
            system_prompt="Security auditor",
            messages=[{"role": "user", "content": "Audit payload"}],
        )
        resp = adapter.complete(req)

        url, payload, headers = mock_post.call_args[0]
        self.assertIn("messages", url)
        self.assertEqual(payload["system"], "Security auditor")
        self.assertEqual(headers["x-api-key"], "mock-key")
        self.assertEqual(resp.content, "Claude response")
        self.assertEqual(resp.input_tokens, 20)

    def test_provider_api_error_properties(self) -> None:
        err_429 = ProviderAPIError("Rate limit reached", status_code=429)
        self.assertTrue(err_429.is_rate_limit)
        self.assertEqual(err_429.status_code, 429)

        err_500 = ProviderAPIError("Internal server error", status_code=500)
        self.assertFalse(err_500.is_rate_limit)


if __name__ == "__main__":
    unittest.main()
