"""Tests for the unified inference provider layer."""

import pytest
from tribune.inference.base import (
    CapabilityTag,
    InferenceRequest,
    InferenceResponse,
    ProviderAuthError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUsage,
)
from tribune.inference.commercial import CommercialAPIProvider
from tribune.inference.costs import calculate_token_cost
from tribune.inference.openai_compatible import OpenAICompatibleProvider
from tribune.inference.registry import (
    InferenceProviderRegistry,
    MockOfflineProvider,
    get_inference_registry,
    reset_inference_registry,
)


def test_mock_offline_provider():
    provider = MockOfflineProvider(canned_response="The claimant is eligible for SNAP.")
    assert provider.provider_id == "mock_offline"
    assert provider.metadata.is_local is True

    req = InferenceRequest(messages=[{"role": "user", "content": "Check eligibility."}])
    resp = provider.complete(req)

    assert resp.text == "The claimant is eligible for SNAP."
    assert resp.usage.total_tokens > 0
    assert resp.finish_reason == "stop"


def test_bounded_retries_non_recursive():
    class FlakyProvider(MockOfflineProvider):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def complete(self, request):
            self.attempts += 1
            if self.attempts < 2:
                raise ProviderTimeoutError("Temporary timeout", "flaky", "mock")
            return super().complete(request)

    provider = FlakyProvider()
    req = InferenceRequest(messages=[{"role": "user", "content": "Retry test."}])
    resp = provider.execute_with_bounded_retries(req, max_retries=2, base_backoff_s=0.01)

    assert resp is not None
    assert provider.attempts == 2


def test_bounded_retries_fail_closed_on_permanent_error():
    class UnrecoverableProvider(MockOfflineProvider):
        def complete(self, request):
            raise ProviderAuthError("Invalid API key", "auth_fail", "mock")

    provider = UnrecoverableProvider()
    req = InferenceRequest(messages=[{"role": "user", "content": "Auth test."}])

    with pytest.raises(ProviderAuthError):
        provider.execute_with_bounded_retries(req, max_retries=3)


def test_mid_run_model_switching():
    reset_inference_registry()
    registry = get_inference_registry()

    p1 = MockOfflineProvider(provider_id="mock_p1", canned_response="Response from P1")
    p2 = MockOfflineProvider(provider_id="mock_p2", canned_response="Response from P2")

    registry.register_provider("mock_p1", p1)
    registry.register_provider("mock_p2", p2)

    # Initial call with default
    registry.switch_tier("worker")
    p, model = registry.get_active_model_and_provider()
    assert model == "deepseek-v4.1-flash"

    # Switch model mid-run
    registry.switch_model("claude-opus-test", provider_id="mock_p2")
    active_p, active_model = registry.get_active_model_and_provider()

    assert active_model == "claude-opus-test"
    assert active_p.provider_id == "mock_p2"

    req = InferenceRequest(messages=[{"role": "user", "content": "Hello"}])
    resp = registry.complete(req)
    assert resp.text == "Response from P2"


def test_token_cost_calculation():
    usage = ProviderUsage(prompt_tokens=10_000, completion_tokens=2_000, cached_tokens=8_000)
    # DeepSeek flash rates: $0.30 uncached / $0.006 cached / $1.20 output per 1M
    est = calculate_token_cost("deepseek-flash", usage)

    # Uncached: (10k - 8k) * 0.30/1M = 2,000 * 0.00000030 = 0.0006
    # Cached: 8,000 * 0.006/1M = 8,000 * 0.000000006 = 0.000048
    # Output: 2,000 * 1.20/1M = 2,000 * 0.00000120 = 0.0024
    assert est.uncached_input_usd == pytest.approx(0.0006, abs=1e-5)
    assert est.cached_input_usd == pytest.approx(0.000048, abs=1e-5)
    assert est.output_usd == pytest.approx(0.0024, abs=1e-5)
    assert est.total_cost_usd > 0.0
    assert est.cache_savings_usd > 0.0
