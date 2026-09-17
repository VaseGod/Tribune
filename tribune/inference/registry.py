"""Inference Provider Registry & Dynamic Model Switching Engine.

Manages registered providers, tracks active model tiers (lead vs worker),
and supports dynamic mid-evaluation model switching through runtime controls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
    ProviderConfigurationError,
    ProviderMetadata,
    ProviderUsage,
)
from .commercial import CommercialAPIProvider
from .config import InferenceConfig
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class MockOfflineProvider(InferenceProvider):
    """Deterministic offline provider for testing and zero-cost simulation."""

    def __init__(
        self,
        provider_id: str = "mock_offline",
        canned_response: str = "Mock deterministic completion.",
    ) -> None:
        self._provider_id = provider_id
        self.canned_response = canned_response
        self.call_history: list[InferenceRequest] = []
        self._metadata = ProviderMetadata(
            provider_id=self._provider_id,
            name="Mock Offline Provider",
            supported_models=["mock-model-v1", "swift-qwen3.8-27b", "deepseek-v4.1-flash", "gpt-4o"],
            is_local=True,
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.call_history.append(request)
        model = request.model or "mock-model-v1"
        prompt_chars = sum(len(str(m.get("content", ""))) for m in request.messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(self.canned_response) // 4)

        return InferenceResponse(
            text=self.canned_response,
            usage=ProviderUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=0,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
            model=model,
            provider_id=self._provider_id,
            duration_ms=5.0,
            cost_usd=0.0,
        )


class InferenceProviderRegistry:
    """Registry and runtime dispatcher for inference providers."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig.load_from_env_and_file()
        self._providers: dict[str, InferenceProvider] = {}
        self._factories: dict[str, Callable[[], InferenceProvider]] = {}
        self._active_provider_id: str | None = None
        self._active_tier: str = self.config.active_tier
        self._model_override: str | None = None

        # Pre-register standard providers
        self.register_provider("mock_offline", MockOfflineProvider())
        self.register_provider("local_rules", MockOfflineProvider(provider_id="local_rules"))

    def register_provider(self, provider_id: str, provider: InferenceProvider) -> None:
        """Register an initialized provider instance."""
        self._providers[provider_id] = provider
        if self._active_provider_id is None:
            self._active_provider_id = provider_id

    def register_factory(self, provider_id: str, factory: Callable[[], InferenceProvider]) -> None:
        """Register a lazy provider factory."""
        self._factories[provider_id] = factory

    def get_provider(self, provider_id: str | None = None) -> InferenceProvider:
        """Resolve a provider by ID, or return active provider."""
        target_id = provider_id or self._active_provider_id
        if not target_id:
            # Fall back to mock offline
            return self._providers["mock_offline"]

        if target_id in self._providers:
            return self._providers[target_id]

        if target_id in self._factories:
            provider = self._factories[target_id]()
            self._providers[target_id] = provider
            return provider

        # Attempt auto-provisioning
        if target_id == "openai_compatible" or target_id == "vllm":
            p = OpenAICompatibleProvider(provider_id=target_id)
            self.register_provider(target_id, p)
            return p
        elif target_id.startswith("commercial:") or target_id in ("openai", "anthropic", "deepseek", "gemini", "grok"):
            service = target_id.split(":")[-1]
            p = CommercialAPIProvider(service=service)
            self.register_provider(target_id, p)
            return p

        raise ProviderConfigurationError(f"Provider '{target_id}' not found in registry")

    # ----------------------------------------------------------------------- #
    # Mid-Evaluation Model and Tier Switching Controls
    # ----------------------------------------------------------------------- #

    def switch_tier(self, tier: str) -> None:
        """Switch active tier between 'lead' and 'worker'."""
        if tier not in ("lead", "worker"):
            raise ProviderConfigurationError(f"Invalid tier '{tier}'. Must be 'lead' or 'worker'")
        self._active_tier = tier
        tier_cfg = self.config.lead_tier if tier == "lead" else self.config.worker_tier
        self._active_provider_id = tier_cfg.provider_id
        self._model_override = None
        logger.info(f"[InferenceRegistry] Switched to {tier} tier (Model: {tier_cfg.model})")

    def switch_model(self, model_name: str, provider_id: str | None = None) -> None:
        """Perform mid-evaluation model switching."""
        self._model_override = model_name
        if provider_id:
            self._active_provider_id = provider_id
        logger.info(f"[InferenceRegistry] Mid-evaluation model switched to '{model_name}' on provider '{self._active_provider_id}'")

    def reset_override(self) -> None:
        self._model_override = None

    def get_active_model_and_provider(self) -> tuple[InferenceProvider, str]:
        """Return the active provider and target model based on tier and overrides."""
        tier_cfg = self.config.lead_tier if self._active_tier == "lead" else self.config.worker_tier
        target_provider_id = self._active_provider_id or tier_cfg.provider_id
        provider = self.get_provider(target_provider_id)
        model = self._model_override or tier_cfg.model
        return provider, model

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Convenience method to dispatch completion using active tier settings."""
        provider, model = self.get_active_model_and_provider()
        req_copy = InferenceRequest(
            messages=request.messages,
            model=request.model or model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            tools=request.tools,
            stream=request.stream,
            response_format=request.response_format,
            budget_tokens=request.budget_tokens,
            timeout_s=request.timeout_s,
            metadata={**request.metadata, "tier": self._active_tier},
        )
        return provider.complete(req_copy)


# Global default singleton registry
_GLOBAL_REGISTRY: InferenceProviderRegistry | None = None


def get_inference_registry() -> InferenceProviderRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = InferenceProviderRegistry()
    return _GLOBAL_REGISTRY


def reset_inference_registry() -> None:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None
