"""Unified Inference Provider Layer — Base Interfaces & Data Structures.

Provides a decoupled, typed interface for model invocation supporting:
1. Open-weight backends (vLLM, SGLang, local llama.cpp).
2. Proprietary commercial APIs (Anthropic, OpenAI, DeepSeek, Gemini, Grok).
3. Capability tagging, model metadata, token accounting, and cost estimation.
4. Deterministic timeouts and bounded retries (strictly non-recursive).
5. Mid-evaluation model switching and health monitoring.
"""

from __future__ import annotations

import abc
import enum
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityTag(str, enum.Enum):
    """Declared capabilities of an inference provider or backend."""

    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"
    PROMPT_CACHING = "prompt_caching"
    BATCH = "batch"
    LOCAL = "local"
    OFFLINE = "offline"


@dataclass(frozen=True)
class ProviderMetadata:
    """Metadata describing a provider implementation and its operational limits."""

    provider_id: str
    name: str
    version: str = "1.0.0"
    supported_models: list[str] = field(default_factory=list)
    capabilities: list[CapabilityTag] = field(default_factory=list)
    is_local: bool = False
    default_timeout_s: float = 60.0
    max_retries: int = 3


@dataclass
class ProviderUsage:
    """Standardized token usage accounting across all provider backends."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit percentage for cached-token architectures."""
        if self.prompt_tokens <= 0:
            return 0.0
        return min(1.0, self.cached_tokens / max(1, self.prompt_tokens + self.cached_tokens))


@dataclass
class InferenceRequest:
    """Unified request model for chat completions and text generation."""

    messages: list[dict[str, Any]]
    model: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = False
    response_format: str | dict[str, Any] | None = None
    budget_tokens: int | None = None
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Unified response model containing generated output, usage, and telemetry."""

    text: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    finish_reason: str = "stop"
    model: str = ""
    provider_id: str = ""
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Error Hierarchy
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """Base exception for all inference provider errors."""

    def __init__(self, message: str, provider_id: str = "", model: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.model = model
        self.retryable = retryable


class ProviderTimeoutError(ProviderError):
    """Raised when an inference request exceeds its designated deadline."""

    def __init__(self, message: str, provider_id: str = "", model: str = "") -> None:
        super().__init__(message, provider_id=provider_id, model=model, retryable=True)


class ProviderRateLimitError(ProviderError):
    """Raised when request quota or rate limits are breached."""

    def __init__(self, message: str, provider_id: str = "", model: str = "", retry_after_s: float | None = None) -> None:
        super().__init__(message, provider_id=provider_id, model=model, retryable=True)
        self.retry_after_s = retry_after_s


class ProviderAuthError(ProviderError):
    """Raised when credentials, tokens, or permissions are rejected."""

    def __init__(self, message: str, provider_id: str = "", model: str = "") -> None:
        super().__init__(message, provider_id=provider_id, model=model, retryable=False)


class ProviderUnavailableError(ProviderError):
    """Raised when endpoint is unreachable, down, or returning 5xx status."""

    def __init__(self, message: str, provider_id: str = "", model: str = "") -> None:
        super().__init__(message, provider_id=provider_id, model=model, retryable=True)


class ProviderConfigurationError(ProviderError):
    """Raised for malformed parameters, missing models, or invalid routing."""

    def __init__(self, message: str, provider_id: str = "", model: str = "") -> None:
        super().__init__(message, provider_id=provider_id, model=model, retryable=False)


# --------------------------------------------------------------------------- #
# Abstract Base Provider
# --------------------------------------------------------------------------- #


class InferenceProvider(abc.ABC):
    """Abstract interface for all model providers within Tribune."""

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for the provider (e.g. 'openai_compatible', 'anthropic')."""
        ...

    @property
    @abc.abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Capabilities and limits of this provider."""
        ...

    @abc.abstractmethod
    def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a completion synchronously."""
        ...

    def stream(self, request: InferenceRequest) -> Iterator[InferenceResponse]:
        """Stream chunks synchronously. Default falls back to single-shot complete."""
        yield self.complete(request)

    def check_health(self) -> bool:
        """Verify endpoint availability and connectivity."""
        return True

    def estimate_cost(self, request: InferenceRequest, usage: ProviderUsage | None = None) -> float:
        """Hook to estimate financial cost of a request given token counts."""
        return 0.0

    def execute_with_bounded_retries(
        self,
        request: InferenceRequest,
        max_retries: int | None = None,
        base_backoff_s: float = 0.5,
    ) -> InferenceResponse:
        """Execute inference with explicit, bounded retries.

        Constraint: Strictly prevents recursive model retry loops. Retries are
        deterministic, finite, and logged.
        """
        retries = max_retries if max_retries is not None else self.metadata.max_retries
        attempts = 0
        last_error: Exception | None = None

        while attempts <= retries:
            try:
                return self.complete(request)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempts >= retries:
                    logger.error(
                        f"[{self.provider_id}] Inference failed permanently after {attempts} retries: {exc}"
                    )
                    raise
                backoff = base_backoff_s * (2 ** attempts)
                logger.warning(
                    f"[{self.provider_id}] Transient failure (attempt {attempts + 1}/{retries + 1}): {exc}. "
                    f"Backing off {backoff:.2f}s."
                )
                time.sleep(backoff)
                attempts += 1
            except Exception as exc:
                logger.critical(f"[{self.provider_id}] Unexpected non-provider error during inference: {exc}")
                raise ProviderError(
                    f"Unexpected error in {self.provider_id}: {exc}",
                    provider_id=self.provider_id,
                    model=request.model or "",
                    retryable=False,
                ) from exc

        if last_error:
            raise last_error
        raise ProviderError(f"Exhausted retries ({retries}) without response", provider_id=self.provider_id)
