"""Commercial Proprietary API Provider Adapter.

Supports proprietary model APIs:
- Anthropic (Claude 3.5 Sonnet / Opus tier)
- OpenAI (GPT-4o, o3, etc.)
- DeepSeek (DeepSeek-V4.1-Flash, DeepSeek-V4-Pro with prompt caching headers)
- Google Gemini (Gemini Flash, Pro)
- xAI / Grok

Adheres strictly to the constraint: credentials loaded ONLY from environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import (
    CapabilityTag,
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
    ProviderAuthError,
    ProviderConfigurationError,
    ProviderError,
    ProviderMetadata,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUsage,
)

logger = logging.getLogger(__name__)


class CommercialAPIProvider(InferenceProvider):
    """Adapter for proprietary commercial API endpoints."""

    def __init__(
        self,
        service: str = "openai",
        base_url: str | None = None,
        api_key_env_var: str | None = None,
        default_model: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        enable_prompt_caching: bool = True,
    ) -> None:
        self._service = service.lower()
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._enable_prompt_caching = enable_prompt_caching

        # Standard defaults per service
        if self._service == "anthropic":
            self._base_url = base_url or "https://api.anthropic.com/v1"
            self._env_var = api_key_env_var or "ANTHROPIC_API_KEY"
            self._default_model = default_model or "claude-3-5-sonnet-20241022"
        elif self._service == "deepseek":
            self._base_url = base_url or "https://api.deepseek.com/v1"
            self._env_var = api_key_env_var or "DEEPSEEK_API_KEY"
            self._default_model = default_model or "deepseek-chat"
        elif self._service == "gemini":
            self._base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            self._env_var = api_key_env_var or "GEMINI_API_KEY"
            self._default_model = default_model or "gemini-2.0-flash"
        elif self._service == "grok" or self._service == "xai":
            self._base_url = base_url or "https://api.x.ai/v1"
            self._env_var = api_key_env_var or "XAI_API_KEY"
            self._default_model = default_model or "grok-4.6"
        else:  # default openai
            self._service = "openai"
            self._base_url = base_url or "https://api.openai.com/v1"
            self._env_var = api_key_env_var or "OPENAI_API_KEY"
            self._default_model = default_model or "gpt-4o"

        self._base_url = self._base_url.rstrip("/")

        self._metadata = ProviderMetadata(
            provider_id=f"commercial:{self._service}",
            name=f"Commercial API ({self._service.title()})",
            supported_models=[self._default_model],
            capabilities=[
                CapabilityTag.STREAMING,
                CapabilityTag.FUNCTION_CALLING,
                CapabilityTag.REASONING,
                CapabilityTag.PROMPT_CACHING,
            ],
            is_local=False,
            default_timeout_s=timeout_s,
            max_retries=max_retries,
        )

    @property
    def provider_id(self) -> str:
        return f"commercial:{self._service}"

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def _get_api_key(self) -> str:
        key = os.getenv(self._env_var, "")
        return key.strip()

    def _http_request(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
        model: str,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401 or exc.code == 403:
                raise ProviderAuthError(f"Authentication failed ({exc.code}): {err_body}", self.provider_id, model) from exc
            if exc.code == 429:
                raise ProviderRateLimitError(f"Rate limited ({exc.code}): {err_body}", self.provider_id, model) from exc
            if exc.code >= 500:
                raise ProviderUnavailableError(f"Service error ({exc.code}): {err_body}", self.provider_id, model) from exc
            raise ProviderConfigurationError(f"Bad request ({exc.code}): {err_body}", self.provider_id, model) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeoutError(f"Request timeout after {timeout_s}s", self.provider_id, model) from exc
            raise ProviderUnavailableError(f"Connection failed: {exc.reason}", self.provider_id, model) from exc
        except Exception as exc:
            raise ProviderError(f"Request execution failed: {exc}", self.provider_id, model, retryable=True) from exc

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        start_t = time.perf_counter()
        timeout_s = request.timeout_s or self._timeout_s
        api_key = self._get_api_key()
        model = request.model or self._default_model

        if not api_key:
            # If no API key is present in environment, raise clear ProviderAuthError (never hardcode fallback credentials)
            raise ProviderAuthError(
                f"Missing API key for {self._service} (environment variable '{self._env_var}' not set)",
                self.provider_id,
                model,
            )

        if self._service == "anthropic":
            return self._complete_anthropic(request, api_key, model, timeout_s, start_t)
        else:
            return self._complete_openai_style(request, api_key, model, timeout_s, start_t)

    def _complete_openai_style(
        self,
        request: InferenceRequest,
        api_key: str,
        model: str,
        timeout_s: float,
        start_t: float,
    ) -> InferenceResponse:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # DeepSeek prompt caching optimization
        if self._service == "deepseek" and self._enable_prompt_caching:
            headers["X-Prompt-Cache"] = "enable"

        messages = list(request.messages)
        if request.system_prompt:
            if not (messages and messages[0].get("role") == "system"):
                messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.tools:
            payload["tools"] = request.tools

        raw = self._http_request(url, payload, headers, timeout_s, model)

        choices = raw.get("choices", [])
        if not choices:
            raise ProviderError("Empty choices from commercial API", self.provider_id, model)

        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls") or []

        raw_usage = raw.get("usage", {})
        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
        cached_tokens = (
            raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or raw_usage.get("cached_tokens", 0)
        )

        usage = ProviderUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return InferenceResponse(
            text=text,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=first_choice.get("finish_reason", "stop"),
            model=model,
            provider_id=self.provider_id,
            duration_ms=duration_ms,
            cost_usd=self.estimate_cost(request, usage),
        )

    def _complete_anthropic(
        self,
        request: InferenceRequest,
        api_key: str,
        model: str,
        timeout_s: float,
        start_t: float,
    ) -> InferenceResponse:
        url = f"{self._base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        # Filter messages for user/assistant roles
        anthropic_msgs = []
        for m in request.messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                anthropic_msgs.append({"role": role, "content": m.get("content", "")})

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        raw = self._http_request(url, payload, headers, timeout_s, model)

        content_blocks = raw.get("content", [])
        text = ""
        for b in content_blocks:
            if b.get("type") == "text":
                text += b.get("text", "")

        raw_usage = raw.get("usage", {})
        prompt_tokens = raw_usage.get("input_tokens", 0)
        completion_tokens = raw_usage.get("output_tokens", 0)
        cached_tokens = raw_usage.get("cache_read_input_tokens", 0)

        usage = ProviderUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return InferenceResponse(
            text=text,
            usage=usage,
            finish_reason=raw.get("stop_reason", "end_turn"),
            model=model,
            provider_id=self.provider_id,
            duration_ms=duration_ms,
            cost_usd=self.estimate_cost(request, usage),
        )

    def check_health(self) -> bool:
        # Fast health check: if API key is missing, report false, else true
        return bool(self._get_api_key())
