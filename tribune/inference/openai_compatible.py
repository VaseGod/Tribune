"""OpenAI-Compatible Inference Provider (vLLM, SGLang, llama.cpp, Local Serving).

Handles standard OpenAI chat completions format over HTTP with:
- Zero heavyweight dependency fallback (uses stdlib urllib if httpx is missing).
- Parsing of reasoning content (DeepSeek/Qwen reasoning fields).
- Token usage extraction and error mapping.
- Health check ping support.
"""

from __future__ import annotations

import json
import logging
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


class OpenAICompatibleProvider(InferenceProvider):
    """Provider for vLLM, SGLang, Ollama, llama.cpp, and OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed-for-local-serving",
        default_model: str = "Qwen/Qwen2.5-7B-Instruct",
        provider_id: str = "openai_compatible",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._provider_id = provider_id
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._extra_headers = extra_headers or {}

        self._metadata = ProviderMetadata(
            provider_id=self._provider_id,
            name="OpenAI-Compatible Provider",
            supported_models=[default_model],
            capabilities=[
                CapabilityTag.STREAMING,
                CapabilityTag.FUNCTION_CALLING,
                CapabilityTag.REASONING,
                CapabilityTag.PROMPT_CACHING,
                CapabilityTag.LOCAL,
            ],
            is_local="localhost" in base_url or "127.0.0.1" in base_url,
            default_timeout_s=timeout_s,
            max_retries=max_retries,
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def _build_payload(self, request: InferenceRequest) -> dict[str, Any]:
        messages = list(request.messages)
        if request.system_prompt:
            # Prepend system prompt if not already first message
            if not (messages and messages[0].get("role") == "system"):
                messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.tools:
            payload["tools"] = request.tools
        if request.response_format:
            if isinstance(request.response_format, dict):
                payload["response_format"] = request.response_format
            elif request.response_format == "json":
                payload["response_format"] = {"type": "json_object"}
        return payload

    def _http_post(self, endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }

        # Try httpx if available for speed and connection pooling
        try:
            import httpx  # type: ignore

            try:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code == 401:
                        raise ProviderAuthError(f"Unauthorized: {resp.text}", self._provider_id, payload.get("model", ""))
                    if resp.status_code == 429:
                        raise ProviderRateLimitError(f"Rate limited: {resp.text}", self._provider_id, payload.get("model", ""))
                    if resp.status_code >= 500:
                        raise ProviderUnavailableError(f"Server error {resp.status_code}: {resp.text}", self._provider_id, payload.get("model", ""))
                    if resp.status_code >= 400:
                        raise ProviderConfigurationError(f"Client error {resp.status_code}: {resp.text}", self._provider_id, payload.get("model", ""))
                    return resp.json()
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(f"Request timed out after {timeout_s}s: {exc}", self._provider_id, payload.get("model", "")) from exc
            except (httpx.ConnectError, httpx.NetworkError) as exc:
                raise ProviderUnavailableError(f"Failed to connect to {url}: {exc}", self._provider_id, payload.get("model", "")) from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"HTTP request failed: {exc}", self._provider_id, payload.get("model", ""), retryable=True) from exc
        except ImportError:
            # Fallback to stdlib urllib
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 401:
                    raise ProviderAuthError(f"Unauthorized: {body}", self._provider_id, payload.get("model", "")) from exc
                if exc.code == 429:
                    raise ProviderRateLimitError(f"Rate limit exceeded: {body}", self._provider_id, payload.get("model", "")) from exc
                if exc.code >= 500:
                    raise ProviderUnavailableError(f"Server error {exc.code}: {body}", self._provider_id, payload.get("model", "")) from exc
                raise ProviderConfigurationError(f"HTTP {exc.code}: {body}", self._provider_id, payload.get("model", "")) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError):
                    raise ProviderTimeoutError(f"Timeout after {timeout_s}s: {exc}", self._provider_id, payload.get("model", "")) from exc
                raise ProviderUnavailableError(f"Endpoint unreachable {url}: {exc}", self._provider_id, payload.get("model", "")) from exc
            except Exception as exc:
                raise ProviderError(f"Request failed: {exc}", self._provider_id, payload.get("model", ""), retryable=True) from exc

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        start_t = time.perf_counter()
        timeout_s = request.timeout_s or self._timeout_s
        payload = self._build_payload(request)
        model = payload.get("model", self._default_model)

        raw = self._http_post("chat/completions", payload, timeout_s)

        choices = raw.get("choices", [])
        if not choices:
            raise ProviderError("Empty choices returned from server", self._provider_id, model, retryable=False)

        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls") or []
        finish_reason = first_choice.get("finish_reason") or "stop"

        # Parse usage block
        raw_usage = raw.get("usage", {})
        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
        cached_tokens = (
            raw_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or raw_usage.get("cached_tokens", 0)
        )
        total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)

        usage = ProviderUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return InferenceResponse(
            text=text,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            model=model,
            provider_id=self._provider_id,
            duration_ms=duration_ms,
            cost_usd=self.estimate_cost(request, usage),
            metadata={"raw_id": raw.get("id", "")},
        )

    def check_health(self) -> bool:
        """Ping the /models or health endpoint."""
        url = f"{self._base_url}/models"
        headers = {"Authorization": f"Bearer {self._api_key}", **self._extra_headers}
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning(f"Health check failed for {self._provider_id} at {url}: {exc}")
            return False
