"""Unified LLM Provider Abstraction & Adapters.

Provides a model-agnostic client protocol (`LLMProvider`) and standard completion
request/response models. Adapts OpenAI, Anthropic, DeepSeek (including
DeepSeek-V4-Flash parameter options like reasoning_effort), and local vLLM endpoints.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config import TribuneSettings, get_settings


class ProviderAPIError(RuntimeError):
    """Raised when an LLM provider API request fails (network, status, rate limit, etc)."""

    def __init__(self, message: str, status_code: int | None = None, is_rate_limit: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_rate_limit = is_rate_limit or (status_code == 429)


@dataclass
class LLMCompletionRequest:
    messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None  # e.g., "low", "medium", "high" for DeepSeek-V4-Flash


@dataclass
class LLMCompletionResponse:
    content: str
    model: str
    provider_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        ...


class BaseHTTPProviderAdapter:
    """Base class for HTTP-based provider adapters using Python standard library urllib."""

    def __init__(self, provider_name: str, base_url: str, api_key: str, model: str, timeout_s: float = 60.0) -> None:
        self.name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[dict[str, Any], float]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        start_t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_body = resp.read().decode("utf-8")
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                return json.loads(raw_body), latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            is_429 = exc.code == 429
            err_msg = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            raise ProviderAPIError(
                f"HTTP {exc.code} from {self.name} at {url}: {err_msg}",
                status_code=exc.code,
                is_rate_limit=is_429,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            raise ProviderAPIError(f"Network error calling {self.name} at {url}: {exc}") from exc


class OpenAIProviderAdapter(BaseHTTPProviderAdapter):
    """Adapter for OpenAI API protocol (/v1/chat/completions)."""

    def __init__(self, settings: TribuneSettings | None = None, model: str | None = None) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.openai_model
        super().__init__(
            provider_name=f"openai:{target_model}",
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
            model=target_model,
            timeout_s=cfg.request_timeout_s,
        )

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        model = request.model or self.model
        msgs: list[dict[str, str]] = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        msgs.extend(request.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.response_format:
            payload["response_format"] = request.response_format

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body, latency_ms = self._post_json(f"{self.base_url}/chat/completions", payload, headers)

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}

        return LLMCompletionResponse(
            content=content,
            model=model,
            provider_name=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            latency_ms=latency_ms,
            raw_response=body,
        )


class AnthropicProviderAdapter(BaseHTTPProviderAdapter):
    """Adapter for Anthropic Messages API protocol (/v1/messages)."""

    def __init__(self, settings: TribuneSettings | None = None, model: str | None = None) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.anthropic_model
        super().__init__(
            provider_name=f"anthropic:{target_model}",
            base_url=cfg.anthropic_base_url,
            api_key=cfg.anthropic_api_key,
            model=target_model,
            timeout_s=cfg.request_timeout_s,
        )

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        model = request.model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens or 2048,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        body, latency_ms = self._post_json(f"{self.base_url}/messages", payload, headers)

        content_blocks = body.get("content", [])
        content = "".join([b.get("text", "") for b in content_blocks if b.get("type") == "text"])
        usage = body.get("usage") or {}

        return LLMCompletionResponse(
            content=content,
            model=model,
            provider_name=self.name,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cached_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            latency_ms=latency_ms,
            raw_response=body,
        )


class DeepSeekProviderAdapter(BaseHTTPProviderAdapter):
    """Adapter for DeepSeek API protocol including DeepSeek-V4-Flash options like reasoning_effort."""

    def __init__(self, settings: TribuneSettings | None = None, model: str | None = None) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.deepseek_model
        self.default_reasoning_effort = cfg.deepseek_reasoning_effort
        super().__init__(
            provider_name=f"deepseek:{target_model}",
            base_url=cfg.deepseek_base_url,
            api_key=cfg.deepseek_api_key,
            model=target_model,
            timeout_s=cfg.request_timeout_s,
        )

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        model = request.model or self.model
        msgs: list[dict[str, str]] = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        msgs.extend(request.messages)

        effort = request.reasoning_effort or self.default_reasoning_effort
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": request.temperature,
            "reasoning_effort": effort,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.response_format:
            payload["response_format"] = request.response_format

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body, latency_ms = self._post_json(f"{self.base_url}/chat/completions", payload, headers)

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}

        return LLMCompletionResponse(
            content=content,
            model=model,
            provider_name=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            latency_ms=latency_ms,
            raw_response=body,
        )


class vLLMProviderAdapter(BaseHTTPProviderAdapter):
    """Adapter for open-weight / vLLM / SGLang local serving endpoints."""

    def __init__(self, settings: TribuneSettings | None = None, model: str | None = None) -> None:
        cfg = settings or get_settings()
        target_model = model or cfg.vllm_model
        super().__init__(
            provider_name=f"vllm:{target_model}",
            base_url=cfg.vllm_base_url,
            api_key=cfg.vllm_api_key,
            model=target_model,
            timeout_s=cfg.request_timeout_s,
        )

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        model = request.model or self.model
        msgs: list[dict[str, str]] = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        msgs.extend(request.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.response_format:
            payload["response_format"] = request.response_format

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body, latency_ms = self._post_json(f"{self.base_url}/chat/completions", payload, headers)

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}

        return LLMCompletionResponse(
            content=content,
            model=model,
            provider_name=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens", 0) or 0),
            latency_ms=latency_ms,
            raw_response=body,
        )


class LocalRulesLLMAdapter:
    """Deterministic offline adapter for local rules and testing."""

    def __init__(self, settings: TribuneSettings | None = None, name: str = "local_rules") -> None:
        self.name = f"local_rules:{name}"
        self.model = "local-deterministic"

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        start_t = time.perf_counter()
        # Simple deterministic responses for local testing
        user_text = " ".join([m.get("content", "") for m in request.messages])
        if "supported" in (request.system_prompt or ""):
            reply = json.dumps({"supported": True, "concerns": []})
        elif "status" in (request.system_prompt or ""):
            reply = json.dumps({
                "status": "likely_eligible",
                "recommended_action": "prepare_application",
                "self_confidence": 0.95,
                "rationale": "Deterministic local evaluation."
            })
        else:
            reply = json.dumps({"status": "ok", "message": "Local rules evaluation complete."})

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        in_tok = max(1, (len(user_text) + 3) // 4)
        out_tok = max(1, (len(reply) + 3) // 4)

        return LLMCompletionResponse(
            content=reply,
            model=self.model,
            provider_name=self.name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=0,
            latency_ms=latency_ms,
            raw_response={"status": "offline_local"},
        )


def get_llm_provider(
    provider_name: str | None = None,
    settings: TribuneSettings | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Factory creating configured LLMProvider instances dynamically without code changes."""
    cfg = settings or get_settings()
    p_kind = (provider_name or cfg.provider).lower()

    if p_kind in ("openai", "openai_compat"):
        return OpenAIProviderAdapter(settings=cfg, model=model)
    if p_kind == "anthropic":
        return AnthropicProviderAdapter(settings=cfg, model=model)
    if p_kind == "deepseek":
        return DeepSeekProviderAdapter(settings=cfg, model=model)
    if p_kind == "vllm":
        return vLLMProviderAdapter(settings=cfg, model=model)
    return LocalRulesLLMAdapter(settings=cfg, name=p_kind)
