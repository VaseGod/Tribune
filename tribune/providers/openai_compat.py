"""OpenAI-compatible model provider.

Targets any endpoint that speaks the OpenAI Chat Completions API — most commonly a
self-hosted model served via vLLM or SGLang. This is how a deploying organization
runs TRIBUNE sovereignly on its own model. The HTTP call uses only the standard
library (``urllib``) so no extra dependency is required; install the optional
``httpx`` extra if you prefer.

Activate with::

    export TRIBUNE_PROVIDER=openai_compat
    export TRIBUNE_OPENAI_BASE_URL=http://localhost:8000/v1
    export TRIBUNE_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct

The model decides the eligibility status, recommended action, and confidence from
the *structured* criteria + citations the agent provides — but the agent still
builds the Assessment, so an uncited eligibility claim remains impossible.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from ..config import TribuneSettings
from ..instrumentation.usage import ESTIMATOR_TOKENIZER_ID, UsageRecorder, estimate_tokens
from ..types import EligibilityStatus, RecommendedAction
from .base import (
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
    derive_status,
    recommend_action,
)

_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thought|reasoning)[^>]*>.*?</(?:think|thought|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning_monologues(text: str) -> str:
    """Isolate explicit, visible model text response and strictly ignore thinking monologues."""
    if not isinstance(text, str):
        return text
    return _THINKING_BLOCK_RE.sub("", text).strip()


class ProviderError(RuntimeError):
    pass


_SYNTH_SYSTEM = (
    "You are TRIBUNE's eligibility assessor for public benefits. You are given the "
    "result of evaluating each governing criterion against a person's documented "
    "evidence, plus the citations for those rules. You must NOT invent facts or "
    "criteria beyond those provided. Decide the eligibility status using only the "
    "supplied criterion outcomes. If any required criterion is 'unknown', the status "
    "is 'indeterminate'. Never issue a binding determination. Respond ONLY with a "
    "JSON object: {\"status\": one of [likely_eligible, likely_ineligible, "
    "indeterminate], \"recommended_action\": one of [prepare_application, "
    "gather_more_evidence, abstain_and_escalate], \"self_confidence\": a number in "
    "[0,1], \"rationale\": a short string}."
)

_REVIEW_SYSTEM = (
    "You are TRIBUNE's independent verifier. Given an eligibility assessment, the "
    "criteria re-derived from the cited rules, and the citations, judge whether the "
    "cited rules actually support the asserted status. Respond ONLY with JSON: "
    "{\"supported\": boolean, \"concerns\": [list of short strings]}."
)


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        settings: TribuneSettings,
        role: str = "proposer",
        recorder: UsageRecorder | None = None,
        extra_body: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.role = role
        self.model = model
        self.name = f"openai_compat:{model}"
        self.version = model
        self.recorder = recorder
        if "grok" in model.lower() or "xai" in model.lower():
            self._base = getattr(settings, "grok_base_url", settings.openai_base_url).rstrip("/")
            self._key = getattr(settings, "grok_api_key", "") or settings.openai_api_key
        elif "deepseek" in model.lower():
            self._base = getattr(settings, "deepseek_base_url", settings.openai_base_url).rstrip("/")
            self._key = getattr(settings, "deepseek_api_key", "") or settings.openai_api_key
        elif "gemini" in model.lower():
            self._base = getattr(settings, "gemini_base_url", settings.openai_base_url).rstrip("/")
            self._key = getattr(settings, "gemini_api_key", "") or settings.openai_api_key
        else:
            self._base = settings.openai_base_url.rstrip("/")
            self._key = settings.openai_api_key
        self._timeout = settings.request_timeout_s
        self.extra_body = extra_body or {}
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def pricing_parameters(self) -> dict[str, float]:
        """Return input and output pricing parameters per 1M tokens."""
        m = self.model.lower()
        if "gemini-3.7" in m or "gemini-3.7-flash" in m:
            return {"input_cost_per_1m": 0.75, "output_cost_per_1m": 3.75}
        if "gpt-5.6" in m or "gpt-5.6-sol-ultrafast" in m:
            return {"input_cost_per_1m": 2.50, "output_cost_per_1m": 10.00}
        if "grok-4.6" in m or "grok" in m:
            return {"input_cost_per_1m": 2.00, "output_cost_per_1m": 6.00}
        if "deepseek-v4-pro" in m:
            return {"input_cost_per_1m": 0.435, "output_cost_per_1m": 0.87}
        return {"input_cost_per_1m": 0.75, "output_cost_per_1m": 3.75}

    def calculate_token_cost(
        self, tokens_input: int, tokens_output: int, cache_read_tokens: int = 0
    ) -> float:
        """Calculate exact token cost for model inference."""
        params = self.pricing_parameters
        uncached_in = max(0, tokens_input - cache_read_tokens)
        cost = (
            uncached_in * params["input_cost_per_1m"]
            + cache_read_tokens * (params["input_cost_per_1m"] * 0.25)
            + tokens_output * params["output_cost_per_1m"]
        ) / 1_000_000.0
        return cost

    # -- transport ---------------------------------------------------------- #

    @staticmethod
    def sanitize_reasoning_payload(payload: dict) -> dict:
        """Strip or neutralize all opaque/unverified reasoning fields from payloads."""
        if not isinstance(payload, dict):
            return payload

        opaque_keys = {
            "encrypted_content",
            "thought_signature",
            "raw_base64_blobs",
            "reasoning_content",
            "thought",
            "thinking",
        }

        clean = {}
        for k, v in payload.items():
            if k in opaque_keys or "encrypted" in k.lower() or "thought_signature" in k.lower():
                continue
            if k == "messages" and isinstance(v, list):
                clean_msgs = []
                for msg in v:
                    if isinstance(msg, dict):
                        clean_msg = {}
                        for mk, mv in msg.items():
                            if mk in opaque_keys or "encrypted" in mk.lower() or "thought_signature" in mk.lower():
                                continue
                            clean_msg[mk] = mv
                        clean_msgs.append(clean_msg)
                    else:
                        clean_msgs.append(msg)
                clean[k] = clean_msgs
            elif isinstance(v, dict):
                clean[k] = OpenAICompatProvider.sanitize_reasoning_payload(v)
            else:
                clean[k] = v

        return clean

    def complete(self, payload: dict) -> dict:
        """Sanitize payload prior to serialization."""
        return self.sanitize_reasoning_payload(payload)

    def _chat(self, system: str, user: str) -> str:  # pragma: no cover - needs endpoint
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.extra_body:
            payload["extra_body"] = self.extra_body
            for k, v in self.extra_body.items():
                payload[k] = v
        payload = self.complete(payload)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"request to {self._base} failed: {exc}") from exc
        content = body["choices"][0]["message"]["content"]
        content = strip_reasoning_monologues(content)
        self._record_usage(body.get("usage"), system + user, content)
        return content

    def _record_usage(self, usage: dict | None, sent_text: str, received_text: str) -> None:
        """Report token usage to the recorder.

        Prefers the serving backend's own ``usage`` block (vLLM, SGLang, and
        llama.cpp's server all return one); falls back to the deterministic
        estimator, flagged as estimated, when it is absent.
        """
        if self.recorder is None:
            return
        if usage and "prompt_tokens" in usage:
            details = usage.get("prompt_tokens_details") or {}
            self.recorder.record_call(
                role=self.role,
                model=self.name,
                tokenizer_id=self.model,
                tokens_input=int(usage.get("prompt_tokens", 0)),
                tokens_output=int(usage.get("completion_tokens", 0)),
                cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
                estimated=False,
            )
        else:
            self.recorder.record_call(
                role=self.role,
                model=self.name,
                tokenizer_id=ESTIMATOR_TOKENIZER_ID,
                tokens_input=estimate_tokens(sent_text),
                tokens_output=estimate_tokens(received_text),
                estimated=True,
            )

    # -- contract ----------------------------------------------------------- #

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:  # pragma: no cover
        user = json.dumps(
            {
                "program": req.program.value,
                "jurisdiction": req.jurisdiction,
                "coverage_complete": req.coverage_complete,
                "required_total": req.required_total,
                "evidence_summary": req.evidence_summary,
                "criteria": [
                    {
                        "criterion_id": c.criterion_id,
                        "description": c.description,
                        "required": c.required,
                        "outcome": c.outcome.value,
                        "citation_ids": c.citation_ids,
                    }
                    for c in req.criteria
                ],
                "citations": [{"id": c.citation_id, "source": c.source, "text": c.text} for c in req.citations],
            }
        )
        raw = self._chat(_SYNTH_SYSTEM, user)
        try:
            obj = json.loads(raw)
            status = EligibilityStatus(obj["status"])
            action = RecommendedAction(obj["recommended_action"])
            conf = float(obj.get("self_confidence", 0.5))
            rationale = str(obj.get("rationale", ""))
            return SynthesisResult(status=status, recommended_action=action, self_confidence=conf, rationale=rationale)
        except (ValueError, KeyError, TypeError):
            # Degrade safely: fall back to the deterministic derivation rather than
            # emitting a malformed or over-confident result.
            status = derive_status(req.criteria, req.coverage_complete)
            return SynthesisResult(
                status=status,
                recommended_action=recommend_action(status),
                self_confidence=0.4,
                rationale="Model output could not be parsed; fell back to rule-based derivation.",
            )

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:  # pragma: no cover
        user = json.dumps(
            {
                "asserted_status": req.assessment.status.value,
                "recomputed_criteria": [
                    {"criterion_id": c.criterion_id, "outcome": c.outcome.value} for c in req.recomputed
                ],
                "citations": [{"id": c.citation_id, "text": c.text} for c in req.citations],
            }
        )
        raw = self._chat(_REVIEW_SYSTEM, user)
        try:
            obj = json.loads(raw)
            return ReviewResult(supported=bool(obj["supported"]), concerns=list(obj.get("concerns", [])))
        except (ValueError, KeyError, TypeError):
            return ReviewResult(supported=True, concerns=["verifier model output unparseable; deferring to structural checks"])
