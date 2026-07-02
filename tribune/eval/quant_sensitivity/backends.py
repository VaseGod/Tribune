"""Quantization-ladder backend adapters.

A rung is *anything that serves the verifier model at one quantization level*.
Two adapters plug in:

* **openai_compat** — a served endpoint. Both llama.cpp's server (GGUF quants:
  Q8_0/Q4_K_M/Q2_K/IQ1_S...) and vLLM (GPTQ/AWQ/FP8/NVFP4...) speak the OpenAI
  chat-completions API, so one adapter covers both; the rung just points the
  *verifier* at a different base_url/model. The proposer configuration is left
  untouched — proposer/verifier separation is preserved per rung.
* **mock** — a deterministic, seeded degradation model used for CI smoke tests
  and free offline runs: with probability ``flip_prob`` (keyed by a stable hash
  of the assessment id, so runs are reproducible) the model-side review flips,
  which exercises exactly the failure mode this harness hunts — quantization
  noise pushing the verifier's judgement, and therefore abstention behavior,
  off its full-precision calibration.

The mock is a stand-in for a *verifier-side model review* only; it never touches
the structural checks (citation integrity, coverage, re-derivation), which stay
exactly as strict as in production.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from ...config import TribuneSettings
from ...instrumentation.usage import ESTIMATOR_TOKENIZER_ID, UsageRecorder, estimate_tokens
from ...providers.base import ReviewRequest, ReviewResult, SynthesisRequest, SynthesisResult
from ...providers.local_rules import LocalRulesProvider


@dataclass(frozen=True)
class QuantRung:
    label: str  # e.g. "fp16", "q8", "q4", "q2", "iq1"
    quant_format: str  # e.g. "fp16", "gguf-q8_0", "gguf-q2_k", "nvfp4"
    provider_kind: str = "mock"  # "mock" | "openai_compat"
    backend: str = "mock"  # "mock" | "llama.cpp" | "vllm" | "sglang"
    model: str = ""  # served model name (openai_compat rungs)
    base_url: str = ""  # served endpoint (openai_compat rungs)
    flip_prob: float = 0.0  # mock rungs: seeded review-flip probability
    notes: str = ""
    reference: bool = False  # the full-precision reference rung


def default_mock_ladder() -> list[QuantRung]:
    """A free, offline, deterministic ladder mirroring a GGUF quant series."""
    return [
        QuantRung("fp16", "fp16", flip_prob=0.0, reference=True,
                  notes="full-precision reference"),
        QuantRung("q8", "gguf-q8_0", flip_prob=0.02),
        QuantRung("q4", "gguf-q4_k_m", flip_prob=0.06),
        QuantRung("q2", "gguf-q2_k", flip_prob=0.22),
        QuantRung("iq1", "gguf-iq1_s", flip_prob=0.40,
                  notes="~1.6-bit; expected to be badly degraded"),
    ]


def smoke_ladder() -> list[QuantRung]:
    """Two rungs for the CI smoke test."""
    return [
        QuantRung("fp16", "fp16", flip_prob=0.0, reference=True),
        QuantRung("q2", "gguf-q2_k", flip_prob=0.30),
    ]


def load_ladder_config(path: str) -> list[QuantRung]:
    """Load a real-endpoint ladder from JSON (see docs/quant_sensitivity.md)."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    rungs = [QuantRung(**r) for r in payload["rungs"]]
    if not any(r.reference for r in rungs):
        raise ValueError("ladder config must mark exactly one rung as the reference")
    return rungs


def settings_for_rung(rung: QuantRung, base: TribuneSettings) -> TribuneSettings:
    """Point the *verifier only* at the rung's endpoint; the proposer is untouched."""
    if rung.provider_kind != "openai_compat":
        return base
    return base.model_copy(
        update={
            "verifier_provider": "openai_compat",
            "verifier_model": rung.model,
            "openai_base_url": rung.base_url or base.openai_base_url,
        }
    )


@dataclass
class MockQuantVerifierProvider:
    """Deterministic quantization-degradation stand-in for the verifier's model review."""

    rung: QuantRung
    recorder: UsageRecorder | None = None
    role: str = "verifier"
    name: str = field(init=False)
    version: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"mock_quant:{self.rung.label}"
        self.version = self.rung.quant_format
        self._base = LocalRulesProvider(role="verifier")

    def _flips(self, key: str) -> bool:
        digest = hashlib.sha256(f"{self.rung.label}:{key}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return u < self.rung.flip_prob

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        # Never used when mounted as the verifier; delegate defensively.
        return self._base.synthesize_assessment(req)

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        result = self._base.review_assessment(req)
        if self._flips(req.assessment.assessment_id):
            result = ReviewResult(
                supported=not result.supported,
                concerns=result.concerns
                + [f"[simulated {self.rung.quant_format} quantization noise]"],
            )
        if self.recorder is not None:
            text = req.assessment.rationale + " ".join(c.text for c in req.citations)
            self.recorder.record_call(
                role=self.role,
                model=self.name,
                tokenizer_id=ESTIMATOR_TOKENIZER_ID,
                tokens_input=estimate_tokens(text),
                tokens_output=estimate_tokens(" ".join(result.concerns) or "supported"),
                estimated=True,
            )
        return result


def mount_rung(pipeline, rung: QuantRung) -> None:
    """Attach a mock rung to an already-built pipeline (verifier side only)."""
    if rung.provider_kind == "mock":
        pipeline.verifier.provider = MockQuantVerifierProvider(rung, recorder=pipeline.recorder)


def hardware_notes() -> str:  # used by the report and docs
    return (
        "Full-ladder runs against real weights need a llama.cpp or vLLM server per "
        "rung (e.g. `llama-server -m model-Q4_K_M.gguf --port 8081`). CPU-only "
        "boxes can run small GGUF models; Q2/IQ1 rungs of large models need "
        f"significant RAM/VRAM. See {os.path.join('docs', 'quant_sensitivity.md')}."
    )
