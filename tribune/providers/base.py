"""The ModelProvider protocol and the typed request/response contracts.

Agents never see a raw text LLM. They hand the provider *structured* inputs
(evaluated criteria, citations) and receive structured outputs. This is what lets
a deterministic local provider and a served model implement the exact same
contract — and what lets the citation invariant be enforced uniformly: the agent
builds the :class:`~tribune.types.Assessment`, and an uncited eligibility claim
cannot be constructed regardless of which provider produced the status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import (
    Assessment,
    Citation,
    CriterionResult,
    EligibilityStatus,
    ProgramId,
    RecommendedAction,
)


@dataclass
class SynthesisRequest:
    program: ProgramId
    jurisdiction: str
    criteria: list[CriterionResult]
    required_total: int
    coverage_complete: bool
    evidence_summary: str
    citations: list[Citation]


@dataclass
class SynthesisResult:
    status: EligibilityStatus
    recommended_action: RecommendedAction
    self_confidence: float
    rationale: str


@dataclass
class ReviewRequest:
    assessment: Assessment
    recomputed: list[CriterionResult]
    citations: list[Citation]


@dataclass
class ReviewResult:
    supported: bool
    concerns: list[str] = field(default_factory=list)


@runtime_checkable
class ModelProvider(Protocol):
    name: str
    version: str

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult: ...

    def review_assessment(self, req: ReviewRequest) -> ReviewResult: ...


@dataclass
class LowLatencyRoutingSpec:
    """Specification for sub-second ultra-low-latency provider pipelines."""
    target_model: str = "gpt-5.6-sol-ultrafast"
    max_latency_ms: float = 1000.0
    tokens_per_second: float = 750.0
    fallback_model: str = "gemini-3.7-flash"
    supports_tools: bool = True


class UltraLowLatencyPipeline:
    """Routing abstraction targeting sub-second inference models for high-speed policy gating."""

    def __init__(
        self,
        primary_provider: ModelProvider,
        fallback_provider: ModelProvider | None = None,
        spec: LowLatencyRoutingSpec | None = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.spec = spec or LowLatencyRoutingSpec()
        self.name = f"ultra_low_latency:{self.spec.target_model}"
        self.version = self.spec.target_model

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        try:
            return self.primary_provider.synthesize_assessment(req)
        except Exception:
            if self.fallback_provider:
                return self.fallback_provider.synthesize_assessment(req)
            raise

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        try:
            return self.primary_provider.review_assessment(req)
        except Exception:
            if self.fallback_provider:
                return self.fallback_provider.review_assessment(req)
            raise


def derive_status(
    criteria: list[CriterionResult], coverage_complete: bool
) -> EligibilityStatus:
    """Shared, conservative status derivation from evaluated criteria.

    Note the deliberate optimism on *coverage*: when every required criterion that
    was retrieved is satisfied, the proposer concludes eligibility even if not all
    required criteria were retrieved. The verifier independently checks coverage
    and rejects this leap, which drives the REPLAN path. Evidence gaps (UNKNOWN
    required criteria), by contrast, force INDETERMINATE here.
    """
    from ..types import CriterionOutcome

    required = [c for c in criteria if c.required]
    if any(c.outcome is CriterionOutcome.NOT_SATISFIED for c in required):
        return EligibilityStatus.LIKELY_INELIGIBLE
    if any(c.outcome is CriterionOutcome.UNKNOWN for c in required):
        return EligibilityStatus.INDETERMINATE
    if required:  # all retrieved required criteria satisfied
        return EligibilityStatus.LIKELY_ELIGIBLE
    return EligibilityStatus.INDETERMINATE


def recommend_action(status: EligibilityStatus) -> RecommendedAction:
    if status is EligibilityStatus.LIKELY_ELIGIBLE:
        return RecommendedAction.PREPARE_APPLICATION
    if status is EligibilityStatus.INDETERMINATE:
        return RecommendedAction.GATHER_MORE_EVIDENCE
    # A "likely ineligible" result is high-stakes; route to a human who can confirm
    # or help the person contest it rather than treating it as a final word.
    return RecommendedAction.ABSTAIN_AND_ESCALATE


def get_provider_for_role(role: str, settings=None, recorder=None):
    """Factory: ``role`` is 'proposer', 'verifier', or 'router'. Selected by config.

    ``recorder`` is an optional :class:`~tribune.instrumentation.usage.UsageRecorder`;
    providers report per-call token usage to it when present.
    """
    from ..config import get_settings

    settings = settings or get_settings()

    if role == "router" or (settings and getattr(settings, "provider", None) == "router"):
        from .router import ModelRouter

        return ModelRouter(settings=settings, recorder=recorder)

    if role == "verifier":
        kind, model = settings.verifier_provider, settings.verifier_model
    else:
        kind, model = settings.provider, settings.openai_model

    if kind == "openai_compat":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(model=model, settings=settings, role=role, recorder=recorder)
    from .local_rules import LocalRulesProvider

    return LocalRulesProvider(role=role, recorder=recorder)


def _canonical_state(state: dict) -> str:
    body = {k: v for k, v in state.items() if k != "_hmac_signature"}
    import json
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def bind_session_state(state: dict, session_key: str) -> dict:
    """Sign a local session state object with HMAC-SHA256."""
    import hashlib
    import hmac

    canonical = _canonical_state(state)
    sig = hmac.new(session_key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(state)
    out["_hmac_signature"] = sig
    return out


def verify_session_state(signed_state: dict, session_key: str) -> dict:
    """Verify session state object HMAC-SHA256 signature, raising ValueError if invalid or missing."""
    import hashlib
    import hmac

    if not isinstance(signed_state, dict) or "_hmac_signature" not in signed_state:
        raise ValueError("Session state block replayed without a valid session key signature.")
    expected_sig = str(signed_state["_hmac_signature"])
    canonical = _canonical_state(signed_state)
    computed_sig = hmac.new(session_key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, computed_sig):
        raise ValueError("Session state HMAC signature verification failed.")
    clean = dict(signed_state)
    clean.pop("_hmac_signature", None)
    return clean

