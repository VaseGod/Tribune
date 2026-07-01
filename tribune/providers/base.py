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


def get_provider_for_role(role: str, settings=None):
    """Factory: ``role`` is 'proposer' or 'verifier'. Selected by config."""
    from ..config import get_settings

    settings = settings or get_settings()
    if role == "verifier":
        kind, model = settings.verifier_provider, settings.verifier_model
    else:
        kind, model = settings.provider, settings.openai_model

    if kind == "openai_compat":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(model=model, settings=settings, role=role)
    from .local_rules import LocalRulesProvider

    return LocalRulesProvider(role=role)
