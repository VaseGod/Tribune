"""Deterministic, offline model provider.

This is the default backend. It makes the entire system runnable and tests
reproducible with no API keys, no GPU, and no network. It does not "predict" — it
applies the shared status-derivation logic to the criteria the agent evaluated and
writes a plain rationale. Swapping in a served model (see ``openai_compat``) is a
one-environment-variable change.
"""

from __future__ import annotations

from ..types import CriterionOutcome
from .base import (
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
    derive_status,
    recommend_action,
)

_VERSION = "0.1.0"


class LocalRulesProvider:
    def __init__(self, role: str = "proposer") -> None:
        self.role = role
        self.name = "local_rules"
        self.version = _VERSION

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        status = derive_status(req.criteria, req.coverage_complete)
        action = recommend_action(status)
        required = [c for c in req.criteria if c.required]
        resolved = [c for c in required if c.outcome is not CriterionOutcome.UNKNOWN]
        resolved_frac = (len(resolved) / len(required)) if required else 0.0
        coverage = (len(required) / req.required_total) if req.required_total else 0.0
        # Naive, *uncalibrated* self-confidence. The calibrator produces the number
        # that actually gates assertion vs. abstention.
        self_conf = round(0.5 + 0.25 * resolved_frac + 0.25 * coverage, 4)

        lines = []
        for c in req.criteria:
            mark = {
                CriterionOutcome.SATISFIED: "met",
                CriterionOutcome.NOT_SATISFIED: "not met",
                CriterionOutcome.UNKNOWN: "insufficient evidence",
            }[c.outcome]
            lines.append(f"- {c.description} -> {mark}")
        rationale = (
            f"Assessed {len(req.criteria)} of {req.required_total} governing criteria "
            f"for this program. Result: {status.value}.\n" + "\n".join(lines)
        )
        return SynthesisResult(
            status=status,
            recommended_action=action,
            self_confidence=self_conf,
            rationale=rationale,
        )

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        # Structural verification lives in the verifier agent (independent
        # re-derivation from the cited rules). The provider's model-side review adds
        # a light consistency check; the local provider flags only obvious issues.
        concerns: list[str] = []
        if req.assessment.is_assertion and req.assessment.self_confidence < 0.5:
            concerns.append("proposer self-confidence is low for an asserted result")
        if req.assessment.is_assertion and not req.assessment.citations:
            concerns.append("asserted result without citations")
        return ReviewResult(supported=len(concerns) == 0, concerns=concerns)
