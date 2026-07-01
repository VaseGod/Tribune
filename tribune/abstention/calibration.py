"""Calibrated abstention.

Given a verified assessment and a set of diagnostic features, decide whether
TRIBUNE is confident enough to *assert* an eligibility result or whether it must
*abstain and escalate to a human*. Being confidently wrong is the cardinal harm;
abstaining on a genuinely uncertain case is the rewarded outcome.

The calibrated confidence is a logistic function of interpretable features:
rule-coverage, the fraction of required criteria actually resolved, how close the
decisive numbers are to a threshold (margin), whether the verifier approved, and a
count of structural ambiguity signals (e.g. the Medicaid coverage gap, a
good-cause quit, an unknown housing waitlist). The weights are sensible, hand-set
priors; they can be *fit* on labeled synthetic data without changing this
interface. Two hard overrides sit on top of the score: an INDETERMINATE status or
a failed verification always abstains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..types import AbstentionScore, Assessment, EligibilityStatus, VerifierVerdict

# Logistic weights (priors). See module docstring.
_W0 = -1.2
_W_COVERAGE = 1.0
_W_RESOLVED = 1.0
_W_MARGIN = 1.0
_W_VERIFIER = 0.8
_W_AMBIGUITY = 3.0


@dataclass
class AssessmentDiagnostics:
    required_total: int
    evaluated_required: int
    unknown_required: int
    coverage: float  # evaluated_required / required_total
    resolved_fraction: float  # resolved required / required_total
    min_margin: float  # min normalized distance-to-threshold over evaluated criteria (1.0 if N/A)
    ambiguity_signals: list[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Calibrator:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def confidence(self, diag: AssessmentDiagnostics, verdict: VerifierVerdict) -> tuple[float, dict[str, float]]:
        c = max(0.0, min(1.0, diag.coverage))
        r = max(0.0, min(1.0, diag.resolved_fraction))
        m = max(0.0, min(1.0, diag.min_margin))
        v = 1.0 if verdict.approved else 0.0
        a = float(len(diag.ambiguity_signals))
        logit = _W0 + _W_COVERAGE * c + _W_RESOLVED * r + _W_MARGIN * m + _W_VERIFIER * v - _W_AMBIGUITY * a
        conf = round(_sigmoid(logit), 4)
        features = {"coverage": c, "resolved": r, "margin": m, "verifier": v, "ambiguity": a}
        return conf, features

    def score(
        self,
        assessment: Assessment,
        diag: AssessmentDiagnostics,
        verdict: VerifierVerdict,
    ) -> AbstentionScore:
        conf, features = self.confidence(diag, verdict)

        # Hard overrides — these always abstain regardless of the logistic score.
        if not verdict.approved:
            return AbstentionScore(
                calibrated_confidence=conf,
                threshold=self.threshold,
                abstain=True,
                reason="verifier did not approve the assessment; cannot assert an unverified result",
                features=features,
            )
        if assessment.status is EligibilityStatus.INDETERMINATE:
            return AbstentionScore(
                calibrated_confidence=conf,
                threshold=self.threshold,
                abstain=True,
                reason="insufficient evidence to resolve all required criteria",
                features=features,
            )

        if diag.ambiguity_signals:
            reason = "case sits on a genuinely ambiguous edge: " + "; ".join(diag.ambiguity_signals)
        elif conf < self.threshold:
            reason = "calibrated confidence below the assertion threshold"
        else:
            reason = "calibrated confidence meets the assertion threshold"

        return AbstentionScore(
            calibrated_confidence=conf,
            threshold=self.threshold,
            abstain=conf < self.threshold,
            reason=reason,
            features=features,
        )
