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

import json
import math
import os
from dataclasses import dataclass, field

from ..types import AbstentionScore, Assessment, EligibilityStatus, VerifierVerdict

_PACKAGED_PARITY_THRESHOLDS = os.path.join(
    os.path.dirname(__file__), "..", "eval", "parity_thresholds.json"
)


def load_parity_thresholds(path: str | None = None) -> dict[str, float]:
    """Load configured parity and self-validation thresholds."""
    target_path = path or _PACKAGED_PARITY_THRESHOLDS
    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return {k: float(v) for k, v in payload.items() if isinstance(v, int | float)}
    return {"min_self_validation_confidence": 0.85}


# Logistic weights (priors). See module docstring.
_W0 = -1.2
_W_COVERAGE = 1.0
_W_RESOLVED = 1.0
_W_MARGIN = 1.0
_W_VERIFIER = 0.8
_W_SELF_TESTING = 0.8
_W_AMBIGUITY = 3.0


class MissingLogprobsError(ValueError):
    """Raised when token-level logprobs are missing from the inference server."""
    pass


def calculate_top5_entropy(logprobs: list[float] | list[dict] | dict | None) -> float:
    """Calculate Shannon entropy over the base model's top-5 token predictions at statutory decision boundaries.

    Supports:
    - list of float log-probabilities (natural log ln(p) or base-e logprobs)
    - list of dicts: e.g. [{"logprob": -0.1}, ...]
    - dict with "top_logprobs" or "logprobs" key

    Returns:
        Entropy H in bits (base 2).
    Raises:
        MissingLogprobsError if logprobs are missing, empty, or unparseable.
    """
    if logprobs is None:
        raise MissingLogprobsError("Logprobs payload is None; top-5 log probabilities required.")

    raw_list: list[float] = []
    if isinstance(logprobs, dict):
        cand = logprobs.get("top_logprobs") or logprobs.get("logprobs") or logprobs.get("content")
        if isinstance(cand, list):
            logprobs = cand
        else:
            raise MissingLogprobsError("Invalid logprobs dictionary structure from inference server.")

    if isinstance(logprobs, list):
        for item in logprobs:
            if isinstance(item, int | float):
                raw_list.append(float(item))
            elif isinstance(item, dict):
                lp = item.get("logprob")
                if lp is not None and isinstance(lp, int | float):
                    raw_list.append(float(lp))
            if len(raw_list) >= 5:
                break

    if not raw_list:
        raise MissingLogprobsError("No valid token logprobs extracted from inference server response.")

    # Convert log-probabilities to probabilities: p_i = exp(logprob)
    # Using log-sum-exp normalization for numerical stability
    max_lp = max(raw_list)
    exps = [math.exp(lp - max_lp) for lp in raw_list]
    sum_exps = sum(exps)
    if sum_exps <= 0.0:
        return 0.0

    probs = [e / sum_exps for e in exps]

    # Shannon entropy: H = - sum(p * log2(p))
    entropy = 0.0
    for p in probs:
        if p > 1e-12:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


def evaluate_decision_entropy(
    logprobs: list[float] | list[dict] | dict | None,
    tau: float = 0.35,
) -> tuple[float, bool]:
    """Evaluate decision node entropy H against threshold tau.

    Returns:
        (entropy, is_gated) where is_gated is True if H >= tau (statutory ambiguity requiring model rollout).
    """
    h = calculate_top5_entropy(logprobs)
    return h, h >= tau


@dataclass
class AssessmentDiagnostics:
    required_total: int
    evaluated_required: int
    unknown_required: int
    coverage: float  # evaluated_required / required_total
    resolved_fraction: float  # resolved required / required_total
    min_margin: float  # min normalized distance-to-threshold over evaluated criteria (1.0 if N/A)
    ambiguity_signals: list[str] = field(default_factory=list)
    self_testing_score: float = 1.0
    entropy: float | None = None
    entropy_gated: bool = False



def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Calibrator:
    def __init__(
        self,
        threshold: float,
        min_self_validation_confidence: float | None = None,
        parity_thresholds_path: str | None = None,
    ) -> None:
        self.threshold = threshold
        parity_thresholds = load_parity_thresholds(parity_thresholds_path)
        self.min_self_validation_confidence = (
            min_self_validation_confidence
            if min_self_validation_confidence is not None
            else parity_thresholds.get("min_self_validation_confidence", 0.85)
        )

    def confidence(self, diag: AssessmentDiagnostics, verdict: VerifierVerdict) -> tuple[float, dict[str, float]]:
        c = max(0.0, min(1.0, diag.coverage))
        r = max(0.0, min(1.0, diag.resolved_fraction))
        m = max(0.0, min(1.0, diag.min_margin))
        v = 1.0 if verdict.approved else 0.0
        st = max(0.0, min(1.0, getattr(verdict, "self_testing_score", getattr(diag, "self_testing_score", 1.0))))
        a = float(len(diag.ambiguity_signals))
        logit = (
            _W0
            + _W_COVERAGE * c
            + _W_RESOLVED * r
            + _W_MARGIN * m
            + _W_VERIFIER * v
            + _W_SELF_TESTING * st
            - _W_AMBIGUITY * a
        )
        conf = round(_sigmoid(logit), 4)
        features = {
            "coverage": c,
            "resolved": r,
            "margin": m,
            "verifier": v,
            "self_testing": st,
            "ambiguity": a,
        }
        return conf, features

    def score(
        self,
        assessment: Assessment,
        diag: AssessmentDiagnostics,
        verdict: VerifierVerdict,
    ) -> AbstentionScore:
        conf, features = self.confidence(diag, verdict)
        self_test_score = getattr(
            verdict, "self_testing_score", getattr(diag, "self_testing_score", 1.0)
        )

        # Hard override 1: intermediate self-validation confidence below parity threshold
        if self_test_score < self.min_self_validation_confidence:
            return AbstentionScore(
                calibrated_confidence=conf,
                threshold=self.threshold,
                abstain=True,
                reason=(
                    f"intermediate self-validation confidence ({self_test_score:.3f}) "
                    f"fell below parity threshold ({self.min_self_validation_confidence:.3f}); "
                    "escalated to manual administrative review"
                ),
                features=features,
            )

        # Hard override 2: verifier did not approve
        if not verdict.approved:
            return AbstentionScore(
                calibrated_confidence=conf,
                threshold=self.threshold,
                abstain=True,
                reason="verifier did not approve the assessment; cannot assert an unverified result",
                features=features,
            )

        # Hard override 3: indeterminate status
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
