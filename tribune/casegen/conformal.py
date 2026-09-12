"""Conformal Risk Control (CRC) & Statistical Calibration Engine.

Provides finite-sample statistical guarantees for runtime verification of
multi-turn simulation trajectories. Calibrates empirical non-conformity cutoffs
incorporating finite-sample correction and Dvoretzky-Kiefer-Wolfowitz / Hoeffding
statistical padding terms:

    q_hat = ceil((n + 1)(1 - alpha)) / n
    epsilon_n = sqrt(ln(2 / delta) / (2 * n))

Guarantees empirical false-alarm rates <= alpha + epsilon_n at confidence level 1 - delta.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Any


class ConformalScoreType(str, enum.Enum):
    """Semantic direction of the evaluation metric."""

    CONFORMITY = "conformity"  # Higher score = more valid / compliant in [0.0, 1.0]
    NON_CONFORMITY = "non_conformity"  # Higher score = greater invariant violation in [0.0, 1.0]


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Rigorous statistical calibration bounds and metadata."""

    n_samples: int
    alpha: float  # Nominal false-alarm tolerance (e.g., 0.10)
    delta: float  # Statistical error tolerance (e.g., 0.05 for 95% confidence)
    q_hat: float  # Finite-sample empirical quantile index
    epsilon_n: float  # Statistical padding term sqrt(ln(2/delta) / (2n))
    raw_threshold: float  # Unpadded empirical cutoff lambda_hat
    padded_threshold: float  # Certified padded threshold lambda_hat_padded
    score_type: ConformalScoreType
    scores: list[float] = field(default_factory=list)

    @property
    def certified_upper_bound(self) -> float:
        """The guaranteed statistical false-alarm bound: alpha + epsilon_n."""
        return min(1.0, self.alpha + self.epsilon_n)


class ConformalCalibrator:
    """Conformal calibration harness for multi-turn statutory simulation verification."""

    def __init__(
        self,
        alpha: float = 0.10,
        delta: float = 0.05,
        score_type: ConformalScoreType = ConformalScoreType.CONFORMITY,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self.alpha = alpha
        self.delta = delta
        self.score_type = score_type
        self._last_result: ConformalCalibrationResult | None = None

    def compute_padding(self, n: int, delta: float | None = None) -> float:
        """Compute finite-sample padding term: epsilon_n = sqrt(ln(2 / delta) / (2 * n))."""
        if n <= 0:
            raise ValueError(f"Sample size n must be positive, got {n}")
        d = delta if delta is not None else self.delta
        return math.sqrt(math.log(2.0 / d) / (2.0 * n))

    def calibrate(
        self,
        scores: list[float],
        alpha: float | None = None,
        delta: float | None = None,
        score_type: ConformalScoreType | None = None,
    ) -> ConformalCalibrationResult:
        """Compute empirical conformal cutoff lambda_hat and padded cutoff lambda_hat_padded.

        For non-conformity scores S_t in [0, 1] (higher = violation):
            q_hat = ceil((n + 1)(1 - alpha)) / n
            lambda_hat = S_{(ceil((n+1)(1-alpha)))}
            lambda_padded = min(1.0, lambda_hat + epsilon_n)
            Transition flagged if S_t > lambda_padded.

        For conformity scores V_t in [0, 1] (higher = valid):
            We sort V ascending. A false alarm occurs when V < lambda.
            To guarantee P(V < lambda) <= alpha, lambda is selected at index floor((n + 1)*alpha).
            lambda_padded = max(0.0, lambda_hat - epsilon_n)
            Transition flagged if V_t < lambda_padded.
        """
        if not scores:
            raise ValueError("Cannot calibrate on an empty score list.")

        n = len(scores)
        a = alpha if alpha is not None else self.alpha
        d = delta if delta is not None else self.delta
        st = score_type if score_type is not None else self.score_type

        sorted_scores = sorted(scores)
        epsilon_n = self.compute_padding(n, d)

        if st == ConformalScoreType.NON_CONFORMITY:
            q_hat = min(1.0, math.ceil((n + 1) * (1.0 - a)) / n)
            idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - a)) - 1))
            raw_threshold = float(sorted_scores[idx])
            padded_threshold = min(1.0, raw_threshold + epsilon_n)
        else:
            # Conformity score: higher is more valid
            q_hat = min(1.0, math.ceil((n + 1) * (1.0 - a)) / n)
            # False alarms occur at the lower tail (scores < threshold)
            idx = min(n - 1, max(0, math.floor((n + 1) * a) - 1))
            raw_threshold = float(sorted_scores[idx])
            padded_threshold = max(0.0, raw_threshold - epsilon_n)

        result = ConformalCalibrationResult(
            n_samples=n,
            alpha=a,
            delta=d,
            q_hat=q_hat,
            epsilon_n=epsilon_n,
            raw_threshold=raw_threshold,
            padded_threshold=padded_threshold,
            score_type=st,
            scores=sorted_scores,
        )
        self._last_result = result
        return result

    def is_valid(
        self,
        score: float,
        threshold: float | None = None,
        score_type: ConformalScoreType | None = None,
    ) -> bool:
        """Evaluate if a transition score passes the calibrated conformal cutoff."""
        st = score_type if score_type is not None else self.score_type
        t = threshold
        if t is None:
            if self._last_result is None:
                raise RuntimeError("Calibrator has not been calibrated and no threshold provided.")
            t = self._last_result.padded_threshold

        if st == ConformalScoreType.NON_CONFORMITY:
            return score <= t
        else:
            return score >= t

    def generate_calibration_traces(
        self,
        n: int = 500,
        world_model: Any = None,
        seed: int = 42,
    ) -> list[float]:
        """Calibration harness generating n historical simulation trace scores.

        Simulates historical compliant transitions with nominal noise to establish
        an empirical baseline of statutory invariant conformity.
        """
        rng = random.Random(seed)
        scores: list[float] = []

        for _ in range(n):
            if world_model is not None and hasattr(world_model, "score_transition"):
                # Synthetic compliant transition evaluated by the world model
                mock_current_state = {
                    "monthly_income": rng.uniform(800.0, 1400.0),
                    "household_size": rng.choice([1, 2, 3]),
                    "days_since_denial": rng.randint(10, 70),
                    "evidence_marked": True,
                    "evidence_offered": True,
                }
                mock_turn = {
                    "action": "document_review",
                    "reported_income": mock_current_state["monthly_income"] + rng.uniform(-10.0, 10.0),
                    "days_since_denial": mock_current_state["days_since_denial"] + 1,
                    "exhibit_status": "admitted",
                }
                score = world_model.score_transition(mock_current_state, mock_turn)
                scores.append(float(score))
            else:
                # High-conformity distribution for valid traces (Beta centered at 0.95)
                # representing nominal probability of invariant satisfaction
                val = rng.betavariate(25.0, 1.5)
                scores.append(round(min(1.0, max(0.0, val)), 4))

        return scores


__all__ = [
    "ConformalScoreType",
    "ConformalCalibrationResult",
    "ConformalCalibrator",
]
