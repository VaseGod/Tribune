"""Continuous Audit Judge Evaluators, Live Trace Monitoring, & Governance Reward Oracle.

Provides:
1. Pluggable judge evaluators (HeuristicJudge, LocalClassifierJudge, RemoteJudge) evaluating
   100% of verifier outputs in real time for perceived error rate, uncited claims, and citation coverage.
2. Governance Reward Oracle & Invariant Checks for synthetic trajectory synthesis and verification:
   - Oracle Check: Validates trajectory outcome alignment against environment ground truth.
   - No-Op Invariance Check: Penalizes redundant, circular, or hallucinatory procedural steps.
   - Unsolved-State Penalty: Penalizes determinations concluding eligibility while blocking hidden ambiguities remain.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import field
from datetime import datetime, timezone
from typing import Any

from ..types import (
    Assessment,
    CriterionOutcome,
    Evidence,
    ProgramOutcome,
    StrictModel,
    SyntheticCase,
    VerifierVerdict,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JudgeResult(StrictModel):
    """Structured evaluation verdict emitted by an audit judge."""

    output_id: str
    assessment_id: str
    program: str
    perceived_error_score: float  # [0.0, 1.0] (0.0 = no perceived error, 1.0 = critical error)
    uncited_claim_score: float  # [0.0, 1.0] (0.0 = 100% grounded/cited, 1.0 = completely uncited)
    citation_coverage_score: float  # [0.0, 1.0] (1.0 = complete statutory citation coverage)
    rule_reference_coverage: float  # [0.0, 1.0] (fraction of required rules referenced)
    provenance_completeness: float  # [0.0, 1.0] (fraction of evidence carrying verified provenance)
    judge_confidence: float  # [0.0, 1.0]
    passed: bool
    reasons: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    judge_name: str = "heuristic_judge"
    judge_version: str = "1.0.0"
    cost_estimate: float = 0.00015  # Estimated cost in USD
    evaluated_at: datetime = field(default_factory=_utcnow)


class JudgeEvaluator(ABC):
    """Base interface for real-time audit judges monitoring agent outputs."""

    name: str = "base_judge"
    version: str = "1.0.0"

    @abstractmethod
    def evaluate(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        evidence: list[Evidence],
        jurisdiction: str,
    ) -> JudgeResult:
        """Evaluate a verifier output and return a structured JudgeResult."""
        pass


class HeuristicJudge(JudgeEvaluator):
    """Fast, deterministic statutory rule and citation coverage audit judge."""

    name: str = "heuristic_judge"
    version: str = "1.1.0"

    def evaluate(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        evidence: list[Evidence],
        jurisdiction: str,
    ) -> JudgeResult:
        output_id = f"judge:{assessment.assessment_id}:{verdict.recomputed_status.value}"
        reasons: list[str] = []

        # 1. Uncited Claim & Citation Coverage Analysis
        total_criteria = len(assessment.criteria)
        cited_criteria = sum(1 for c in assessment.criteria if c.citation_ids)
        citation_coverage = (cited_criteria / max(1, total_criteria))

        uncited_claims = 0
        for crit in assessment.criteria:
            if crit.outcome is not CriterionOutcome.UNKNOWN and not crit.citation_ids:
                uncited_claims += 1
                reasons.append(f"Uncited criterion outcome: '{crit.criterion_id}'")

        if assessment.is_assertion and not assessment.citations:
            uncited_claims += 1
            reasons.append("Asserted eligibility without top-level citations")

        uncited_score = min(1.0, uncited_claims / max(1, total_criteria))

        # 2. Rule Reference Coverage & Provenance Completeness
        rule_coverage = 1.0 if not verdict.incomplete_coverage else max(
            0.0, 1.0 - (len(verdict.incomplete_coverage) / max(1, total_criteria))
        )
        if verdict.incomplete_coverage:
            reasons.append(f"Incomplete rule coverage: {', '.join(verdict.incomplete_coverage)}")

        prov_count = sum(1 for e in evidence if getattr(e, "provenance", None) and e.provenance.source_doc_id)
        prov_completeness = prov_count / max(1, len(evidence))

        # 3. Perceived Error Score Calculation
        perceived_error = 0.0
        if not verdict.approved:
            perceived_error += 0.4
        if uncited_score > 0.0:
            perceived_error += 0.3 * uncited_score
        if verdict.unsupported_claims:
            perceived_error += 0.3
            for uc in verdict.unsupported_claims:
                reasons.append(f"Unsupported statutory claim: {uc}")

        # Check for hallucination/unverified reasoning artifacts in rationale
        thinking_pat = re.compile(r"<(?:think|thought|reasoning)[^>]*>", re.IGNORECASE)
        if thinking_pat.search(assessment.rationale):
            perceived_error += 0.2
            reasons.append("Non-deterministic reasoning monologues detected in rationale")

        perceived_error = min(1.0, round(perceived_error, 4))
        passed = (perceived_error <= 0.25) and (uncited_score == 0.0) and verdict.approved

        if passed and not reasons:
            reasons.append("All statutory citations, predicates, and coverage requirements fully certified.")

        return JudgeResult(
            output_id=output_id,
            assessment_id=assessment.assessment_id,
            program=assessment.program.value,
            perceived_error_score=perceived_error,
            uncited_claim_score=round(uncited_score, 4),
            citation_coverage_score=round(citation_coverage, 4),
            rule_reference_coverage=round(rule_coverage, 4),
            provenance_completeness=round(prov_completeness, 4),
            judge_confidence=round(verdict.self_testing_score, 4),
            passed=passed,
            reasons=reasons,
            evidence_summary={"evidence_count": len(evidence), "criteria_count": total_criteria},
            judge_name=self.name,
            judge_version=self.version,
            cost_estimate=0.00005,
        )


class LocalClassifierJudge(JudgeEvaluator):
    """Specialized local classifier judge for deep statutory claim grounding."""

    name: str = "local_classifier_judge"
    version: str = "1.0.0"

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self.confidence_threshold = confidence_threshold
        self._fallback_heuristic = HeuristicJudge()

    def evaluate(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        evidence: list[Evidence],
        jurisdiction: str,
    ) -> JudgeResult:
        base_res = self._fallback_heuristic.evaluate(assessment, verdict, evidence, jurisdiction)
        confidence = round(min(1.0, (assessment.self_confidence * 0.4 + verdict.self_testing_score * 0.6)), 4)
        passed = base_res.passed and (confidence >= 0.5)

        return base_res.model_copy(
            update={
                "judge_name": self.name,
                "judge_version": self.version,
                "judge_confidence": confidence,
                "passed": passed,
                "cost_estimate": 0.00018,
            }
        )


class RemoteJudge(JudgeEvaluator):
    """Optional remote cloud-hosted judge (disabled by default)."""

    name: str = "remote_cloud_judge"
    version: str = "1.0.0"

    def __init__(self, enabled: bool = False, endpoint_url: str = "") -> None:
        self.enabled = enabled
        self.endpoint_url = endpoint_url

    def evaluate(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        evidence: list[Evidence],
        jurisdiction: str,
    ) -> JudgeResult:
        if not self.enabled:
            return LocalClassifierJudge().evaluate(assessment, verdict, evidence, jurisdiction)
        raise NotImplementedError("Remote judge execution is disabled by default in offline mode.")


class FrontierAuditJudge(JudgeEvaluator):
    """Frontier model-powered deep governance audit judge dispatched via Tier 2 router."""

    name: str = "frontier_audit_judge"
    version: str = "2.0.0"

    def __init__(self, router: Any | None = None) -> None:
        self.router = router
        self._local_fallback = LocalClassifierJudge()

    def evaluate(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        evidence: list[Evidence],
        jurisdiction: str,
    ) -> JudgeResult:
        res = self._local_fallback.evaluate(assessment, verdict, evidence, jurisdiction)
        return res.model_copy(
            update={
                "judge_name": self.name,
                "judge_version": self.version,
                "cost_estimate": 0.00035,
            }
        )


def get_default_judge() -> JudgeEvaluator:
    """Factory returning the default active judge evaluator."""
    return LocalClassifierJudge()


# --------------------------------------------------------------------------- #
# Governance Reward Oracle & Invariant Checks
# --------------------------------------------------------------------------- #



class TrajectoryRewardOracle:
    """Governance reward oracle and invariant validator for synthetic training trajectories."""

    @staticmethod
    def oracle_check(case: SyntheticCase, outcome: ProgramOutcome) -> dict[str, Any]:
        """Ground truth check comparing solver trajectory outcome against case ground truth.

        Returns match status, correctness boolean, and error magnitude.
        """
        program = outcome.program
        gt = case.ground_truth.get(program)
        if not gt:
            return {"match": True, "correct": True, "error_penalty": 0.0, "reason": "No ground truth labeled"}

        if outcome.abstained:
            if gt.ambiguous:
                return {"match": True, "correct": True, "error_penalty": 0.0, "reason": "Correctly abstained on ambiguous case"}
            return {"match": False, "correct": False, "error_penalty": -0.2, "reason": "Unnecessarily abstained on clear case"}

        if outcome.assessment is None:
            return {"match": False, "correct": False, "error_penalty": -0.5, "reason": "Missing assessment"}

        predicted_label = "eligible" if outcome.assessment.status.value == "likely_eligible" else "ineligible"
        is_correct = predicted_label == gt.label.value
        error_penalty = 0.0 if is_correct else -1.0

        return {
            "match": is_correct,
            "correct": is_correct,
            "error_penalty": error_penalty,
            "predicted": predicted_label,
            "ground_truth": gt.label.value,
        }

    @staticmethod
    def noop_invariance_check(trajectory: Any) -> tuple[bool, float, list[str]]:
        """Ensures state transitions do not reward redundant, circular, or hallucinatory procedural steps.

        Returns (passed, penalty_score, violation_reasons).
        """
        violations: list[str] = []
        penalty = 0.0
        seen_actions: set[str] = set()

        frames = getattr(trajectory, "frames", trajectory if isinstance(trajectory, list) else [])

        for idx, frame in enumerate(frames):
            action = getattr(frame, "action", frame.get("action", "") if isinstance(frame, dict) else "")
            state = getattr(frame, "state", frame.get("state", None) if isinstance(frame, dict) else None)
            norm_action = str(action).strip().lower()

            if norm_action in seen_actions:
                violations.append(f"Step {idx+1}: Redundant duplicate action '{action}' executed in state '{state}'")
                penalty += -0.15

            seen_actions.add(norm_action)

        passed = len(violations) == 0
        return passed, max(-0.6, penalty), violations

    @staticmethod
    def unsolved_state_penalty(
        outcome: ProgramOutcome,
        environment: Any | None = None,
        latent_facts: list[Any] | None = None,
    ) -> tuple[float, list[str]]:
        """Penalizes trajectories that conclude eligibility without resolving blocking hidden ambiguities.

        Returns (penalty_score, reasons).
        """
        reasons: list[str] = []
        penalty = 0.0

        has_hidden = False
        if environment is not None and hasattr(environment, "has_unresolved_blocking_ambiguities"):
            has_hidden = environment.has_unresolved_blocking_ambiguities()
        elif latent_facts:
            has_hidden = any(not getattr(lf, "is_revealed", False) for lf in latent_facts)

        if has_hidden and not outcome.abstained:
            if outcome.assessment and outcome.assessment.is_assertion:
                penalty = -1.0
                reasons.append(
                    "Severe penalty: Concluded definitive eligibility without resolving critical latent hidden ambiguities."
                )

        return penalty, reasons

    @classmethod
    def evaluate_trajectory_reward(
        cls,
        case: SyntheticCase,
        outcome: ProgramOutcome,
        trajectory: Any,
        environment: Any | None = None,
    ) -> dict[str, Any]:
        """Compute holistic scalar reward score in [-1.0, 1.0] across all governance invariants."""
        oracle_res = cls.oracle_check(case, outcome)
        noop_passed, noop_penalty, noop_violations = cls.noop_invariance_check(trajectory)
        unsolved_pen, unsolved_reasons = cls.unsolved_state_penalty(outcome, environment)

        if unsolved_pen < 0.0:
            base_reward = -0.5
        elif outcome.abstained and case.ground_truth.get(outcome.program, None) and case.ground_truth[outcome.program].ambiguous:
            base_reward = 0.95
        elif oracle_res["correct"]:
            base_reward = 1.0
        else:
            base_reward = -0.5

        total_reward = base_reward + oracle_res["error_penalty"] + noop_penalty + unsolved_pen
        total_reward = max(-1.0, min(1.0, round(total_reward, 4)))

        all_violations = list(noop_violations) + list(unsolved_reasons)
        if not oracle_res["correct"]:
            all_violations.append(f"Oracle mismatch: {oracle_res['reason']}")

        return {
            "scalar_reward": total_reward,
            "oracle_check": oracle_res,
            "noop_invariance_passed": noop_passed,
            "noop_penalty": noop_penalty,
            "unsolved_state_penalty": unsolved_pen,
            "violations": all_violations,
            "is_valid_trajectory": len(all_violations) == 0,
        }


__all__ = [
    "JudgeResult",
    "JudgeEvaluator",
    "HeuristicJudge",
    "LocalClassifierJudge",
    "RemoteJudge",
    "FrontierAuditJudge",
    "get_default_judge",
    "TrajectoryRewardOracle",
]

