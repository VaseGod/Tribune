"""Bounded Verifier Self-Testing and Verification Bounding.

Implements bounded reflection loops with explicit token and recursion depth ceilings,
yielding a structured AbstainResult when legal ambiguity is intractable or resources are breached,
rather than unbounded deliberation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import Assessment, Evidence, VerifierVerdict


@dataclass(frozen=True)
class AbstainResult:
    """Structured result returned when verification bounding limits are breached or ambiguity is intractable."""

    abstained: bool = True
    reason: str = ""
    token_budget_exceeded: bool = False
    recursion_ceiling_hit: bool = False
    total_tokens_consumed: int = 0
    recursion_depth: int = 0
    decision_boundary_reached: bool = False
    intermediate_steps: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    status: str = "abstained"

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "reason": self.reason,
            "token_budget_exceeded": self.token_budget_exceeded,
            "recursion_ceiling_hit": self.recursion_ceiling_hit,
            "total_tokens_consumed": self.total_tokens_consumed,
            "recursion_depth": self.recursion_depth,
            "decision_boundary_reached": self.decision_boundary_reached,
            "intermediate_steps": list(self.intermediate_steps),
            "status": self.status,
        }


class BoundedVerifierSelfTesting:
    """Bounded reflection loop execution for verifier self-testing with strict ceilings."""

    def __init__(
        self,
        token_budget: int = 4096,
        recursion_ceiling: int = 5,
        tokens_per_reflection_step: int = 256,
    ) -> None:
        self.token_budget = token_budget
        self.recursion_ceiling = recursion_ceiling
        self.tokens_per_reflection_step = tokens_per_reflection_step

    def run_bounded_reflection(
        self,
        verifier: Any,
        assessment: Assessment,
        evidence: list[Evidence],
        jurisdiction: str = "EX",
        mock_unresolvable_ambiguity: bool = False,
        simulated_tokens_per_turn: int | None = None,
        force_token_budget_exceeded: bool = False,
    ) -> VerifierVerdict | AbstainResult:
        """Execute bounded multi-step verification reflection loop.

        Guarantees that reflection never recurses indefinitely or burns tokens unboundedly
        when faced with contradictory statutory predicates or intractable factual ambiguity.
        """
        step_tokens = simulated_tokens_per_turn or self.tokens_per_reflection_step
        tokens_consumed = 0
        depth = 0
        intermediate_steps: list[dict[str, Any]] = []

        while depth < self.recursion_ceiling:
            depth += 1
            tokens_consumed += step_tokens

            # Check Token Budget Ceiling
            if force_token_budget_exceeded or tokens_consumed > self.token_budget:
                return AbstainResult(
                    abstained=True,
                    reason=f"Verification token budget exceeded: consumed {tokens_consumed} tokens (limit: {self.token_budget}).",
                    token_budget_exceeded=True,
                    recursion_ceiling_hit=False,
                    total_tokens_consumed=tokens_consumed,
                    recursion_depth=depth,
                    decision_boundary_reached=False,
                    intermediate_steps=tuple(intermediate_steps),
                )

            # Record turn
            step_record = {
                "turn": depth,
                "tokens_turn": step_tokens,
                "cumulative_tokens": tokens_consumed,
                "action": "statutory_self_testing_reflection",
            }
            intermediate_steps.append(step_record)

            if mock_unresolvable_ambiguity:
                # Ambiguity remains unresolvable across reflections
                continue

            # Attempt verification pass
            try:
                verdict = verifier.verify(assessment, evidence, jurisdiction)
                # If a clear decision boundary is reached (verdict approved or definite refusal)
                if verdict.approved or getattr(verdict, "sanity_score", 1.0) == 1.0:
                    return verdict
            except Exception as exc:
                intermediate_steps.append({"turn": depth, "error": str(exc)})

        # Recursion Ceiling Hit without reaching clean decision boundary
        return AbstainResult(
            abstained=True,
            reason=f"Verification recursion ceiling hit: depth {depth} reached without resolving statutory boundary (limit: {self.recursion_ceiling}).",
            token_budget_exceeded=False,
            recursion_ceiling_hit=True,
            total_tokens_consumed=tokens_consumed,
            recursion_depth=depth,
            decision_boundary_reached=False,
            intermediate_steps=tuple(intermediate_steps),
        )


@dataclass(frozen=True)
class StructuredVerificationResult:
    """Structured verification outcome for a specific task requirement."""

    requirement_id: str
    satisfied: bool
    evidence: str = ""
    missing_artifacts: tuple[str, ...] = field(default_factory=tuple)
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    severity: str = "HIGH"
    reproducible_command_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "satisfied": self.satisfied,
            "evidence": self.evidence,
            "missing_artifacts": list(self.missing_artifacts),
            "failed_checks": list(self.failed_checks),
            "severity": self.severity,
            "reproducible_command_hint": self.reproducible_command_hint,
        }


def verify_task_requirements(
    requirements: list[Any],
    execution_artifacts: list[str],
    passed_tests: list[str] | None = None,
) -> list[StructuredVerificationResult]:
    """Evaluate task requirements against observed artifacts and validation checks."""
    results: list[StructuredVerificationResult] = []
    observed_artifacts = set(execution_artifacts)
    passed_set = set(passed_tests or [])

    for req in requirements:
        req_id = getattr(req, "requirement_id", str(req))
        expected_artifacts = getattr(req, "missing_artifacts", []) or getattr(req, "expected_artifacts", [])
        missing = [art for art in expected_artifacts if art not in observed_artifacts]

        failed_checks = []
        expected_checks = getattr(req, "failed_checks", []) or getattr(req, "expected_verifier_conditions", [])
        for chk in expected_checks:
            if chk not in passed_set:
                failed_checks.append(chk)

        satisfied = len(missing) == 0 and len(failed_checks) == 0
        evidence = f"Verified with {len(observed_artifacts)} artifacts present" if satisfied else f"Missing: {missing or failed_checks}"
        severity = getattr(req, "severity", "HIGH")
        hint = getattr(req, "reproducible_command_hint", "") or (f"ls {missing[0]}" if missing else "")

        results.append(
            StructuredVerificationResult(
                requirement_id=req_id,
                satisfied=satisfied,
                evidence=evidence,
                missing_artifacts=tuple(missing),
                failed_checks=tuple(failed_checks),
                severity=severity,
                reproducible_command_hint=hint,
            )
        )
    return results


def feed_unmet_requirements(state_bank: Any, verification_results: list[StructuredVerificationResult]) -> None:
    """Update MemoryStateBank.unmet_task_requirements from verification results."""
    from ..memory.state_bank import TaskRequirement

    updated_requirements: list[TaskRequirement] = []
    for res in verification_results:
        updated_requirements.append(
            TaskRequirement(
                requirement_id=res.requirement_id,
                description=f"Requirement {res.requirement_id}",
                satisfied=res.satisfied,
                evidence=res.evidence,
                severity=res.severity,
                missing_artifacts=list(res.missing_artifacts),
                failed_checks=list(res.failed_checks),
                reproducible_command_hint=res.reproducible_command_hint,
            )
        )
    state_bank.unmet_task_requirements = updated_requirements


__all__ = [
    "AbstainResult",
    "BoundedVerifierSelfTesting",
    "StructuredVerificationResult",
    "verify_task_requirements",
    "feed_unmet_requirements",
]

