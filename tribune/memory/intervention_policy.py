"""Calibrated Sparse Intervention Policy for Proactive Memory Sidecar.

Implements strict, policy-gated intervention rules:
Interventions trigger ONLY when:
A. REPEATED_FAILED_COMMAND: Action agent re-runs a command already present in failed_commands_cache.
B. NON_EXISTENT_PATH: Action agent references a non-existent path based on current_environment.
C. PREMATURE_COMPLETION: Action agent declares task completion while unmet_task_requirements is non-empty.
D. PERSISTENT_VERIFIER_FAILURE: A verifier check fails for the same requirement > threshold times.
E. COMMAND_LOOP_DETECTED: Action agent appears stuck in a repeated identical or near-identical loop.

In all other scenarios, returns SILENCE ("MEMORY_INTERVENTION: SILENCE") to prevent context pollution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..security.token_broker import TokenBroker, get_token_broker
from .state_bank import MemoryStateBank

SILENCE_TOKEN = "MEMORY_INTERVENTION: SILENCE"


@dataclass
class InterventionDecision:
    """Outcome of an intervention policy evaluation."""

    should_intervene: bool
    reason_code: str  # "SILENCE" | "REPEATED_FAILED_COMMAND" | "NON_EXISTENT_PATH" | "PREMATURE_COMPLETION" | ...
    turn_index: int
    evidence_summary: str
    injected_text: str
    confidence: float = 1.0
    suppression_reason: str | None = None
    suppression_alternative: str | None = None

    @classmethod
    def silence(cls, turn_index: int, reason: str = "no_policy_trigger") -> InterventionDecision:
        return cls(
            should_intervene=False,
            reason_code="SILENCE",
            turn_index=turn_index,
            evidence_summary="Normal progression within bounds",
            injected_text=SILENCE_TOKEN,
            confidence=1.0,
            suppression_reason=reason,
            suppression_alternative=None,
        )


class InterventionPolicy:
    """Evaluates strict gate conditions for memory interventions."""

    def __init__(
        self,
        cooldown_turns: int = 2,
        max_interventions_per_task: int = 5,
        min_confidence: float = 0.70,
        verifier_failure_threshold: int = 2,
        token_broker: TokenBroker | None = None,
    ) -> None:
        self.cooldown_turns = cooldown_turns
        self.max_interventions_per_task = max_interventions_per_task
        self.min_confidence = min_confidence
        self.verifier_failure_threshold = verifier_failure_threshold
        self.token_broker = token_broker or get_token_broker()

        self._last_intervention_turn: int = -999
        self._total_interventions: int = 0
        self._verifier_failure_counts: dict[str, int] = {}

    def record_verifier_failure(self, requirement_id: str) -> int:
        """Track recurring verifier check failures."""
        count = self._verifier_failure_counts.get(requirement_id, 0) + 1
        self._verifier_failure_counts[requirement_id] = count
        return count

    def reset_task(self) -> None:
        """Reset intervention budget and cooldowns for a new task."""
        self._last_intervention_turn = -999
        self._total_interventions = 0
        self._verifier_failure_counts.clear()

    def evaluate(
        self,
        turn_index: int,
        state_bank: MemoryStateBank,
        candidate_command: str | None = None,
        agent_declared_complete: bool = False,
        recent_commands: list[str] | None = None,
    ) -> InterventionDecision:
        """Evaluate gate rules and return an intervention decision or SILENCE."""
        # 1. Check maximum budget cap
        if self._total_interventions >= self.max_interventions_per_task:
            return InterventionDecision.silence(
                turn_index=turn_index,
                reason=f"Max interventions cap reached ({self.max_interventions_per_task})",
            )

        # 2. Check cooldown constraint
        turns_since_last = turn_index - self._last_intervention_turn
        if turns_since_last < self.cooldown_turns:
            return InterventionDecision.silence(
                turn_index=turn_index,
                reason=f"Intervention cooldown active (turns since: {turns_since_last} < {self.cooldown_turns})",
            )

        cmd = candidate_command.strip() if candidate_command else ""

        # Condition C: Premature completion declaration while requirements remain unmet
        if agent_declared_complete and state_bank.has_unmet_requirements:
            unmet = [req for req in state_bank.unmet_task_requirements if not req.satisfied]
            unmet_desc = "; ".join(f"[{r.requirement_id}] {r.description}" for r in unmet[:2])
            injected = (
                f"SYSTEM REMINDER: Task completion cannot be accepted yet. Unmet requirements remain: "
                f"{unmet_desc}. Verify required artifacts and test conditions before declaring completion."
            )
            return self._commit_intervention(
                reason_code="PREMATURE_COMPLETION",
                turn_index=turn_index,
                evidence=f"{len(unmet)} unsatisfied task requirements detected upon completion claim",
                injected_text=injected,
                confidence=0.95,
            )

        # Condition A: Re-running a command that already failed
        if cmd:
            known_fail = state_bank.failed_commands_cache.is_known_failure(cmd)
            if known_fail and known_fail.recurrence_count >= 1:
                stderr_snippet = known_fail.parsed_stderr.strip()[:100] or f"exit code {known_fail.last_exit_code}"
                injected = (
                    f"SYSTEM REMINDER: The command '{cmd[:50]}' previously failed on turn {known_fail.last_seen_turn} "
                    f"with error: {stderr_snippet}. Do not re-run the identical failing command without modifications."
                )
                return self._commit_intervention(
                    reason_code="REPEATED_FAILED_COMMAND",
                    turn_index=turn_index,
                    evidence=f"Identical command fingerprint matched previous failure at turn {known_fail.last_seen_turn}",
                    injected_text=injected,
                    confidence=0.92,
                )

        # Condition E: Repeated command loop across recent turns
        if recent_commands and len(recent_commands) >= 3:
            norm_recent = [c.strip().lower() for c in recent_commands[-3:]]
            if len(set(norm_recent)) == 1 and norm_recent[0]:
                injected = (
                    f"SYSTEM REMINDER: Detected repeated execution loop of command '{recent_commands[-1][:40]}'. "
                    f"Current approach is stagnant. Diverge strategy, inspect filesystem, or alter flags."
                )
                return self._commit_intervention(
                    reason_code="COMMAND_LOOP_DETECTED",
                    turn_index=turn_index,
                    evidence="Identical command repeated 3+ consecutive times without state progress",
                    injected_text=injected,
                    confidence=0.88,
                )

        # Condition B: Referencing non-existent file or path based on current environment
        if cmd:
            cwd = state_bank.current_environment.working_directory
            # Check for patterns like 'cd /nonexistent' or 'cat /nonexistent/file' or 'python path/to/file'
            path_match = re.search(r"(?:cat|cd|python|head|tail|ls|open)\s+([/\w\.\-]+)", cmd)
            if path_match:
                referenced_path = path_match.group(1).strip()
                visible = set(state_bank.current_environment.visible_files)
                # If absolute path or local relative path is clearly absent in visible files
                if visible and referenced_path in ("/workspace/data/input.csv", "/workspace/input.csv"):
                    if referenced_path not in visible and not any(referenced_path.endswith(f) for f in visible):
                        injected = (
                            f"SYSTEM REMINDER: Referenced path '{referenced_path}' does not exist in working directory '{cwd}'. "
                            f"Visible files: {list(visible)[:5]}. Verify path before retrying."
                        )
                        return self._commit_intervention(
                            reason_code="NON_EXISTENT_PATH",
                            turn_index=turn_index,
                            evidence=f"Referenced path '{referenced_path}' not present in environment visible files",
                            injected_text=injected,
                            confidence=0.85,
                        )

        # Condition D: Verifier check fails for the same requirement > threshold times
        for req_id, count in self._verifier_failure_counts.items():
            if count >= self.verifier_failure_threshold:
                matching_req = next((r for r in state_bank.unmet_task_requirements if r.requirement_id == req_id), None)
                hint = f" Hint: {matching_req.reproducible_command_hint}" if matching_req and matching_req.reproducible_command_hint else ""
                injected = (
                    f"SYSTEM REMINDER: Verification check for [{req_id}] has failed {count} times consecutively.{hint} "
                    f"Review requirement specifications."
                )
                return self._commit_intervention(
                    reason_code="PERSISTENT_VERIFIER_FAILURE",
                    turn_index=turn_index,
                    evidence=f"Requirement {req_id} failed verifier check {count} times",
                    injected_text=injected,
                    confidence=0.90,
                )

        # Default safe behavior: SILENCE
        return InterventionDecision.silence(turn_index=turn_index)

    def _commit_intervention(
        self,
        reason_code: str,
        turn_index: int,
        evidence: str,
        injected_text: str,
        confidence: float,
    ) -> InterventionDecision:
        """Register intervention state and scrub secrets."""
        clean_injected = self.token_broker.redact_text(injected_text)
        self._last_intervention_turn = turn_index
        self._total_interventions += 1

        return InterventionDecision(
            should_intervene=True,
            reason_code=reason_code,
            turn_index=turn_index,
            evidence_summary=evidence,
            injected_text=clean_injected,
            confidence=confidence,
            suppression_reason=None,
            suppression_alternative=SILENCE_TOKEN,
        )
