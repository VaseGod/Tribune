"""Structured Five-Track Operational State Bank for Proactive Memory Sidecar.

Maintains execution state across 5 distinct operational tracks without directly
polluting the action model's context window:
1. current_environment (working directory, paths, visible files, env vars)
2. successful_mutations (verified file edits, created artifacts, passed tests)
3. unmet_task_requirements (unsatisfied objectives from seedset/verifier)
4. active_subgoals (immediate operational sequence, hypotheses, next commands)
5. failed_commands_cache (non-zero exits, parsed stderr, recurrence counts)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvironmentState:
    """Track A: Observed system and directory environment."""

    working_directory: str = "/workspace"
    path_variables: list[str] = field(default_factory=lambda: ["/usr/local/bin", "/usr/bin", "/bin"])
    active_virtual_environment: str | None = None
    visible_files: list[str] = field(default_factory=list)
    relevant_environment_variables: dict[str, str] = field(default_factory=dict)
    last_observed_filesystem_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationRecord:
    """Track B: Verified filesystem or environment mutation."""

    mutation_id: str
    mutation_type: str  # "file_created" | "file_modified" | "test_passed" | "download_verified"
    target: str
    timestamp: float = field(default_factory=time.time)
    verified: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequirement:
    """Track C: Formal task objective derived from seed specification and verifier."""

    requirement_id: str
    description: str
    satisfied: bool = False
    evidence: str = ""
    severity: str = "HIGH"  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    missing_artifacts: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    reproducible_command_hint: str = ""


@dataclass
class Subgoal:
    """Track D: Active operational sequence and hypotheses."""

    subgoal_id: str
    description: str
    completed: bool = False
    next_commands: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class FailedCommandRecord:
    """Entry in the failed command recurrence cache."""

    command_fingerprint: str
    raw_command: str
    last_exit_code: int
    parsed_stderr: str
    recurrence_count: int
    first_seen_turn: int
    last_seen_turn: int


@dataclass
class FailedCommandCache:
    """Track E: Non-zero exit command history and fingerprinting."""

    entries: dict[str, FailedCommandRecord] = field(default_factory=dict)

    @staticmethod
    def fingerprint(command: str) -> str:
        """Normalize command string to detect identical or near-identical attempts."""
        # Strip leading/trailing whitespace, multiple spaces, and common prefixes
        norm = command.strip().lower()
        norm = re.sub(r"\s+", " ", norm)
        # Normalize relative path indicators
        norm = re.sub(r"\./", "", norm)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    def record_failure(
        self,
        command: str,
        exit_code: int,
        stderr: str,
        turn_index: int,
    ) -> FailedCommandRecord:
        """Register a command execution failure and increment recurrence count."""
        fp = self.fingerprint(command)
        if fp in self.entries:
            rec = self.entries[fp]
            rec.recurrence_count += 1
            rec.last_seen_turn = turn_index
            rec.last_exit_code = exit_code
            rec.parsed_stderr = stderr[:300]
        else:
            rec = FailedCommandRecord(
                command_fingerprint=fp,
                raw_command=command,
                last_exit_code=exit_code,
                parsed_stderr=stderr[:300],
                recurrence_count=1,
                first_seen_turn=turn_index,
                last_seen_turn=turn_index,
            )
            self.entries[fp] = rec
        return rec

    def is_known_failure(self, command: str) -> FailedCommandRecord | None:
        """Check if command matches a previous failed command fingerprint."""
        fp = self.fingerprint(command)
        return self.entries.get(fp)


@dataclass
class MemoryStateBank:
    """Unified Five-Track Execution Memory State Bank."""

    current_environment: EnvironmentState = field(default_factory=EnvironmentState)
    successful_mutations: list[MutationRecord] = field(default_factory=list)
    unmet_task_requirements: list[TaskRequirement] = field(default_factory=list)
    active_subgoals: list[Subgoal] = field(default_factory=list)
    failed_commands_cache: FailedCommandCache = field(default_factory=FailedCommandCache)
    updated_at_turn: int = 0
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_unmet_requirements(self) -> bool:
        """Return True if any critical/high task requirement remains unsatisfied."""
        return any(not req.satisfied for req in self.unmet_task_requirements)

    def mark_requirement_satisfied(self, requirement_id: str, evidence: str = "") -> bool:
        """Mark a specific requirement as completed."""
        for req in self.unmet_task_requirements:
            if req.requirement_id == requirement_id:
                req.satisfied = True
                req.evidence = evidence
                return True
        return False
