"""Sandbox Telemetry & Terminal Exploit / Reward-Gaming Detection Engine.

Detects terminal exploit loops, command flooding, privilege escalation attempts,
and environment exfiltration, triggering automated session quarantine and bounded evidence logs.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Documenting an anomalous or prohibited execution attempt."""

    event_type: str
    command: str
    severity: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    timestamp: float
    details: str
    mitigation_applied: str


@dataclass
class ExploitTelemetryStats:
    """Aggregated exploit detection statistics."""

    total_commands_inspected: int = 0
    identical_command_loops_detected: int = 0
    output_flooding_events: int = 0
    privilege_escalation_events: int = 0
    exfiltration_attempts: int = 0
    rapid_noop_loops: int = 0
    quarantined: bool = False
    security_events: list[SecurityEvent] = field(default_factory=list)


class ExploitDetectionEngine:
    """Analyzes command streams and execution outcomes to trap gaming and terminal exploit loops."""

    def __init__(
        self,
        loop_threshold: int = 3,
        max_output_bytes: int = 1048576,  # 1 MB
        max_evidence_entries: int = 50,
    ) -> None:
        self.loop_threshold = loop_threshold
        self.max_output_bytes = max_output_bytes
        self.max_evidence_entries = max_evidence_entries

        self._recent_commands: collections.deque[str] = collections.deque(maxlen=20)
        self._command_timestamps: collections.deque[float] = collections.deque(maxlen=20)
        self.evidence_log: collections.deque[SecurityEvent] = collections.deque(maxlen=max_evidence_entries)
        self.stats = ExploitTelemetryStats()
        self.is_quarantined = False

    def inspect_command(self, command_str: str) -> tuple[bool, str | None]:
        """Inspect command before execution. Returns (is_safe, violation_reason)."""
        self.stats.total_commands_inspected += 1
        now = time.time()
        cmd_clean = command_str.strip()

        if self.is_quarantined:
            return False, "Execution denied: Sandbox session is quarantined due to security violations."

        # 1. Repeated Identical Command Loop Detection
        recent_count = sum(1 for c in self._recent_commands if c == cmd_clean)
        if recent_count >= self.loop_threshold:
            self.stats.identical_command_loops_detected += 1
            evt = self._record_violation(
                event_type="IDENTICAL_COMMAND_LOOP",
                command=cmd_clean,
                severity="HIGH",
                details=f"Command repeated {recent_count + 1} times sequentially without state divergence.",
                mitigation="TERMINATE_AND_QUARANTINE",
            )
            self._apply_quarantine()
            return False, f"Exploit loop detected: {evt.details}"

        # 2. Environment Exfiltration Detection
        exfil_patterns = ["printenv", "env |", "export -p", ".env", "AWS_SECRET", "TOKEN="]
        if any(p in cmd_clean for p in exfil_patterns):
            self.stats.exfiltration_attempts += 1
            evt = self._record_violation(
                event_type="ENV_EXFILTRATION_ATTEMPT",
                command=cmd_clean,
                severity="CRITICAL",
                details="Attempt to inspect or exfiltrate environment variables/credentials.",
                mitigation="TERMINATE_AND_QUARANTINE",
            )
            self._apply_quarantine()
            return False, f"Security violation: {evt.details}"

        # 3. Privilege Escalation Attempts
        priv_patterns = ["sudo ", "su -", "chmod +s", "setuid", "chown root"]
        if any(p in cmd_clean for p in priv_patterns):
            self.stats.privilege_escalation_events += 1
            evt = self._record_violation(
                event_type="PRIVILEGE_ESCALATION",
                command=cmd_clean,
                severity="CRITICAL",
                details="Attempted privilege escalation in sandboxed environment.",
                mitigation="TERMINATE_AND_QUARANTINE",
            )
            self._apply_quarantine()
            return False, f"Security violation: {evt.details}"

        # 4. Rapid No-Op Loops (e.g. echo 1, true, :, called in rapid bursts)
        noop_cmds = [":", "true", "echo 1", "sleep 0"]
        if cmd_clean in noop_cmds:
            if len(self._command_timestamps) >= 5 and (now - self._command_timestamps[-5]) < 1.0:
                self.stats.rapid_noop_loops += 1
                evt = self._record_violation(
                    event_type="RAPID_NOOP_LOOP",
                    command=cmd_clean,
                    severity="MEDIUM",
                    details="Rapid no-op command loop detected.",
                    mitigation="TERMINATE_LOOP",
                )
                return False, f"Exploit warning: {evt.details}"

        self._recent_commands.append(cmd_clean)
        self._command_timestamps.append(now)
        return True, None

    def inspect_output(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        """Inspect and truncate excessive output flooding to prevent token exhaustion."""
        out_bytes = len(stdout.encode("utf-8", errors="ignore"))
        flooded = False

        if out_bytes > self.max_output_bytes:
            self.stats.output_flooding_events += 1
            flooded = True
            trunc_notice = f"\n[... TRUNCATED: Output exceeded {self.max_output_bytes} bytes ...]\n"
            # Keep head and tail
            head = stdout[: self.max_output_bytes // 2]
            tail = stdout[-self.max_output_bytes // 2 :]
            stdout = head + trunc_notice + tail

            self._record_violation(
                event_type="OUTPUT_FLOODING",
                command="[STDOUT_STREAM]",
                severity="MEDIUM",
                details=f"Output volume ({out_bytes} bytes) exceeded ceiling ({self.max_output_bytes} bytes).",
                mitigation="BOUNDED_TRUNCATION",
            )

        return stdout, stderr, flooded

    def _record_violation(
        self,
        event_type: str,
        command: str,
        severity: str,
        details: str,
        mitigation: str,
    ) -> SecurityEvent:
        evt = SecurityEvent(
            event_type=event_type,
            command=command,
            severity=severity,
            timestamp=time.time(),
            details=details,
            mitigation_applied=mitigation,
        )
        self.evidence_log.append(evt)
        self.stats.security_events.append(evt)
        logger.error(f"[ExploitDetectionEngine] {severity} {event_type}: {details} (Action: {mitigation})")
        return evt

    def _apply_quarantine(self) -> None:
        self.is_quarantined = True
        self.stats.quarantined = True

    def get_evidence_log(self) -> list[SecurityEvent]:
        return list(self.evidence_log)


__all__ = [
    "SecurityEvent",
    "ExploitTelemetryStats",
    "ExploitDetectionEngine",
]
