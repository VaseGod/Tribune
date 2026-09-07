"""Security Audit Logging Subsystem.

Provides structured, append-only security event auditing across Tribune subsystems:
1. MODEL_FALLBACK_DETECTED: Upstream safety or routing fallback to legacy checkpoints.
2. UNGROUNDED_ASSERTION_VIOLATION: Factual or code claims lacking cited provenance.
3. DEFECT_ESCALATED: Environment, test, or specification defect halts.
4. GRADER_AWARENESS_ALERT: Eval harness gaming or grader awareness detected.
5. ASTRA_CLASS_CONTAINMENT_BREACH: Decoy tripwire access or sandbox escape attempts.

Integrates with the governance audit hash chain and tracing instrumentation.
"""

from __future__ import annotations

import enum
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..governance.audit import AuditLog, sanitize_audit_data
from ..instrumentation import tracing
from ..types import SMState

logger = logging.getLogger(__name__)


class SecurityEventType(str, enum.Enum):
    MODEL_FALLBACK_DETECTED = "MODEL_FALLBACK_DETECTED"
    UNGROUNDED_ASSERTION_VIOLATION = "UNGROUNDED_ASSERTION_VIOLATION"
    DEFECT_ESCALATED = "DEFECT_ESCALATED"
    GRADER_AWARENESS_ALERT = "GRADER_AWARENESS_ALERT"
    ASTRA_CLASS_CONTAINMENT_BREACH = "ASTRA_CLASS_CONTAINMENT_BREACH"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"


@dataclass
class SecurityAuditEvent:
    """Structured security audit event record."""

    event_type: SecurityEventType | str
    severity: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    source: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    case_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"sec_evt_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(self.event_type, enum.Enum):
            data["event_type"] = self.event_type.value
        return sanitize_audit_data(data)


class SecurityAuditLogger:
    """Thread-safe, append-only security audit log."""

    def __init__(self, governance_audit_log: AuditLog | None = None) -> None:
        self._events: list[SecurityAuditEvent] = []
        self._governance_audit = governance_audit_log
        self._lock = threading.RLock()

    def record_event(
        self,
        event_type: SecurityEventType | str,
        source: str,
        message: str,
        severity: str = "HIGH",
        details: dict[str, Any] | None = None,
        case_id: str | None = None,
    ) -> SecurityAuditEvent:
        """Record and dispatch an immutable security event."""
        with self._lock:
            event = SecurityAuditEvent(
                event_type=event_type,
                severity=severity,
                source=source,
                message=message,
                details=details or {},
                case_id=case_id,
            )
            self._events.append(event)

            evt_name = event_type.value if isinstance(event_type, enum.Enum) else str(event_type)

            # Emit to tracing instrumentation
            tracing.log(
                "security_audit_event",
                event_type=evt_name,
                severity=severity,
                source=source,
                message=message,
                details=event.details,
                case_id=case_id,
            )

            # Log to python logger
            log_msg = f"[SECURITY-AUDIT] [{evt_name}] [{severity}] {source}: {message} details={event.details}"
            if severity == "CRITICAL":
                logger.critical(log_msg)
            elif severity == "HIGH":
                logger.error(log_msg)
            elif severity == "MEDIUM":
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

            # If associated with a case and governance audit log is available, chain it
            if case_id and self._governance_audit:
                try:
                    self._governance_audit.append(
                        case_id=case_id,
                        state=SMState.ABSTAIN if severity in ("HIGH", "CRITICAL") else SMState.PLAN,
                        agent=source,
                        action=f"security_event:{evt_name}",
                        payload={
                            "event_type": evt_name,
                            "severity": severity,
                            "message": message,
                            "details": json.dumps(event.details),
                        },
                    )
                except Exception as exc:
                    logger.error(f"Failed to record security event in governance audit: {exc}")

            return event

    def get_events(
        self,
        event_type: SecurityEventType | str | None = None,
        case_id: str | None = None,
    ) -> list[SecurityAuditEvent]:
        """Query security audit events with optional filters."""
        with self._lock:
            result = list(self._events)
            if event_type is not None:
                match_val = event_type.value if isinstance(event_type, enum.Enum) else str(event_type)
                result = [
                    e for e in result
                    if (e.event_type.value if isinstance(e.event_type, enum.Enum) else str(e.event_type)) == match_val
                ]
            if case_id is not None:
                result = [e for e in result if e.case_id == case_id]
            return result

    def clear(self) -> None:
        """Clear recorded events (used for test isolation)."""
        with self._lock:
            self._events.clear()


_GLOBAL_SECURITY_AUDIT = SecurityAuditLogger()


def get_security_audit_logger() -> SecurityAuditLogger:
    """Return singleton security audit logger instance."""
    return _GLOBAL_SECURITY_AUDIT


def record_security_event(
    event_type: SecurityEventType | str,
    source: str,
    message: str,
    severity: str = "HIGH",
    details: dict[str, Any] | None = None,
    case_id: str | None = None,
) -> SecurityAuditEvent:
    """Convenience helper to record a structured security event."""
    return _GLOBAL_SECURITY_AUDIT.record_event(
        event_type=event_type,
        source=source,
        message=message,
        severity=severity,
        details=details,
        case_id=case_id,
    )


__all__ = [
    "SecurityEventType",
    "SecurityAuditEvent",
    "SecurityAuditLogger",
    "get_security_audit_logger",
    "record_security_event",
]
