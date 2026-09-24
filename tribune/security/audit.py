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
    LATERAL_ESCALATION_ATTEMPT = "LATERAL_ESCALATION_ATTEMPT"
    # Hardened-memory provenance events (additive; existing consumers unaffected).
    CONSOLIDATION_SIGNED = "CONSOLIDATION_SIGNED"
    CONSOLIDATION_REJECTED = "CONSOLIDATION_REJECTED"
    HMAC_VERIFICATION_FAILURE = "HMAC_VERIFICATION_FAILURE"
    UNSIGNED_VECTOR_BLOCKED = "UNSIGNED_VECTOR_BLOCKED"
    INTENT_GRAPH_ALERT = "INTENT_GRAPH_ALERT"
    SESSION_SUSPENDED = "SESSION_SUSPENDED"
    TOOL_LOOP_DETECTED = "TOOL_LOOP_DETECTED"
    # Sentinel security and isolation events
    COMMAND_REQUESTED = "command_requested"
    COMMAND_ALLOWED = "command_allowed"
    COMMAND_DENIED = "command_denied"
    NETWORK_REQUESTED = "network_requested"
    NETWORK_ALLOWED = "network_allowed"
    NETWORK_DENIED = "network_denied"
    SECRET_REDACTED = "secret_redacted"
    SURROGATE_TOKEN_ISSUED = "surrogate_token_issued"
    CONTAINER_SPAWNED = "container_spawned"
    CONTAINER_EXITED = "container_exited"


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

    @property
    def events(self) -> list[SecurityAuditEvent]:
        """Return shallow copy of recorded security events."""
        with self._lock:
            return list(self._events)

    def record(self, event: SecurityAuditEvent) -> SecurityAuditEvent:
        """Record an already constructed SecurityAuditEvent."""
        return self.record_event(
            event_type=event.event_type,
            source=event.source,
            message=event.message,
            severity=event.severity,
            details=event.details,
            case_id=event.case_id,
        )

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
    "sign_consolidated_node",
    "verify_consolidated_node",
]


def sign_consolidated_node(
    source_ids: list[str],
    schema_payload: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """HMAC-sign a consolidated schema payload and record it in the audit chain.

    Returns dict with signature, key_id, payload_digest, source_ids,
    timestamp, previous_hash, entry_digest. Secrets come from
    ``TRIBUNE_HMAC_SECRET`` (never hardcoded).
    """
    from .provenance import get_provenance_log, payload_digest_hex

    log = get_provenance_log()
    entry = log.sign_and_append(source_ids, schema_payload, timestamp=timestamp)
    record_security_event(
        event_type=SecurityEventType.CONSOLIDATION_SIGNED,
        source="tribune.security.audit.sign_consolidated_node",
        message="Consolidated node signed and chained.",
        severity="LOW",
        details={
            "key_id": entry.key_id,
            "payload_digest": entry.payload_digest,
            "source_ids": entry.source_ids,
            "timestamp": entry.timestamp,
            "previous_hash": entry.previous_hash[:16] if entry.previous_hash else "",
            "entry_digest": entry.entry_digest[:16],
        },
    )
    return {
        "signature": entry.signature,
        "key_id": entry.key_id,
        "payload_digest": payload_digest_hex(schema_payload),
        "source_ids": entry.source_ids,
        "timestamp": entry.timestamp,
        "previous_hash": entry.previous_hash,
        "entry_digest": entry.entry_digest,
    }


def verify_consolidated_node(
    source_ids: list[str],
    schema_payload: dict[str, Any],
    timestamp: str,
    signature: str,
    key_id: str,
) -> bool:
    """Verify HMAC + audit-log presence. Fails closed (False) on any mismatch."""
    from .provenance import get_provenance_log, payload_digest_hex

    log = get_provenance_log()
    ok = log.verify_entry_signature(source_ids, schema_payload, timestamp, signature, key_id)
    if not ok:
        record_security_event(
            event_type=SecurityEventType.HMAC_VERIFICATION_FAILURE,
            source="tribune.security.audit.verify_consolidated_node",
            message="HMAC verification failed for consolidated node.",
            severity="HIGH",
            details={"key_id": key_id, "source_ids": source_ids},
        )
        return False
    entry = log.lookup(payload_digest_hex(schema_payload))
    if entry is None:
        record_security_event(
            event_type=SecurityEventType.UNSIGNED_VECTOR_BLOCKED,
            source="tribune.security.audit.verify_consolidated_node",
            message="No audit-log entry for payload digest; blocking activation.",
            severity="HIGH",
            details={"payload_digest": payload_digest_hex(schema_payload)},
        )
        return False
    return True
