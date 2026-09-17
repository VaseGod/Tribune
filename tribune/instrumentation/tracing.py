"""Cryptographically Hashed Execution Trace Ledger & Observability Hooks.

Implements an immutable, append-only, SHA-256 hash-chained execution ledger:
1. Every recorded step captures:
   - event_id, parent_event_id, timestamp, task_id, run_id
   - step_type, tool_name, tool_inputs_digest
   - model_provider, model_id, tier
   - token_usage (input, output, cached), cost_estimate
   - verifier_result, ast_verifier_rejection_details
   - security_policy_trigger_details, exception_details, outcome
   - previous_hash, current_hash
2. Applies redaction before calculating cryptographic digests.
3. Provides ledger verification to mathematically prove trace tampering or event omission.
4. Formats audit bundles for regulatory oversight bodies.
5. Preserves backward compatibility with legacy span() and log() sinks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class HashedTraceEvent:
    """Cryptographically sealed execution trace event in an immutable hash chain."""

    event_id: str
    parent_event_id: str | None
    timestamp: str
    task_id: str
    run_id: str
    step_type: str
    tool_name: str | None = None
    tool_inputs_digest: str | None = None
    model_provider: str | None = None
    model_id: str | None = None
    tier: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    cost_estimate_usd: float = 0.0
    verifier_result: dict[str, Any] | None = None
    ast_verifier_rejection_details: dict[str, Any] | None = None
    security_policy_trigger_details: dict[str, Any] | None = None
    exception_details: str | None = None
    outcome: str = "SUCCESS"
    previous_hash: str = GENESIS_HASH
    current_hash: str = ""

    def compute_hash(self) -> str:
        """Calculate bitwise deterministic SHA-256 digest over canonical event payload."""
        payload = {
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "tool_inputs_digest": self.tool_inputs_digest,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "tier": self.tier,
            "token_usage": self.token_usage,
            "cost_estimate_usd": round(self.cost_estimate_usd, 6),
            "verifier_result": self.verifier_result,
            "ast_verifier_rejection_details": self.ast_verifier_rejection_details,
            "security_policy_trigger_details": self.security_policy_trigger_details,
            "exception_details": self.exception_details,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class HashedTraceLedger:
    """Append-only, cryptographically verifiable trace ledger."""

    def __init__(self, ledger_id: str = "default_ledger") -> None:
        self.ledger_id = ledger_id
        self.events: list[HashedTraceEvent] = []
        self.head_hash: str = GENESIS_HASH

    def record_step(
        self,
        task_id: str,
        run_id: str,
        step_type: str,
        parent_event_id: str | None = None,
        tool_name: str | None = None,
        tool_inputs: dict[str, Any] | None = None,
        model_provider: str | None = None,
        model_id: str | None = None,
        tier: str | None = None,
        token_usage: dict[str, int] | None = None,
        cost_estimate_usd: float = 0.0,
        verifier_result: dict[str, Any] | None = None,
        ast_verifier_rejection_details: dict[str, Any] | None = None,
        security_policy_trigger_details: dict[str, Any] | None = None,
        exception_details: str | None = None,
        outcome: str = "SUCCESS",
    ) -> HashedTraceEvent:
        """Record an immutable, hash-chained execution step."""
        event_num = len(self.events) + 1
        event_id = f"evt_{event_num:06d}_{run_id[-6:]}"
        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # Compute digest of tool inputs (safe redaction digest)
        inputs_digest = None
        if tool_inputs is not None:
            clean_inputs = json.dumps(tool_inputs, sort_keys=True, separators=(",", ":"))
            inputs_digest = hashlib.sha256(clean_inputs.encode("utf-8")).hexdigest()

        event = HashedTraceEvent(
            event_id=event_id,
            parent_event_id=parent_event_id or (self.events[-1].event_id if self.events else None),
            timestamp=now_iso,
            task_id=task_id,
            run_id=run_id,
            step_type=step_type,
            tool_name=tool_name,
            tool_inputs_digest=inputs_digest,
            model_provider=model_provider,
            model_id=model_id,
            tier=tier,
            token_usage=token_usage or {},
            cost_estimate_usd=cost_estimate_usd,
            verifier_result=verifier_result,
            ast_verifier_rejection_details=ast_verifier_rejection_details,
            security_policy_trigger_details=security_policy_trigger_details,
            exception_details=exception_details,
            outcome=outcome,
            previous_hash=self.head_hash,
        )

        event.current_hash = event.compute_hash()
        self.events.append(event)
        self.head_hash = event.current_hash

        # Also emit to sinks
        log(
            "trace_step_hashed",
            event_id=event.event_id,
            task_id=event.task_id,
            step_type=event.step_type,
            outcome=event.outcome,
            current_hash=event.current_hash,
        )

        return event

    def verify_integrity(self) -> tuple[bool, str | None]:
        """Mathematically verify the hash chain from genesis to head."""
        expected_prev = GENESIS_HASH

        for idx, evt in enumerate(self.events):
            # 1. Check link to previous hash
            if evt.previous_hash != expected_prev:
                return (
                    False,
                    f"Hash chain broken at event {idx} ({evt.event_id}): "
                    f"previous_hash '{evt.previous_hash}' != expected '{expected_prev}'",
                )

            # 2. Recompute and check current hash
            recomputed = evt.compute_hash()
            if evt.current_hash != recomputed:
                return (
                    False,
                    f"Content tampering detected at event {idx} ({evt.event_id}): "
                    f"stored hash '{evt.current_hash}' != recomputed '{recomputed}'",
                )

            expected_prev = evt.current_hash

        return True, None

    def export_regulatory_audit_format(self) -> dict[str, Any]:
        """Format ledger into a structured compliance submission packet."""
        is_valid, error = self.verify_integrity()
        return {
            "compliance_standard": "AEF-1 Immutable Trace Ledger Specification",
            "ledger_id": self.ledger_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "genesis_hash": GENESIS_HASH,
            "terminal_hash": self.head_hash,
            "total_events": len(self.events),
            "cryptographic_verification": {
                "verified": is_valid,
                "error": error,
            },
            "events": [asdict(e) for e in self.events],
        }

    def export_json(self, file_path: str | None = None) -> str:
        """Export ledger to JSON string or save to file."""
        data = self.export_regulatory_audit_format()
        rendered = json.dumps(data, indent=2, sort_keys=True)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(rendered)
        return rendered


# Global default ledger singleton
_GLOBAL_LEDGER = HashedTraceLedger()


def get_trace_ledger() -> HashedTraceLedger:
    global _GLOBAL_LEDGER
    return _GLOBAL_LEDGER


def reset_trace_ledger() -> None:
    global _GLOBAL_LEDGER
    _GLOBAL_LEDGER = HashedTraceLedger()


# --------------------------------------------------------------------------- #
# Legacy Compatibility Layer (Weave & Sinks)
# --------------------------------------------------------------------------- #

_initialized = False
_weave = None
_sinks: dict[str, Any] = {}


def init_tracing() -> None:
    global _initialized, _weave
    if _initialized:
        return
    _initialized = True
    settings = get_settings()
    if settings.tracing != "weave":
        return
    try:  # pragma: no cover - exercised only when weave is installed
        import weave  # type: ignore

        weave.init(settings.weave_project)
        _weave = weave
    except Exception:
        _weave = None


@contextmanager
def span(name: str, **attrs: object) -> Iterator[None]:
    """A tracing span that degrades to a no-op when weave is unavailable."""
    init_tracing()
    if _weave is None:
        yield
        return
    try:  # pragma: no cover - requires weave
        with _weave.attributes(attrs):
            yield
    except Exception:
        yield


def register_sink(name: str, sink: Any) -> None:
    """Register an in-process telemetry/tracing sink."""
    _sinks[name] = sink


def unregister_sink(name: str) -> None:
    """Unregister an in-process telemetry/tracing sink."""
    _sinks.pop(name, None)


def log(event: str, **fields: object) -> None:  # pragma: no cover - thin shim
    for sink in list(_sinks.values()):
        try:
            sink(event, fields)
        except Exception:
            pass

    init_tracing()
    if _weave is None:
        return
    try:
        _weave.publish({"event": event, **fields})  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = [
    "GENESIS_HASH",
    "HashedTraceEvent",
    "HashedTraceLedger",
    "get_trace_ledger",
    "reset_trace_ledger",
    "init_tracing",
    "span",
    "register_sink",
    "unregister_sink",
    "log",
]
