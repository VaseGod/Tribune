"""Chronological State Store: MemoryEventsTimeline.

Append-only, temporally immutable event store capturing:
- timestamp
- transaction_id
- state_change_delta
- executed_tool_invocation
- environment_feedback
- operator_commands

All entries are encapsulated as provenanced tuples: τ = (x, π(x)).
Provides deterministic historical state reconciliation resolving strictly by sequence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .retention import (
    SCHEMA_VERSION as RETENTION_SCHEMA_VERSION,
)
from .retention import (
    EphemeralObservation,
    InMemoryColdStorage,
    RetentionMetrics,
    RetentionPolicy,
    StateDelta,
    ToolLoopDetector,
    ToolLoopSignal,
    estimate_tokens,
)
from .retrieval import ProvenancedTuple, ProvenancePointer

logger = logging.getLogger(__name__)


class TemporalImmutabilityError(RuntimeError):
    """Raised when an attempt is made to mutate or overwrite historical timeline records."""
    pass


@dataclass(frozen=True)
class TimelineEventPayload:
    """State change and environmental feedback payload."""

    transaction_id: str
    state_change_delta: dict[str, Any]
    executed_tool_invocation: dict[str, Any] | str | None = None
    environment_feedback: dict[str, Any] | str | None = None
    operator_commands: dict[str, Any] | str | None = None


@dataclass
class TimelineEvent:
    """Provenanced event record in the append-only chronological timeline."""

    sequence: int
    provenanced_tuple: ProvenancedTuple[TimelineEventPayload]

    @property
    def payload(self) -> TimelineEventPayload:
        return self.provenanced_tuple.data

    @property
    def provenance(self) -> ProvenancePointer:
        return self.provenanced_tuple.provenance

    @property
    def timestamp(self) -> float:
        return self.provenance.timestamp

    @property
    def transaction_id(self) -> str:
        return self.payload.transaction_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id,
            "state_change_delta": self.payload.state_change_delta,
            "executed_tool_invocation": self.payload.executed_tool_invocation,
            "environment_feedback": self.payload.environment_feedback,
            "operator_commands": self.payload.operator_commands,
            "provenance": self.provenance.to_dict(),
        }


class MemoryEventsTimeline:
    """Append-only timeline with Protocol-Aware Retention.

    Backward compatible: :meth:`append` keeps its original signature. New
    optional kwargs split each event into a durable :class:`StateDelta`
    (kept in active context) and an :class:`EphemeralObservation` (flushed
    to cold storage at turn end).
    """

    def __init__(
        self,
        timeline_id: str = "main_timeline",
        retention_policy: RetentionPolicy | None = None,
        cold_storage: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        self.timeline_id = timeline_id
        self._events: list[TimelineEvent] = []
        self._by_tx: dict[str, TimelineEvent] = {}
        self._lock = threading.RLock()
        self.retention_policy = retention_policy or RetentionPolicy()
        if cold_storage is not None:
            self._cold = cold_storage
        else:
            try:
                from .retention import build_cold_storage

                self._cold = build_cold_storage(self.retention_policy)
            except Exception:
                self._cold = InMemoryColdStorage()
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        self._state_deltas: dict[str, StateDelta] = {}
        self._observations: dict[str, EphemeralObservation] = {}
        self._active_event_ids: list[str] = []  # flushed-window membership
        self._loop_detector = ToolLoopDetector(
            repeat_threshold=self.retention_policy.loop_repeat_threshold,
            similarity_threshold=self.retention_policy.loop_similarity_threshold,
            oscillation_window=self.retention_policy.loop_oscillation_window,
        )
        self.metrics = RetentionMetrics()
        self._loop_warnings: list[dict[str, Any]] = []

    # -- legacy append (backward compatible) -------------------------------- #
    def append(
        self,
        transaction_id: str,
        state_change_delta: dict[str, Any],
        executed_tool_invocation: dict[str, Any] | str | None = None,
        environment_feedback: dict[str, Any] | str | None = None,
        operator_commands: dict[str, Any] | str | None = None,
        source_uri: str | None = None,
        commit_hash: str = "HEAD",
        timestamp: float | None = None,
        line_start: int = 1,
        line_end: int = 1,
        # -- retention-aware extensions (all optional) ---------------------- #
        event_id: str | None = None,
        parent_event_id: str | None = None,
        turn_id: str | None = None,
        session_id: str | None = None,
        actor: str = "agent",
        tool_name: str | None = None,
        operation_type: str = "tool_call",
        entity_ids: list[str] | None = None,
        file_changes: list[dict[str, Any]] | None = None,
        exit_code: int | None = None,
        command_status: str = "unknown",
        result_summary: dict[str, Any] | None = None,
        causal_refs: list[str] | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        raw_json: Any | None = None,
    ) -> TimelineEvent:
        """Append a new provenanced event to the chronological timeline."""
        with self._lock:
            evt_timestamp = timestamp if timestamp is not None else time.time()
            seq = len(self._events)

            # Temporal ordering invariant
            if self._events and evt_timestamp < self._events[-1].timestamp:
                # Monotonic clock guarantee for timeline consistency
                evt_timestamp = self._events[-1].timestamp + 0.0001

            uri = source_uri or f"timeline://{self.timeline_id}/tx/{transaction_id}"
            pointer = ProvenancePointer.create(
                source_uri=uri,
                commit_hash=commit_hash,
                line_start=line_start,
                line_end=line_end,
                timestamp=evt_timestamp,
            )

            payload = TimelineEventPayload(
                transaction_id=transaction_id,
                state_change_delta=copy.deepcopy(state_change_delta),
                executed_tool_invocation=copy.deepcopy(executed_tool_invocation),
                environment_feedback=copy.deepcopy(environment_feedback),
                operator_commands=copy.deepcopy(operator_commands),
            )

            p_tuple = ProvenancedTuple(data=payload, provenance=pointer)
            event = TimelineEvent(sequence=seq, provenanced_tuple=p_tuple)

            self._events.append(event)
            self._by_tx[transaction_id] = event

            # Protocol-Aware Retention split (non-breaking additive path).
            if self.retention_policy.enabled:
                self._split_retention(
                    transaction_id=transaction_id,
                    state_change_delta=state_change_delta,
                    executed_tool_invocation=executed_tool_invocation,
                    environment_feedback=environment_feedback,
                    event_id=event_id,
                    parent_event_id=parent_event_id,
                    turn_id=turn_id,
                    session_id=session_id,
                    actor=actor,
                    tool_name=tool_name,
                    operation_type=operation_type,
                    entity_ids=entity_ids,
                    file_changes=file_changes,
                    exit_code=exit_code,
                    command_status=command_status,
                    result_summary=result_summary,
                    causal_refs=causal_refs,
                    stdout=stdout,
                    stderr=stderr,
                    raw_json=raw_json,
                    timestamp=evt_timestamp,
                )
            return event

    # -- retention internals -------------------------------------------------- #
    def _derive_tool_name(self, executed: Any, explicit: str | None) -> str:
        if explicit:
            return explicit
        if isinstance(executed, dict):
            for k in ("tool", "tool_name", "name", "command"):
                v = executed.get(k)
                if isinstance(v, str) and v:
                    return v
        if isinstance(executed, str) and executed:
            return executed.split()[0][:64]
        return "unknown"

    def _split_retention(self, **kw: Any) -> StateDelta:
        executed = kw.get("executed_tool_invocation")
        feedback = kw.get("environment_feedback")
        event_id = kw.get("event_id") or f"evt_{uuid.uuid4().hex[:12]}"
        parent_event_id = kw.get("parent_event_id") or (
            self._active_event_ids[-1] if self._active_event_ids else None
        )
        turn_id = kw.get("turn_id") or self._turn_id
        session_id = kw.get("session_id") or self.session_id
        tool_name = self._derive_tool_name(executed, kw.get("tool_name"))
        stdout = kw.get("stdout")
        stderr = kw.get("stderr")
        raw_json = kw.get("raw_json")
        if stdout is None and stderr is None and raw_json is None:
            # Derive ephemeral payload from environment_feedback (raw tool output).
            if isinstance(feedback, dict):
                stdout = str(feedback.get("stdout", ""))
                stderr = str(feedback.get("stderr", ""))
                raw_json = feedback.get("raw", feedback)
            elif isinstance(feedback, str):
                stdout = feedback
                stderr = ""
                raw_json = {"text": feedback}
            else:
                stdout, stderr, raw_json = "", "", None
        obs = EphemeralObservation.build(
            event_id=event_id, stdout=stdout or "", stderr=stderr or "", raw_json=raw_json
        )
        try:
            cold_uri = self._cold.write(obs)
        except Exception as exc:
            logger.warning("Cold storage write failed: %s", exc)
            cold_uri = obs.cold_uri
        obs = EphemeralObservation(
            event_id=obs.event_id,
            stdout=obs.stdout if self.retention_policy.emergency_debug_retention else "",
            stderr=obs.stderr if self.retention_policy.emergency_debug_retention else "",
            raw_json=obs.raw_json if self.retention_policy.emergency_debug_retention else None,
            payload_size=obs.payload_size,
            created_at=obs.created_at,
            content_hash=obs.content_hash,
            cold_uri=cold_uri,
            serializer=obs.serializer,
        )
        delta = StateDelta(
            event_id=event_id,
            parent_event_id=parent_event_id,
            turn_id=turn_id,
            session_id=session_id,
            timestamp=kw.get("timestamp") or time.time(),
            actor=kw.get("actor") or "agent",
            tool_name=tool_name,
            operation_type=kw.get("operation_type") or "tool_call",
            entity_ids=list(kw.get("entity_ids") or []),
            file_changes=list(kw.get("file_changes") or []),
            exit_code=kw.get("exit_code"),
            command_status=kw.get("command_status") or "unknown",
            result_summary=dict(kw.get("result_summary") or {}) or self._compact_result(
                kw.get("state_change_delta") or {}
            ),
            causal_refs=list(kw.get("causal_refs") or []),
            observation_digest=obs.content_hash,
            schema_version=RETENTION_SCHEMA_VERSION,
        )
        tokens_before = delta.active_tokens() + estimate_tokens(stdout or "") + estimate_tokens(
            json.dumps(raw_json, default=str) if raw_json is not None else ""
        )
        tokens_after = delta.active_tokens()
        self.metrics.tokens_before += tokens_before
        self.metrics.tokens_after += tokens_after
        self.metrics.tokens_flushed += max(0, tokens_before - tokens_after)
        self.metrics.events_split += 1
        self.metrics.cold_writes += 1
        self._state_deltas[event_id] = delta
        self._observations[event_id] = obs
        self._active_event_ids.append(event_id)
        self._enforce_active_window()
        # Tool-loop detection on every append.
        signals = self._loop_detector.observe(
            tool_name, executed if isinstance(executed, dict) else {"raw": executed},
            kw.get("state_change_delta"),
        )
        for sig in signals:
            self._record_loop_warning(sig, event_id, turn_id)
        return delta

    @staticmethod
    def _compact_result(delta: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for k, v in list(delta.items())[:12]:
            s = json.dumps(v, default=str)
            compact[k] = v if len(s) <= 300 else (s[:300] + "…[truncated]")
        return compact

    def _enforce_active_window(self) -> None:
        policy = self.retention_policy
        while len(self._active_event_ids) > policy.max_active_events:
            self._active_event_ids.pop(0)
        # Token-budget enforcement: drop oldest actives (deltas stay queryable).
        while (
            self.active_context_tokens() > policy.max_active_context_tokens
            and len(self._active_event_ids) > 1
        ):
            self._active_event_ids.pop(0)

    def _record_loop_warning(self, sig: ToolLoopSignal, event_id: str, turn_id: str) -> None:
        blocked = bool(self.retention_policy.block_repeated_calls)
        sig.blocked = blocked
        self.metrics.loop_detections += 1
        if blocked:
            self.metrics.blocked_calls += 1
        warning = {
            "kind": sig.kind,
            "tool_name": sig.tool_name,
            "count": sig.count,
            "message": sig.message,
            "event_id": event_id,
            "turn_id": turn_id,
            "blocked": blocked,
            "timestamp": time.time(),
        }
        self._loop_warnings.append(warning)
        logger.warning("[RETENTION-LOOP] %s", sig.message)

    # -- retention public API --------------------------------------------------- #
    def begin_turn(self, turn_id: str | None = None) -> str:
        with self._lock:
            self._turn_id = turn_id or f"turn_{uuid.uuid4().hex[:8]}"
            return self._turn_id

    def end_turn(self, turn_id: str | None = None) -> dict[str, Any]:
        """Flush ephemeral observations from the active window at turn end.

        StateDeltas + provenance refs remain; raw payloads live only in cold storage
        (unless emergency debug retention is enabled).
        """
        with self._lock:
            flushed = 0
            freed_tokens = 0
            for eid in list(self._active_event_ids):
                obs = self._observations.get(eid)
                if obs is None:
                    continue
                freed_tokens += obs.active_tokens()
                flushed += 1
                if not self.retention_policy.emergency_debug_retention:
                    self._observations[eid] = EphemeralObservation(
                        event_id=obs.event_id,
                        stdout="",
                        stderr="",
                        raw_json=None,
                        payload_size=obs.payload_size,
                        created_at=obs.created_at,
                        content_hash=obs.content_hash,
                        cold_uri=obs.cold_uri or f"cold://observations/{obs.content_hash}",
                        serializer=obs.serializer,
                    )
            self.metrics.tokens_flushed += freed_tokens
            self.metrics.active_context_tokens = self.active_context_tokens()
            return {
                "turn_id": turn_id or self._turn_id,
                "flushed_observations": flushed,
                "freed_token_estimate": freed_tokens,
                "active_context_tokens": self.metrics.active_context_tokens,
            }

    def get_state_delta(self, event_id: str) -> StateDelta | None:
        with self._lock:
            return self._state_deltas.get(event_id)

    def get_observation(self, event_id: str, privileged: bool = False) -> EphemeralObservation | None:
        """Raw observations require explicit provenance lookup (privileged flag)."""
        with self._lock:
            if not privileged and not self.retention_policy.emergency_debug_retention:
                return None
            return self._observations.get(event_id)

    def fetch_cold_observation(self, content_hash: str) -> EphemeralObservation | None:
        try:
            return self._cold.read(content_hash)
        except Exception as exc:
            logger.warning("Cold storage read failed: %s", exc)
            return None

    def active_context(
        self, include_provenance_refs: bool = True
    ) -> list[dict[str, Any]]:
        """Dense factual active context: StateDeltas + provenance refs only."""
        with self._lock:
            ctx: list[dict[str, Any]] = []
            for eid in self._active_event_ids:
                delta = self._state_deltas.get(eid)
                if delta is None:
                    continue
                entry = delta.to_dict()
                if include_provenance_refs:
                    obs = self._observations.get(eid)
                    entry["provenance_ref"] = {
                        "observation_digest": delta.observation_digest,
                        "cold_uri": (obs.cold_uri if obs else f"cold://observations/{delta.observation_digest}"),
                    }
                ctx.append(entry)
            return ctx

    def active_context_tokens(self) -> int:
        total = 0
        for eid in self._active_event_ids:
            delta = self._state_deltas.get(eid)
            if delta is not None:
                total += delta.active_tokens()
        if self.retention_policy.emergency_debug_retention:
            for eid in self._active_event_ids:
                obs = self._observations.get(eid)
                if obs is not None:
                    total += obs.active_tokens()
        return total

    def loop_warnings(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._loop_warnings)

    def check_tool_call(self, tool_name: str, args: Any) -> ToolLoopSignal | None:
        """Pre-dispatch loop check; raises-free, optionally blocks per policy."""
        with self._lock:
            recent = [h for h in self._loop_detector._history if h[1] == tool_name]
            sig_hex = hashlib.sha256(
                json.dumps({"t": tool_name, "a": args}, sort_keys=True, default=str).encode()
            ).hexdigest()
            repeats = sum(1 for h in recent[-self.retention_policy.loop_repeat_threshold :] if h[0] == sig_hex)
            if repeats >= self.retention_policy.loop_repeat_threshold:
                sig = ToolLoopSignal("exact_repeat", tool_name, repeats, f"Pre-dispatch block: '{tool_name}' loop.")
                self._record_loop_warning(sig, "pre-dispatch", self._turn_id)
                if self.retention_policy.block_repeated_calls:
                    from .retention import ToolLoopSignal as _S  # noqa: F401 (re-export clarity)

                    return sig
                return sig
            return None

    def retention_metrics(self) -> dict[str, Any]:
        with self._lock:
            self.metrics.active_context_tokens = self.active_context_tokens()
            d = self.metrics.to_dict()
            d["active_events"] = len(self._active_event_ids)
            d["loop_warnings"] = len(self._loop_warnings)
            return d

    def get_by_sequence(self, seq: int) -> TimelineEvent | None:
        """Fetch event by sequence number."""
        with self._lock:
            if 0 <= seq < len(self._events):
                return self._events[seq]
            return None

    def get_by_transaction_id(self, tx_id: str) -> TimelineEvent | None:
        """Fetch event by transaction id."""
        with self._lock:
            return self._by_tx.get(tx_id)

    def query_sequence_range(self, from_seq: int, to_seq: int) -> list[TimelineEvent]:
        """Fetch range of events [from_seq, to_seq] inclusive."""
        with self._lock:
            return [
                self._events[i]
                for i in range(max(0, from_seq), min(len(self._events), to_seq + 1))
            ]

    def reconcile_history(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[TimelineEvent]:
        """Query chronological events strictly ordered by sequence within time window."""
        with self._lock:
            results: list[TimelineEvent] = []
            for evt in self._events:
                if start_time is not None and evt.timestamp < start_time:
                    continue
                if end_time is not None and evt.timestamp > end_time:
                    continue
                results.append(evt)
            return results

    def get_state_at(self, timestamp: float) -> dict[str, Any]:
        """Reconstruct the cumulative state strictly up to the given timestamp.

        Rolls forward state_change_deltas in chronological order.
        """
        with self._lock:
            state: dict[str, Any] = {}
            for evt in self._events:
                if evt.timestamp > timestamp:
                    break
                # Apply delta
                for k, v in evt.payload.state_change_delta.items():
                    if isinstance(v, dict) and isinstance(state.get(k), dict):
                        state[k].update(v)
                    else:
                        state[k] = copy.deepcopy(v)
            return state

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def verify_temporal_immutability(self) -> bool:
        """Verify that sequence numbers are monotonic and timestamps strictly non-decreasing."""
        with self._lock:
            prev_t = -1.0
            for i, evt in enumerate(self._events):
                if evt.sequence != i:
                    raise TemporalImmutabilityError(f"Sequence break at index {i}: got {evt.sequence}")
                if evt.timestamp < prev_t:
                    raise TemporalImmutabilityError(f"Timestamp inversion at sequence {i}")
                prev_t = evt.timestamp
            return True


__all__ = [
    "TemporalImmutabilityError",
    "TimelineEventPayload",
    "TimelineEvent",
    "MemoryEventsTimeline",
    "StateDelta",
    "EphemeralObservation",
    "RetentionPolicy",
    "RetentionMetrics",
    "ToolLoopSignal",
]
