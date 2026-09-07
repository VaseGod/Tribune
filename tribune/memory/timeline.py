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
import threading
import time
from dataclasses import dataclass
from typing import Any

from .retrieval import ProvenancedTuple, ProvenancePointer


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
    """Append-only, temporally immutable memory store with historical state reconciliation."""

    def __init__(self, timeline_id: str = "main_timeline") -> None:
        self.timeline_id = timeline_id
        self._events: list[TimelineEvent] = []
        self._by_tx: dict[str, TimelineEvent] = {}
        self._lock = threading.RLock()

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
            return event

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
]
