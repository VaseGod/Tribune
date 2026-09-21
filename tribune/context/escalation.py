"""Escalation Queue & Handler for Low-Confidence Context Graph Decisions.

Ensures that relationship candidates with confidence below the calibrated threshold
(default 0.85) are never written directly into the graph and are instead routed to
a structured escalation queue for review or offline verification.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any

from .entities import EscalationRecord

logger = logging.getLogger(__name__)


class EscalationQueue:
    """Thread-safe queue holding unverified or low-confidence edge decisions."""

    def __init__(self, queue_name: str = "context_escalation") -> None:
        self.queue_name = queue_name
        self._queue: collections.deque[EscalationRecord] = collections.deque()
        self._records_by_id: dict[str, EscalationRecord] = {}
        self._lock = threading.RLock()
        self.escalation_count = 0
        self.resolved_count = 0

    def enqueue(self, record: EscalationRecord) -> None:
        """Enqueue an escalation record."""
        with self._lock:
            self._queue.append(record)
            self._records_by_id[record.record_id] = record
            self.escalation_count += 1
            logger.warning(
                f"[EscalationQueue:{self.queue_name}] Enqueued record {record.record_id} "
                f"({record.candidate.source_id} -> {record.candidate.target_id}): {record.reason} "
                f"(confidence={record.decision.confidence:.4f})"
            )

    def peek(self) -> EscalationRecord | None:
        """View oldest pending escalation record without removal."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def dequeue(self) -> EscalationRecord | None:
        """Pop the oldest pending escalation record."""
        with self._lock:
            if not self._queue:
                return None
            record = self._queue.popleft()
            return record

    def resolve(
        self,
        record_id: str,
        resolution: str = "resolved",
        notes: str = "",
    ) -> EscalationRecord | None:
        """Mark an escalation record as resolved or dismissed."""
        with self._lock:
            record = self._records_by_id.get(record_id)
            if not record:
                return None
            record.status = resolution
            record.resolution_notes = notes
            self.resolved_count += 1
            # Remove from active queue if present
            try:
                self._queue.remove(record)
            except ValueError:
                pass
            return record

    def get_record(self, record_id: str) -> EscalationRecord | None:
        with self._lock:
            return self._records_by_id.get(record_id)

    def list_pending(self) -> list[EscalationRecord]:
        with self._lock:
            return [r for r in self._queue if r.status == "pending"]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_name": self.queue_name,
                "pending_count": len(self._queue),
                "total_escalated": self.escalation_count,
                "total_resolved": self.resolved_count,
            }

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._records_by_id.clear()
            self.escalation_count = 0
            self.resolved_count = 0


__all__ = [
    "EscalationQueue",
]
