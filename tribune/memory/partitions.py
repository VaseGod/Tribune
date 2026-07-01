"""Typed, per-case memory partitions.

A :class:`CasePartition` is the only handle agents use to read or write case
memory. It is bound to a single ``case_id`` and passes that id as both the target
and the requester on every store call, so a partition can *only* ever touch its
own case. Attempting to read another case's data raises
:class:`~tribune.memory.store.AccessDenied`.

Agents are expected to read-first: load whatever is already known for the case
before writing new conclusions, so memory accumulates coherently.
"""

from __future__ import annotations

from .store import AccessDenied, InMemoryStore, MemoryRecord, MemoryStore


class CasePartition:
    def __init__(self, case_id: str, store: MemoryStore) -> None:
        self.case_id = case_id
        self._store = store

    def write(self, kind: str, key: str, record_type: str, payload: dict, ttl_s: float | None = None) -> None:
        rec = MemoryRecord(
            case_id=self.case_id,
            kind=kind,
            key=key,
            record_type=record_type,
            payload=payload,
            ttl_s=ttl_s,
        )
        self._store.put(rec, requester=self.case_id)

    def read(self, kind: str, key: str) -> MemoryRecord | None:
        return self._store.get(self.case_id, kind, key, requester=self.case_id)

    def read_all(self, kind: str) -> list[MemoryRecord]:
        return self._store.list(self.case_id, kind, requester=self.case_id)

    def delete(self, kind: str, key: str) -> None:
        self._store.delete(self.case_id, kind, key, requester=self.case_id)

    def purge_expired(self) -> int:
        return self._store.purge_expired(self.case_id, requester=self.case_id)

    def try_read_other_case(self, other_case_id: str, kind: str, key: str) -> MemoryRecord | None:
        """Explicit cross-case read attempt. Always raises AccessDenied unless the
        target is this partition's own case. Exists so the isolation guarantee can
        be exercised directly in tests."""
        return self._store.get(other_case_id, kind, key, requester=self.case_id)


class PartitionManager:
    """Hands out case-scoped partitions over a shared backing store."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store: MemoryStore = store or InMemoryStore()

    def open(self, case_id: str) -> CasePartition:
        return CasePartition(case_id, self._store)


__all__ = ["CasePartition", "PartitionManager", "AccessDenied"]
