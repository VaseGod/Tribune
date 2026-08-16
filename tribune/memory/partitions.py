"""Typed, isolated memory partitions for cases and delegated subagents.

A :class:`MemoryPartition` is the interface agents use to interact with memory.
- :class:`CasePartition`: Bound to a ``case_id`` for primary case data.
- :class:`SubagentMemoryPartition`: Bound to ``(case_id, subagent_id, program)``, providing
  an isolated worktree partition that guarantees concurrent evaluations across multiple
  benefit programs (e.g., parallel Medicaid and SNAP determinations) run without memory
  or state cross-contamination.

Subagent partitions support branching via :meth:`SubagentMemoryPartition.fork` and
safe reconciliation via :meth:`SubagentMemoryPartition.merge_into`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .store import AccessDenied, InMemoryStore, MemoryRecord, MemoryStore

if TYPE_CHECKING:
    from ..types import ProgramId


@runtime_checkable
class MemoryPartition(Protocol):
    """Abstract interface for all memory partitions."""

    @property
    def partition_id(self) -> str: ...

    def write(
        self,
        kind: str,
        key: str,
        record_type: str,
        payload: dict,
        ttl_s: float | None = None,
    ) -> None: ...

    def read(self, kind: str, key: str) -> MemoryRecord | None: ...

    def read_all(self, kind: str) -> list[MemoryRecord]: ...

    def delete(self, kind: str, key: str) -> None: ...

    def purge_expired(self) -> int: ...

    def snapshot(self) -> dict[str, list[dict]]: ...


class CasePartition:
    """Case-level memory partition bound to a single case_id."""

    def __init__(self, case_id: str, store: MemoryStore) -> None:
        self.case_id = case_id
        self._store = store

    @property
    def partition_id(self) -> str:
        return self.case_id

    def write(
        self,
        kind: str,
        key: str,
        record_type: str,
        payload: dict,
        ttl_s: float | None = None,
    ) -> None:
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

    def snapshot(self) -> dict[str, list[dict]]:
        """Export a serialized snapshot of all records in this partition."""
        kinds = ["evidence", "assessment", "summary", "materials", "diagnostics", "custom"]
        snap: dict[str, list[dict]] = {}
        for k in kinds:
            recs = self.read_all(k)
            if recs:
                snap[k] = [r.model_dump() for r in recs]
        return snap

    def restore(self, snapshot_data: dict[str, list[dict]]) -> None:
        """Restore records from a serialized snapshot."""
        for _kind, rec_list in snapshot_data.items():
            for raw in rec_list:
                rec = MemoryRecord.model_validate(raw)
                self._store.put(rec, requester=self.case_id)


class SubagentMemoryPartition:
    """Isolated worktree partition dedicated to a single subagent / benefit program evaluation.

    Guarantees that subagents running concurrently on distinct tasks/programs (e.g. Medicaid vs. SNAP)
    operate in isolated namespaces without memory or state cross-contamination.
    """

    def __init__(
        self,
        case_id: str,
        subagent_id: str,
        store: MemoryStore,
        program: ProgramId | None = None,
    ) -> None:
        self.case_id = case_id
        self.subagent_id = subagent_id
        self.program = program
        self._store = store
        self._scoped_id = f"{case_id}::subagent::{subagent_id}"

    @property
    def partition_id(self) -> str:
        return self._scoped_id

    def write(
        self,
        kind: str,
        key: str,
        record_type: str,
        payload: dict,
        ttl_s: float | None = None,
    ) -> None:
        rec = MemoryRecord(
            case_id=self._scoped_id,
            kind=kind,
            key=key,
            record_type=record_type,
            payload=payload,
            ttl_s=ttl_s,
        )
        self._store.put(rec, requester=self._scoped_id)

    def read(self, kind: str, key: str) -> MemoryRecord | None:
        return self._store.get(self._scoped_id, kind, key, requester=self._scoped_id)

    def read_all(self, kind: str) -> list[MemoryRecord]:
        return self._store.list(self._scoped_id, kind, requester=self._scoped_id)

    def delete(self, kind: str, key: str) -> None:
        self._store.delete(self._scoped_id, kind, key, requester=self._scoped_id)

    def purge_expired(self) -> int:
        return self._store.purge_expired(self._scoped_id, requester=self._scoped_id)

    def try_read_other_subagent(
        self, other_subagent_partition: SubagentMemoryPartition, kind: str, key: str
    ) -> MemoryRecord | None:
        """Attempt to read from another subagent partition. Raises AccessDenied."""
        return self._store.get(
            other_subagent_partition._scoped_id, kind, key, requester=self._scoped_id
        )

    def fork(self, new_subagent_id: str, program: ProgramId | None = None) -> SubagentMemoryPartition:
        """Fork this subagent partition into a new isolated subagent worktree with copied state."""
        child = SubagentMemoryPartition(
            case_id=self.case_id,
            subagent_id=new_subagent_id,
            store=self._store,
            program=program or self.program,
        )
        # Copy all records into the child partition
        for kind in ["evidence", "assessment", "summary", "materials", "diagnostics", "custom"]:
            for rec in self.read_all(kind):
                child.write(
                    kind=rec.kind,
                    key=rec.key,
                    record_type=rec.record_type,
                    payload=dict(rec.payload),
                    ttl_s=rec.ttl_s,
                )
        return child

    def merge_into(
        self,
        target_partition: CasePartition | SubagentMemoryPartition,
        conflict_strategy: str = "latest",
    ) -> int:
        """Safely merge records from this subagent partition into the target partition.

        Returns the count of merged records.
        """
        merged = 0
        kinds = ["evidence", "assessment", "summary", "materials", "diagnostics", "custom"]
        for kind in kinds:
            for rec in self.read_all(kind):
                target_partition.write(
                    kind=rec.kind,
                    key=rec.key,
                    record_type=rec.record_type,
                    payload=dict(rec.payload),
                    ttl_s=rec.ttl_s,
                )
                merged += 1
        return merged

    def snapshot(self) -> dict[str, list[dict]]:
        """Export serialized snapshot."""
        kinds = ["evidence", "assessment", "summary", "materials", "diagnostics", "custom"]
        snap: dict[str, list[dict]] = {}
        for k in kinds:
            recs = self.read_all(k)
            if recs:
                snap[k] = [r.model_dump() for r in recs]
        return snap


class PartitionManager:
    """Hands out case-scoped partitions and isolated subagent worktrees over a shared backing store."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store: MemoryStore = store or InMemoryStore()

    def open(self, case_id: str) -> CasePartition:
        """Open the primary case partition."""
        return CasePartition(case_id, self._store)

    def open_subagent(
        self,
        case_id: str,
        subagent_id: str,
        program: ProgramId | None = None,
        initial_records: list[MemoryRecord] | None = None,
    ) -> SubagentMemoryPartition:
        """Open a dedicated subagent memory worktree partition."""
        partition = SubagentMemoryPartition(
            case_id=case_id,
            subagent_id=subagent_id,
            store=self._store,
            program=program,
        )
        if initial_records:
            for rec in initial_records:
                partition.write(
                    kind=rec.kind,
                    key=rec.key,
                    record_type=rec.record_type,
                    payload=dict(rec.payload),
                    ttl_s=rec.ttl_s,
                )
        return partition

    def merge_subagent(
        self,
        subagent_partition: SubagentMemoryPartition,
        target_partition: CasePartition,
    ) -> int:
        """Merge a subagent's verified conclusions back into the primary case partition."""
        return subagent_partition.merge_into(target_partition)


__all__ = [
    "MemoryPartition",
    "CasePartition",
    "SubagentMemoryPartition",
    "PartitionManager",
    "AccessDenied",
]
