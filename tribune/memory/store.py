"""MemoryStore interface + a local in-memory implementation.

Access control is enforced at the store boundary: every read/write names the
``case_id`` it targets and the ``requester`` asking for it, and a requester may
only touch its own case. Cross-case access raises :class:`AccessDenied`. This is
the mechanism that keeps one person's data unreadable from another's case.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class AccessDenied(PermissionError):
    pass


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    kind: str  # "evidence" | "assessment" | "summary" | ...
    key: str
    record_type: str
    payload: dict
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ttl_s: float | None = None

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.updated_at) > self.ttl_s


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: MemoryRecord, *, requester: str) -> None: ...

    def get(self, case_id: str, kind: str, key: str, *, requester: str) -> MemoryRecord | None: ...

    def list(self, case_id: str, kind: str, *, requester: str) -> list[MemoryRecord]: ...

    def delete(self, case_id: str, kind: str, key: str, *, requester: str) -> None: ...

    def purge_expired(self, case_id: str, *, requester: str) -> int: ...


def _check(case_id: str, requester: str) -> None:
    if requester != case_id:
        raise AccessDenied(
            f"requester '{requester}' may not access memory partition for case '{case_id}'"
        )


class InMemoryStore:
    """Process-local store. Swap for a SQLite/db-backed store in deployment."""

    def __init__(self) -> None:
        # case_id -> kind -> key -> record
        self._data: dict[str, dict[str, dict[str, MemoryRecord]]] = {}

    def put(self, record: MemoryRecord, *, requester: str) -> None:
        _check(record.case_id, requester)
        self._data.setdefault(record.case_id, {}).setdefault(record.kind, {})[record.key] = record

    def get(self, case_id: str, kind: str, key: str, *, requester: str) -> MemoryRecord | None:
        _check(case_id, requester)
        return self._data.get(case_id, {}).get(kind, {}).get(key)

    def list(self, case_id: str, kind: str, *, requester: str) -> list[MemoryRecord]:
        _check(case_id, requester)
        return list(self._data.get(case_id, {}).get(kind, {}).values())

    def delete(self, case_id: str, kind: str, key: str, *, requester: str) -> None:
        _check(case_id, requester)
        self._data.get(case_id, {}).get(kind, {}).pop(key, None)

    def purge_expired(self, case_id: str, *, requester: str) -> int:
        _check(case_id, requester)
        removed = 0
        now = time.time()
        for bucket in self._data.get(case_id, {}).values():
            for key in list(bucket.keys()):
                if bucket[key].is_expired(now):
                    del bucket[key]
                    removed += 1
        return removed
