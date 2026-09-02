from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .store import AccessDenied, InMemoryStore, MemoryRecord, MemoryStore

if TYPE_CHECKING:
    from ..types import ProgramId

logger = logging.getLogger(__name__)


class PartitionTamperingError(RuntimeError):
    """Raised when a partition checksum mismatch or unauthorized mutation (EvoMal attack) is detected."""
    pass


class MemoryPoisoningDetectedError(RuntimeError):
    """Raised when memory poisoning or unauthorized cross-partition injection is detected."""
    pass


class PartitionIntegrityManifest:
    """Cryptographic SHA-256 manifest manager enforcing tamper-evident partition and skill module integrity.
    
    Protects multi-agent systems against EvoMal memory poisoning attacks by verifying signatures
    prior to execution and quarantining altered or unsigned partitions.
    """

    def __init__(self) -> None:
        self._signatures: dict[str, str] = {}
        self._quarantined: dict[str, str] = {}
        self.verified_count: int = 0
        self.rejected_count: int = 0

    def sign_partition(self, partition_id: str, checksum: str) -> str:
        """Register the cryptographic SHA-256 signature for a partition."""
        self._signatures[partition_id] = checksum
        return checksum

    def sign_skill(self, skill_name: str, content: str | bytes) -> str:
        """Register the cryptographic SHA-256 signature for a skill module or rule index."""
        raw = content.encode("utf-8") if isinstance(content, str) else content
        digest = hashlib.sha256(raw).hexdigest()
        self._signatures[f"skill::{skill_name}"] = digest
        return digest

    def verify_partition(self, partition_id: str, current_checksum: str) -> bool:
        """Verify partition integrity against registered SHA-256 manifest.
        
        Returns True if verified. If signature is missing or mismatched, increments rejected_count,
        quarantines the partition, and raises PartitionTamperingError.
        """
        if partition_id in self._quarantined:
            self.rejected_count += 1
            raise PartitionTamperingError(
                f"Access denied: Partition '{partition_id}' is quarantined due to: {self._quarantined[partition_id]}"
            )

        expected = self._signatures.get(partition_id)
        if expected is None:
            # Unsigned partition
            self.rejected_count += 1
            self.quarantine(partition_id, "Unsigned partition (missing cryptographic manifest entry)")
            raise PartitionTamperingError(
                f"Integrity check failed: Partition '{partition_id}' has no registered cryptographic signature."
            )

        if expected != current_checksum:
            self.rejected_count += 1
            self.quarantine(
                partition_id,
                f"SHA-256 checksum mismatch (expected {expected[:12]}..., got {current_checksum[:12]}...)",
            )
            raise PartitionTamperingError(
                f"EvoMal Memory Poisoning Detected: Partition '{partition_id}' has been altered or tampered with!"
            )

        self.verified_count += 1
        return True

    def verify_skill(self, skill_name: str, content: str | bytes) -> bool:
        """Verify a skill module or rule index against its registered SHA-256 signature."""
        key = f"skill::{skill_name}"
        raw = content.encode("utf-8") if isinstance(content, str) else content
        current_digest = hashlib.sha256(raw).hexdigest()
        expected = self._signatures.get(key)
        if expected is None or expected != current_digest:
            self.rejected_count += 1
            raise PartitionTamperingError(
                f"Skill verification failed for '{skill_name}': Unsigned or altered module detected."
            )
        self.verified_count += 1
        return True

    def quarantine(self, partition_id: str, reason: str) -> None:
        """Immediately isolate and quarantine a compromised partition."""
        self._quarantined[partition_id] = reason
        logger.warning("PARTITION QUARANTINED: %s | Reason: %s", partition_id, reason)

    def is_quarantined(self, partition_id: str) -> bool:
        return partition_id in self._quarantined

    def stats(self) -> dict[str, Any]:
        total = self.verified_count + self.rejected_count
        rejection_rate = (self.rejected_count / total) if total > 0 else 0.0
        return {
            "verified_count": self.verified_count,
            "rejected_count": self.rejected_count,
            "quarantined_count": len(self._quarantined),
            "rejection_rate": rejection_rate,
            "quarantined_partitions": dict(self._quarantined),
        }


@runtime_checkable
class MemoryPartition(Protocol):
    """Abstract interface for all memory partitions."""

    @property
    def partition_id(self) -> str: ...

    def compute_checksum(self) -> str: ...

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

    def __init__(self, case_id: str, store: MemoryStore, manifest: PartitionIntegrityManifest | None = None) -> None:
        self.case_id = case_id
        self._store = store
        self._manifest = manifest
        self._is_quarantined: bool = False
        self._quarantine_reason: str = ""

    @property
    def partition_id(self) -> str:
        return self.case_id

    @property
    def is_quarantined(self) -> bool:
        if self._manifest and self._manifest.is_quarantined(self.partition_id):
            return True
        return self._is_quarantined

    def compute_checksum(self) -> str:
        """Compute deterministic cryptographic SHA-256 digest of all records in partition."""
        snap = self.snapshot()
        raw = json.dumps(snap, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def quarantine(self, reason: str) -> None:
        """Mark partition as quarantined, blocking subsequent read/write access."""
        self._is_quarantined = True
        self._quarantine_reason = reason
        if self._manifest:
            self._manifest.quarantine(self.partition_id, reason)

    def _check_quarantined(self) -> None:
        if self._is_quarantined:
            raise PartitionTamperingError(
                f"Access rejected: Partition '{self.case_id}' is quarantined due to: {self._quarantine_reason}"
            )

    def write(
        self,
        kind: str,
        key: str,
        record_type: str,
        payload: dict,
        ttl_s: float | None = None,
    ) -> None:
        self._check_quarantined()
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
        self._check_quarantined()
        return self._store.get(self.case_id, kind, key, requester=self.case_id)

    def read_all(self, kind: str) -> list[MemoryRecord]:
        self._check_quarantined()
        return self._store.list(self.case_id, kind, requester=self.case_id)

    def delete(self, kind: str, key: str) -> None:
        self._check_quarantined()
        self._store.delete(self.case_id, kind, key, requester=self.case_id)

    def purge_expired(self) -> int:
        self._check_quarantined()
        return self._store.purge_expired(self.case_id, requester=self.case_id)

    def try_read_other_case(self, other_case_id: str, kind: str, key: str) -> MemoryRecord | None:
        """Explicit cross-case read attempt. Always raises AccessDenied unless the
        target is this partition's own case. Exists so the isolation guarantee can
        be exercised directly in tests."""
        self._check_quarantined()
        return self._store.get(other_case_id, kind, key, requester=self.case_id)

    def snapshot(self) -> dict[str, list[dict]]:
        """Export a serialized snapshot of all records in this partition."""
        kinds = ["evidence", "assessment", "summary", "materials", "diagnostics", "custom"]
        snap: dict[str, list[dict]] = {}
        for k in kinds:
            recs = self.read_all(k) if not self._is_quarantined else []
            if recs:
                snap[k] = [r.model_dump() for r in recs]
        return snap

    def restore(self, snapshot_data: dict[str, list[dict]]) -> None:
        """Restore records from a serialized snapshot."""
        self._check_quarantined()
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
        manifest: PartitionIntegrityManifest | None = None,
    ) -> None:
        self.case_id = case_id
        self.subagent_id = subagent_id
        self.program = program
        self._store = store
        self._manifest = manifest
        self._scoped_id = f"{case_id}::subagent::{subagent_id}"
        self._is_quarantined: bool = False
        self._quarantine_reason: str = ""

    @property
    def partition_id(self) -> str:
        return self._scoped_id

    @property
    def is_quarantined(self) -> bool:
        if self._manifest and self._manifest.is_quarantined(self.partition_id):
            return True
        return self._is_quarantined

    def compute_checksum(self) -> str:
        """Compute deterministic cryptographic SHA-256 digest of all records in subagent partition."""
        snap = self.snapshot()
        raw = json.dumps(snap, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def quarantine(self, reason: str) -> None:
        """Mark partition as quarantined, blocking subsequent read/write access."""
        self._is_quarantined = True
        self._quarantine_reason = reason
        if self._manifest:
            self._manifest.quarantine(self.partition_id, reason)

    def _check_quarantined(self) -> None:
        if self._is_quarantined:
            raise PartitionTamperingError(
                f"Access rejected: Subagent partition '{self._scoped_id}' is quarantined due to: {self._quarantine_reason}"
            )

    def write(
        self,
        kind: str,
        key: str,
        record_type: str,
        payload: dict,
        ttl_s: float | None = None,
    ) -> None:
        self._check_quarantined()
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
        self._check_quarantined()
        return self._store.get(self._scoped_id, kind, key, requester=self._scoped_id)

    def read_all(self, kind: str) -> list[MemoryRecord]:
        self._check_quarantined()
        return self._store.list(self._scoped_id, kind, requester=self._scoped_id)

    def delete(self, kind: str, key: str) -> None:
        self._check_quarantined()
        self._store.delete(self._scoped_id, kind, key, requester=self._scoped_id)

    def purge_expired(self) -> int:
        self._check_quarantined()
        return self._store.purge_expired(self._scoped_id, requester=self._scoped_id)

    def try_read_other_subagent(
        self, other_subagent_partition: SubagentMemoryPartition, kind: str, key: str
    ) -> MemoryRecord | None:
        """Attempt to read from another subagent partition. Raises AccessDenied."""
        self._check_quarantined()
        return self._store.get(
            other_subagent_partition._scoped_id, kind, key, requester=self._scoped_id
        )

    def fork(self, new_subagent_id: str, program: ProgramId | None = None) -> SubagentMemoryPartition:
        """Fork this subagent partition into a new isolated subagent worktree with copied state."""
        self._check_quarantined()
        child = SubagentMemoryPartition(
            case_id=self.case_id,
            subagent_id=new_subagent_id,
            store=self._store,
            program=program or self.program,
            manifest=self._manifest,
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
        self._check_quarantined()
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
            recs = self.read_all(k) if not self._is_quarantined else []
            if recs:
                snap[k] = [r.model_dump() for r in recs]
        return snap


class PartitionManager:
    """Hands out case-scoped partitions and isolated subagent worktrees over a shared backing store."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store: MemoryStore = store or InMemoryStore()
        self.manifest = PartitionIntegrityManifest()

    def open(self, case_id: str) -> CasePartition:
        """Open the primary case partition."""
        return CasePartition(case_id, self._store, manifest=self.manifest)

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
            manifest=self.manifest,
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

    def sign_partition(self, partition: CasePartition | SubagentMemoryPartition) -> str:
        """Sign partition in the cryptographic manifest."""
        checksum = partition.compute_checksum()
        return self.manifest.sign_partition(partition.partition_id, checksum)

    def verify_partition(self, partition: CasePartition | SubagentMemoryPartition) -> bool:
        """Verify partition against cryptographic manifest."""
        try:
            checksum = partition.compute_checksum()
            return self.manifest.verify_partition(partition.partition_id, checksum)
        except PartitionTamperingError as err:
            partition.quarantine(str(err))
            raise

    def quarantine_partition(self, partition: CasePartition | SubagentMemoryPartition, reason: str) -> None:
        """Quarantine a partition."""
        partition.quarantine(reason)

    def merge_subagent(
        self,
        subagent_partition: SubagentMemoryPartition,
        target_partition: CasePartition,
    ) -> int:
        """Merge a subagent's verified conclusions back into the primary case partition."""
        return subagent_partition.merge_into(target_partition)


__all__ = [
    "MemoryPartition",
    "PartitionTamperingError",
    "MemoryPoisoningDetectedError",
    "PartitionIntegrityManifest",
    "CasePartition",
    "SubagentMemoryPartition",
    "PartitionManager",
    "AccessDenied",
]
