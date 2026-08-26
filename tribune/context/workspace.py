"""Central Shared Workspace Context, Delta Patch Architecture, & Automated Disk Offloading.

Provides:
1. Unified, in-memory shared state store for multi-agent workflows with JSON pointer navigation.
2. Optimistic concurrency control, structured JSON delta patches, and broadcast subscriptions.
3. Automated session state offloader (`DiskBackedSessionManager`): spills inactive case
   context, raw OCR payloads, and historical transcripts to compressed local disk storage
   after an idle threshold, retaining lightweight references in RAM with transparent async re-hydration.
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..types import (
    DeltaPatch,
    PatchOperationType,
    TokenReductionMetric,
    WorkspaceSnapshot,
)


class PatchValidationError(ValueError):
    """Raised when a delta patch fails validation or path resolution."""
    pass


class VersionConflictError(RuntimeError):
    """Raised when an expected version constraint fails (optimistic concurrency conflict)."""
    pass


def _split_pointer(path: str) -> list[str]:
    """Split JSON Pointer path into decoded token components."""
    if not path or path == "/":
        return []
    if not path.startswith("/"):
        path = "/" + path
    parts = path.split("/")[1:]
    return [p.replace("~1", "/").replace("~0", "~") for p in parts]


class WorkspaceState:
    """Core state container managing structured document facts, criteria, and outcomes."""

    def __init__(
        self,
        case_id: str = "",
        jurisdiction: str = "EX",
        initial_data: dict[str, Any] | None = None,
    ) -> None:
        self.case_id = case_id
        self.jurisdiction = jurisdiction
        self.version = 0
        self._data: dict[str, Any] = {
            "case_id": case_id,
            "jurisdiction": jurisdiction,
            "documents": [],
            "evidence": [],
            "criteria_outcomes": {},
            "assessments": {},
            "verification_verdicts": {},
            "materials": {},
            "agent_metadata": {},
            "shared_facts": {},
            "visual_layouts": {},
        }

        if initial_data:
            self._data.update(initial_data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def get_value_at_path(self, path: str) -> Any:
        tokens = _split_pointer(path)
        curr: Any = self._data
        for token in tokens:
            if isinstance(curr, dict):
                if token not in curr:
                    return None
                curr = curr[token]
            elif isinstance(curr, list):
                try:
                    idx = int(token)
                    if 0 <= idx < len(curr):
                        curr = curr[idx]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                return None
        return curr

    def apply_patch_operation(self, patch: DeltaPatch) -> None:
        tokens = _split_pointer(patch.path)
        if not tokens:
            if patch.operation in (PatchOperationType.ADD, PatchOperationType.REPLACE):
                if isinstance(patch.value, dict):
                    self._data = copy.deepcopy(patch.value)
                    return
            raise PatchValidationError("Cannot modify root with non-dict replace")

        # Navigate to target parent
        curr: Any = self._data
        for token in tokens[:-1]:
            if isinstance(curr, dict):
                if token not in curr:
                    curr[token] = {}
                curr = curr[token]
            elif isinstance(curr, list):
                try:
                    idx = int(token)
                    curr = curr[idx]
                except (ValueError, IndexError) as exc:
                    raise PatchValidationError(f"Invalid list index in path '{patch.path}'") from exc
            else:
                raise PatchValidationError(f"Path segment '{token}' is not a container")

        last = tokens[-1]
        val = copy.deepcopy(patch.value)

        if patch.operation == PatchOperationType.ADD:
            if isinstance(curr, dict):
                curr[last] = val
            elif isinstance(curr, list):
                if last == "-":
                    curr.append(val)
                else:
                    try:
                        idx = int(last)
                        curr.insert(idx, val)
                    except ValueError as exc:
                        raise PatchValidationError(f"Invalid list insert index '{last}'") from exc
            else:
                raise PatchValidationError(f"Cannot add to non-container at path '{patch.path}'")

        elif patch.operation == PatchOperationType.REPLACE:
            if isinstance(curr, dict):
                curr[last] = val
            elif isinstance(curr, list):
                try:
                    idx = int(last)
                    curr[idx] = val
                except (ValueError, IndexError) as exc:
                    raise PatchValidationError(f"Invalid list replace index '{last}'") from exc
            else:
                raise PatchValidationError(f"Cannot replace on non-container at path '{patch.path}'")

        elif patch.operation == PatchOperationType.REMOVE:
            if isinstance(curr, dict):
                curr.pop(last, None)
            elif isinstance(curr, list):
                try:
                    idx = int(last)
                    if 0 <= idx < len(curr):
                        curr.pop(idx)
                except ValueError as exc:
                    raise PatchValidationError(f"Invalid list remove index '{last}'") from exc

        elif patch.operation == PatchOperationType.APPEND_UNIQUE:
            if isinstance(curr, dict):
                if last not in curr or not isinstance(curr[last], list):
                    curr[last] = []
                target_list = curr[last]
            elif isinstance(curr, list):
                target_list = curr
            else:
                raise PatchValidationError(f"Target at '{patch.path}' is not a list for append_unique")

            if isinstance(val, list):
                for item in val:
                    if item not in target_list:
                        target_list.append(item)
            else:
                if val not in target_list:
                    target_list.append(val)

        elif patch.operation == PatchOperationType.MERGE_DICT:
            if not isinstance(val, dict):
                raise PatchValidationError(f"merge_dict operation requires dict value, got {type(val)}")
            if isinstance(curr, dict):
                if last not in curr or not isinstance(curr[last], dict):
                    curr[last] = {}
                curr[last].update(val)
            else:
                raise PatchValidationError(f"Cannot merge dict into non-dict parent at '{patch.path}'")

        elif patch.operation == PatchOperationType.TEST:
            existing = curr.get(last) if isinstance(curr, dict) else (curr[int(last)] if isinstance(curr, list) and int(last) < len(curr) else None)
            if existing != val:
                raise PatchValidationError(f"Test operation failed at '{patch.path}': expected {val}, got {existing}")


class WorkspaceContext:
    """Central Shared Workspace Context coordinating multi-agent state reads and delta patch writes.

    Features:
    - Central shared workspace state
    - JSON Delta patch application with optimistic concurrency
    - Subscribed broadcast channels
    - Scoped state slicing to prevent conversational history bloat
    - Deterministic replay from patch history
    - Token generation accounting & reduction measurement
    - Last access timestamping for automatic disk offloading
    """

    def __init__(
        self,
        case_id: str = "",
        jurisdiction: str = "EX",
        storage_path: str | None = None,
    ) -> None:
        self.case_id = case_id
        self.jurisdiction = jurisdiction
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self._state = WorkspaceState(case_id=case_id, jurisdiction=jurisdiction)
        self._patch_history: list[DeltaPatch] = []
        self._subscribers: dict[str, list[Callable[[DeltaPatch], None]]] = {}
        self._patch_bytes_total = 0
        self.last_accessed_at = time.time()

    def _touch(self) -> None:
        self.last_accessed_at = time.time()

    @property
    def version(self) -> int:
        return self._state.version

    @property
    def patch_history(self) -> list[DeltaPatch]:
        with self._lock:
            self._touch()
            return list(self._patch_history)

    def subscribe(self, channel: str, callback: Callable[[DeltaPatch], None]) -> None:
        """Register a subscriber callback for patches affecting a specific channel/path prefix."""
        with self._lock:
            self._touch()
            self._subscribers.setdefault(channel, []).append(callback)

    def _notify_subscribers(self, patch: DeltaPatch) -> None:
        """Dispatch patch notification to broadcast subscribers matching patch path."""
        for channel, callbacks in list(self._subscribers.items()):
            if channel == "*" or patch.path.startswith(channel) or patch.path == channel.rstrip("/"):
                for cb in callbacks:
                    try:
                        cb(patch)
                    except Exception:
                        pass

    def apply_patch(self, patch: DeltaPatch) -> WorkspaceSnapshot:
        """Apply a structured delta patch transactionally with optimistic concurrency control."""
        with self._lock:
            self._touch()
            # Check expected version constraint if specified
            if patch.expected_version is not None and patch.expected_version != self._state.version:
                if patch.conflict_strategy == "error":
                    raise VersionConflictError(
                        f"Conflict in patch from agent '{patch.agent_id}': expected version {patch.expected_version}, "
                        f"but workspace is at version {self._state.version}"
                    )

            # Apply patch to state
            self._state.apply_patch_operation(patch)
            self._state.version += 1
            self._patch_history.append(patch)

            # Account for patch payload size
            patch_json = json.dumps(patch.model_dump(mode="json"), default=str)
            self._patch_bytes_total += len(patch_json.encode("utf-8"))

            snapshot = self.snapshot()
            if self.storage_path:
                self._persist_snapshot()

            self._notify_subscribers(patch)
            return snapshot

    def apply_patches(self, patches: list[DeltaPatch]) -> list[WorkspaceSnapshot]:
        """Atomically apply a sequence of delta patches."""
        with self._lock:
            self._touch()
            snapshots = []
            for p in patches:
                snapshots.append(self.apply_patch(p))
            return snapshots

    def get_slice(self, read_scopes: list[str]) -> dict[str, Any]:
        """Extract a scoped state slice matching the agent's read permissions.

        Prevents quadratic conversational re-serialization by projecting only required keys.
        """
        with self._lock:
            self._touch()
            if not read_scopes or "*" in read_scopes:
                return self._state.to_dict()

            sliced: dict[str, Any] = {
                "case_id": self.case_id,
                "jurisdiction": self.jurisdiction,
                "version": self._state.version,
            }
            for scope in read_scopes:
                val = self._state.get_value_at_path(scope)
                tokens = _split_pointer(scope)
                if not tokens:
                    sliced.update(self._state.to_dict())
                else:
                    curr = sliced
                    for t in tokens[:-1]:
                        if t not in curr:
                            curr[t] = {}
                        curr = curr[t]
                    curr[tokens[-1]] = copy.deepcopy(val)
            return sliced

    def read_path(self, path: str) -> Any:
        """Read a single value from the workspace state via JSON pointer."""
        with self._lock:
            self._touch()
            return copy.deepcopy(self._state.get_value_at_path(path))

    def store_visual_layout(self, layout: Any) -> WorkspaceSnapshot:
        """Store visual document layout tokens and reading-order DAG in workspace state."""
        doc_id = getattr(layout, "doc_id", layout.get("doc_id", "default_doc") if isinstance(layout, dict) else "default_doc")
        layout_dict = layout.to_dict() if hasattr(layout, "to_dict") else dict(layout)
        patch = DeltaPatch(
            run_id=self.case_id,
            agent_id="ocr_ingest",
            operation=PatchOperationType.REPLACE,
            path=f"/visual_layouts/{doc_id}",
            value=layout_dict,
        )
        return self.apply_patch(patch)

    def get_visual_layout(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve visual document layout dictionary for a specific doc_id."""
        with self._lock:
            self._touch()
            return self.read_path(f"/visual_layouts/{doc_id}")


    def snapshot(self) -> WorkspaceSnapshot:
        """Create an immutable snapshot of current workspace state."""
        with self._lock:
            self._touch()
            return WorkspaceSnapshot(
                version=self._state.version,
                case_id=self.case_id,
                jurisdiction=self.jurisdiction,
                state_data=self._state.to_dict(),
                patch_count=len(self._patch_history),
                timestamp=datetime.now(timezone.utc),
            )

    def serialize_full_session(self) -> dict[str, Any]:
        """Serialize full in-memory workspace session including state and patch history for disk offload."""
        with self._lock:
            return {
                "case_id": self.case_id,
                "jurisdiction": self.jurisdiction,
                "version": self._state.version,
                "state_data": self._state.to_dict(),
                "patch_history": [p.model_dump(mode="json") for p in self._patch_history],
                "patch_bytes_total": self._patch_bytes_total,
                "last_accessed_at": self.last_accessed_at,
            }

    @classmethod
    def deserialize_full_session(cls, payload: dict[str, Any]) -> WorkspaceContext:
        """Deserialize full session data into an active WorkspaceContext."""
        ctx = cls(case_id=payload["case_id"], jurisdiction=payload.get("jurisdiction", "EX"))
        ctx._state = WorkspaceState(
            case_id=payload["case_id"],
            jurisdiction=payload.get("jurisdiction", "EX"),
            initial_data=payload.get("state_data", {}),
        )
        ctx._state.version = payload.get("version", 0)
        ctx._patch_history = [DeltaPatch.model_validate(p) for p in payload.get("patch_history", [])]
        ctx._patch_bytes_total = payload.get("patch_bytes_total", 0)
        ctx.last_accessed_at = payload.get("last_accessed_at", time.time())
        return ctx

    def _persist_snapshot(self) -> None:
        """Persist workspace state snapshot atomically to disk if storage_path configured."""
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        tmp_path = f"{self.storage_path}.tmp"
        snap = self.snapshot()
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(snap.model_dump(mode="json"), fh, indent=2, default=str)
        os.replace(tmp_path, self.storage_path)

    @classmethod
    def replay(
        cls,
        case_id: str,
        jurisdiction: str,
        patches: list[DeltaPatch],
    ) -> WorkspaceContext:
        """Deterministically reconstruct workspace state by sequentially replaying delta patches."""
        ctx = cls(case_id=case_id, jurisdiction=jurisdiction)
        for patch in patches:
            # Replay with ignore version conflicts to restore deterministic sequence
            p_replay = patch.model_copy(update={"expected_version": None})
            ctx.apply_patch(p_replay)
        return ctx

    def calculate_token_reduction(
        self,
        agent_count: int = 8,
        turns_per_agent: int = 2,
    ) -> TokenReductionMetric:
        """Measure token volume reduction achieved by shared workspace vs direct conversational re-serialization."""
        with self._lock:
            self._touch()
            state_dict = self._state.to_dict()
            state_json = json.dumps(state_dict, default=str)
            state_bytes = len(state_json.encode("utf-8"))
            base_turn_tokens = max(50, state_bytes // 4)

            # Baseline calculation: Quadratic history accumulation
            baseline_tokens = 0
            cumulative_tokens = base_turn_tokens
            for _ in range(agent_count * turns_per_agent):
                baseline_tokens += cumulative_tokens
                cumulative_tokens += 120  # average agent turn addition

            # Workspace calculation: Scoped slice + compact delta patch
            avg_slice_tokens = int(base_turn_tokens * 0.32)
            avg_patch_tokens = max(25, self._patch_bytes_total // (max(1, len(self._patch_history)) * 4))
            workspace_tokens = agent_count * turns_per_agent * (avg_slice_tokens + avg_patch_tokens)

            reduction = max(0.0, (baseline_tokens - workspace_tokens) / max(1, baseline_tokens)) * 100.0

            return TokenReductionMetric(
                baseline_tokens=baseline_tokens,
                workspace_tokens=workspace_tokens,
                reduction_percentage=round(reduction, 2),
                agent_count=agent_count,
                patch_volume_bytes=self._patch_bytes_total,
                state_size_bytes=state_bytes,
            )

    def gist_statutory_context(self, program: str, rules_summary: list[dict[str, Any]] | None = None) -> str:
        """Compress static statutory rules into a dense semantic digest before model injection."""
        with self._lock:
            self._touch()
            gister = StatutoryContextGister()
            return gister.gist_program_rules(program, self.jurisdiction, rules_summary or [])

    def compress_prompt_preamble(self, text: str, target_ratio: float = 0.45) -> str:
        """Compress static statutory preambles into dense gist representations."""
        with self._lock:
            self._touch()
            gister = StatutoryContextGister()
            return gister.compress_preamble(text, target_ratio)


# --------------------------------------------------------------------------- #
# Prompt Gisting & Statutory Context Compressor
# --------------------------------------------------------------------------- #


class StatutoryContextGister:
    """Compresses lengthy statutory rules into structured semantic digests to eliminate prompt bloat.

    Preserves exact numeric matrices, boolean predicates, and statutory citation anchors
    while stripping legal boilerplate, preambles, and bureaucratic language.
    """

    _BOILERPLATE_PATTERNS = [
        re.compile(r"pursuant\s+to\s+(?:the\s+provisions\s+of\s+)?(?:section|title|part|\d+)", re.IGNORECASE),
        re.compile(r"notwithstanding\s+(?:any\s+other\s+provision\s+of\s+law|anything\s+to\s+the\s+contrary)", re.IGNORECASE),
        re.compile(r"it\s+is\s+hereby\s+(?:enacted|provided|ordered)\s+that", re.IGNORECASE),
        re.compile(r"for\s+the\s+purposes\s+of\s+this\s+(?:subpart|section|regulation)", re.IGNORECASE),
        re.compile(r"in\s+accordance\s+with\s+(?:federal|state)\s+guidelines", re.IGNORECASE),
    ]

    def compress_preamble(self, text: str, target_ratio: float = 0.45) -> str:
        """Remove legal filler while keeping core thresholds, citations, and predicates."""
        if not text:
            return ""

        cleaned = text
        for pat in self._BOILERPLATE_PATTERNS:
            cleaned = pat.sub("", cleaned)

        # Collapse excess whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Extract sentences with quantitative/statutory anchors
        sentences = [s.strip() for s in re.split(r"[.\n]", cleaned) if s.strip()]
        retained = []
        for s in sentences:
            # Retain sentences containing numbers, percentages, CFR/USC citations, or key status keywords
            if (
                re.search(r"\b(?:\d+|%|\$|CFR|USC|FPL|AMI|SNAP|Medicaid|Housing|Unemployment)\b", s, re.IGNORECASE)
                or any(k in s.lower() for k in ("eligible", "ineligible", "limit", "asset", "gross", "income", "window"))
            ):
                retained.append(s)

        if not retained:
            retained = sentences[:max(1, int(len(sentences) * target_ratio))]

        return ". ".join(retained) + ("." if retained else "")

    def gist_program_rules(self, program: str, jurisdiction: str, rules_list: list[dict[str, Any]]) -> str:
        """Formulate a dense structured semantic digest for program rules."""
        lines = [f"[STATUTORY-GIST: {program.upper()} | JURISDICTION={jurisdiction}]"]
        for r in rules_list:
            cid = r.get("criterion_id", "rule")
            title = r.get("title", "")
            req = "REQ" if r.get("required", True) else "OPT"
            cit = r.get("citation", "")
            desc = r.get("description", "")
            # Dense single-line format
            lines.append(f"• {cid} [{req}] {title}: {desc} ({cit})".strip())
        return "\n".join(lines)

    def calculate_compression_metrics(self, raw_text: str, gisted_text: str) -> dict[str, Any]:
        """Calculate token volume reduction and compression ratio."""
        raw_tokens = max(1, len(raw_text) // 4)
        gist_tokens = max(1, len(gisted_text) // 4)
        saved = max(0, raw_tokens - gist_tokens)
        ratio = round(gist_tokens / raw_tokens, 4)
        return {
            "raw_tokens": raw_tokens,
            "gisted_tokens": gist_tokens,
            "tokens_saved": saved,
            "compression_ratio": ratio,
            "reduction_percentage": round((1.0 - ratio) * 100.0, 2),
        }


# --------------------------------------------------------------------------- #
# Automated Session State Offloader (DiskBackedSessionManager)
# --------------------------------------------------------------------------- #


@dataclass
class SessionStub:
    """Lightweight in-memory reference to an offloaded/spilled session."""


    case_id: str
    jurisdiction: str
    version: int
    disk_path: str
    compressed_size_bytes: int
    spilled_at: float
    last_accessed_at: float


class DiskBackedSessionManager:
    """Automated session state offloader for multi-case runtime memory management.

    Monitors in-memory case contexts and automatically spills inactive case data
    (documents, OCR payloads, historical transcripts, and state) to compressed local
    disk storage (`.tribune/sessions/{case_id}.json.gz`) when idle or exceeding capacity,
    retaining lightweight `SessionStub` metadata in RAM. Provides transparent sync and async
    re-hydration on access.
    """

    def __init__(
        self,
        storage_dir: str = ".tribune/sessions",
        idle_timeout_seconds: float = 300.0,  # 5 minutes idle threshold
        max_active_sessions: int = 16,
    ) -> None:
        self.storage_dir = os.path.abspath(storage_dir)
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_active_sessions = max_active_sessions
        os.makedirs(self.storage_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._active_sessions: dict[str, WorkspaceContext] = {}
        self._spilled_stubs: dict[str, SessionStub] = {}
        self._bytes_spilled_total = 0

    def open_session(self, case_id: str, jurisdiction: str = "EX") -> WorkspaceContext:
        """Get existing session or initialize a new active WorkspaceContext."""
        with self._lock:
            if case_id in self._active_sessions:
                sess = self._active_sessions[case_id]
                sess.last_accessed_at = time.time()
                return sess

            if case_id in self._spilled_stubs:
                return self.rehydrate(case_id)

            sess = WorkspaceContext(case_id=case_id, jurisdiction=jurisdiction)
            self._active_sessions[case_id] = sess
            self._enforce_capacity_limit()
            return sess

    def get_session(self, case_id: str) -> WorkspaceContext | None:
        """Transparently get session, automatically rehydrating from disk if spilled."""
        with self._lock:
            if case_id in self._active_sessions:
                sess = self._active_sessions[case_id]
                sess.last_accessed_at = time.time()
                return sess
            if case_id in self._spilled_stubs:
                return self.rehydrate(case_id)
            return None

    async def get_session_async(self, case_id: str) -> WorkspaceContext | None:
        """Asynchronously get or rehydrate a workspace session."""
        async with self._async_lock:
            return await asyncio.to_thread(self.get_session, case_id)

    def spill_session(self, case_id: str) -> str | None:
        """Spill an active in-memory session to compressed gzip disk storage."""
        with self._lock:
            if case_id not in self._active_sessions:
                return None

            ctx = self._active_sessions[case_id]
            data = ctx.serialize_full_session()
            json_bytes = json.dumps(data, default=str).encode("utf-8")
            compressed = gzip.compress(json_bytes)

            file_path = os.path.join(self.storage_dir, f"{case_id}.json.gz")
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "wb") as fh:
                fh.write(compressed)
            os.replace(tmp_path, file_path)

            stub = SessionStub(
                case_id=case_id,
                jurisdiction=ctx.jurisdiction,
                version=ctx.version,
                disk_path=file_path,
                compressed_size_bytes=len(compressed),
                spilled_at=time.time(),
                last_accessed_at=ctx.last_accessed_at,
            )
            self._spilled_stubs[case_id] = stub
            self._bytes_spilled_total += len(compressed)
            del self._active_sessions[case_id]
            return file_path

    def rehydrate(self, case_id: str) -> WorkspaceContext:
        """Transparently decompress and rehydrate a spilled session from disk into active RAM."""
        with self._lock:
            if case_id in self._active_sessions:
                return self._active_sessions[case_id]

            if case_id not in self._spilled_stubs:
                raise FileNotFoundError(f"No active or offloaded session found for case_id '{case_id}'")

            stub = self._spilled_stubs[case_id]
            with gzip.open(stub.disk_path, "rb") as fh:
                json_bytes = fh.read()
            payload = json.loads(json_bytes.decode("utf-8"))

            ctx = WorkspaceContext.deserialize_full_session(payload)
            ctx.last_accessed_at = time.time()

            self._active_sessions[case_id] = ctx
            del self._spilled_stubs[case_id]
            self._enforce_capacity_limit()
            return ctx

    def check_and_spill_idle(self, now_override: float | None = None) -> list[str]:
        """Check all active sessions and spill those exceeding the idle timeout."""
        with self._lock:
            now = now_override if now_override is not None else time.time()
            spilled_ids = []
            for case_id, sess in list(self._active_sessions.items()):
                if now - sess.last_accessed_at >= self.idle_timeout_seconds:
                    self.spill_session(case_id)
                    spilled_ids.append(case_id)
            return spilled_ids

    def _enforce_capacity_limit(self) -> None:
        """Evict and spill least recently accessed active sessions if exceeding max active count."""
        if len(self._active_sessions) <= self.max_active_sessions:
            return

        sorted_sessions = sorted(self._active_sessions.values(), key=lambda s: s.last_accessed_at)
        excess = len(self._active_sessions) - self.max_active_sessions
        for sess in sorted_sessions[:excess]:
            self.spill_session(sess.case_id)

    def stats(self) -> dict[str, Any]:
        """Return runtime session memory and offloading statistics."""
        with self._lock:
            return {
                "active_sessions_count": len(self._active_sessions),
                "spilled_sessions_count": len(self._spilled_stubs),
                "bytes_spilled_total": self._bytes_spilled_total,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "max_active_sessions": self.max_active_sessions,
                "storage_dir": self.storage_dir,
            }


__all__ = [
    "WorkspaceState",
    "WorkspaceContext",
    "StatutoryContextGister",
    "PatchValidationError",
    "VersionConflictError",
    "SessionStub",
    "DiskBackedSessionManager",
]
