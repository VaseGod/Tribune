"""Protocol-Aware Retention: StateDelta / EphemeralObservation split + cold storage.

Decouples durable structural state changes (StateDelta) from noisy raw tool
payloads (EphemeralObservation). Ephemeral payloads are content-addressed into
an append-only cold store and flushed from the active prompt window at turn end.

Security posture: timeline content, tool output, and logs are untrusted data.
StateDelta fields are typed/compact; raw observations never enter default
associative retrieval.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from .tokens import estimate_tokens as _backend_estimate_tokens


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def estimate_tokens(text: str) -> int:
    """Token estimate via the configured backend (see tribune.memory.tokens).

    Defaults to the repo-standard ~4-chars-per-token heuristic; set
    ``TRIBUNE_TOKEN_COUNTER=tiktoken`` for real tokenizer counts.
    """
    return _backend_estimate_tokens(text)


SCHEMA_VERSION = "tribune.retention/v1"


@dataclass(frozen=True)
class StateDelta:
    """Compact, durable structural state change. Safe for active context."""

    event_id: str
    parent_event_id: str | None
    turn_id: str
    session_id: str
    timestamp: float
    actor: str
    tool_name: str
    operation_type: str
    entity_ids: list[str] = field(default_factory=list)
    file_changes: list[dict[str, Any]] = field(default_factory=list)
    exit_code: int | None = None
    command_status: str = "unknown"
    result_summary: dict[str, Any] = field(default_factory=dict)
    causal_refs: list[str] = field(default_factory=list)
    observation_digest: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def active_tokens(self) -> int:
        return estimate_tokens(_stable_json(self.to_dict()))


@dataclass(frozen=True)
class EphemeralObservation:
    """Noisy raw execution payload. Excluded from default retrieval."""

    event_id: str
    stdout: str = ""
    stderr: str = ""
    raw_json: Any = None
    payload_size: int = 0
    created_at: float = field(default_factory=time.time)
    content_hash: str = ""
    cold_uri: str = ""
    serializer: str = "json/v1"

    @classmethod
    def build(
        cls,
        event_id: str,
        stdout: str = "",
        stderr: str = "",
        raw_json: Any = None,
        serializer: str = "json/v1",
    ) -> EphemeralObservation:
        body = _stable_json({"stdout": stdout, "stderr": stderr, "raw": raw_json})
        digest = _sha256_hex(body.encode("utf-8"))
        size = len(stdout) + len(stderr) + len(body)
        return cls(
            event_id=event_id,
            stdout=stdout,
            stderr=stderr,
            raw_json=raw_json,
            payload_size=size,
            created_at=time.time(),
            content_hash=digest,
            cold_uri=f"cold://observations/{digest}",
            serializer=serializer,
        )

    def active_tokens(self) -> int:
        return estimate_tokens(self.stdout) + estimate_tokens(self.stderr) + estimate_tokens(
            _stable_json(self.raw_json)
        )


@dataclass
class RetentionPolicy:
    """Configurable retention behaviour with safe defaults."""

    enabled: bool = True
    max_active_context_tokens: int = 32000
    max_active_events: int = 200
    ephemeral_ttl_s: float = 3600.0
    cold_storage_path: str = ".tribune/cold_observations.jsonl"
    emergency_debug_retention: bool = False
    loop_repeat_threshold: int = 3
    loop_similarity_threshold: float = 0.95
    loop_oscillation_window: int = 6
    block_repeated_calls: bool = False

    @classmethod
    def from_env(cls) -> RetentionPolicy:
        def _b(name: str, default: bool) -> bool:
            return os.getenv(name, str(default)).lower() in ("true", "1", "yes")

        return cls(
            enabled=_b("TRIBUNE_RETENTION_ENABLED", True),
            max_active_context_tokens=int(os.getenv("TRIBUNE_RETENTION_MAX_TOKENS", "32000")),
            max_active_events=int(os.getenv("TRIBUNE_RETENTION_MAX_EVENTS", "200")),
            ephemeral_ttl_s=float(os.getenv("TRIBUNE_RETENTION_EPHEMERAL_TTL_S", "3600")),
            cold_storage_path=os.getenv(
                "TRIBUNE_RETENTION_COLD_PATH", ".tribune/cold_observations.jsonl"
            ),
            emergency_debug_retention=_b("TRIBUNE_RETENTION_DEBUG_MODE", False),
            loop_repeat_threshold=int(os.getenv("TRIBUNE_RETENTION_LOOP_THRESHOLD", "3")),
            block_repeated_calls=_b("TRIBUNE_RETENTION_BLOCK_LOOPS", False),
        )


class ColdStorageBackend:
    """Abstract cold storage backend."""

    def write(self, obs: EphemeralObservation) -> str:
        raise NotImplementedError

    def read(self, content_hash: str) -> EphemeralObservation | None:
        raise NotImplementedError


class FileColdStorage(ColdStorageBackend):
    """Append-only content-addressed JSONL cold store (0600 file permissions).

    Holds raw tool output verbatim: restrict access per least privilege.
    For encryption at rest, wrap with :class:`EncryptedColdStorage`.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._harden_permissions()

    def _harden_permissions(self) -> None:
        try:
            if os.path.exists(self.path):
                os.chmod(self.path, 0o600)
        except OSError:
            pass

    def write(self, obs: EphemeralObservation) -> str:
        with self._lock:
            record = {
                "content_hash": obs.content_hash,
                "event_id": obs.event_id,
                "stdout": obs.stdout,
                "stderr": obs.stderr,
                "raw_json": obs.raw_json,
                "payload_size": obs.payload_size,
                "created_at": obs.created_at,
                "serializer": obs.serializer,
            }
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            fd = os.open(self.path, flags, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(_stable_json(record) + "\n")
        return f"cold://observations/{obs.content_hash}"

    def read(self, content_hash: str) -> EphemeralObservation | None:
        with self._lock:
            if not os.path.exists(self.path):
                return None
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or content_hash not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("content_hash") == content_hash:
                        return EphemeralObservation(
                            event_id=rec.get("event_id", ""),
                            stdout=rec.get("stdout", ""),
                            stderr=rec.get("stderr", ""),
                            raw_json=rec.get("raw_json"),
                            payload_size=int(rec.get("payload_size", 0)),
                            created_at=float(rec.get("created_at", time.time())),
                            content_hash=rec.get("content_hash", ""),
                            cold_uri=f"cold://observations/{content_hash}",
                            serializer=rec.get("serializer", "json/v1"),
                        )
        return None


class InMemoryColdStorage(ColdStorageBackend):
    """Test / fallback backend with no disk I/O."""

    def __init__(self) -> None:
        self._store: dict[str, EphemeralObservation] = {}
        self._lock = threading.RLock()

    def write(self, obs: EphemeralObservation) -> str:
        with self._lock:
            self._store[obs.content_hash] = obs
        return f"mem://observations/{obs.content_hash}"

    def read(self, content_hash: str) -> EphemeralObservation | None:
        with self._lock:
            return self._store.get(content_hash)


class EncryptedColdStorage(ColdStorageBackend):
    """Fernet-encrypted wrapper around any record-level cold backend.

    Requires the ``cryptography`` package (``pip install tribune[security]``).
    Fails closed with a clear error when the package is absent — raw tool
    output is never silently written unencrypted when encryption was requested.
    Key: 32-byte urlsafe-b64 Fernet key from ``TRIBUNE_RETENTION_COLD_ENCRYPT_KEY``
    (or ``_FILE`` / ``_CMD`` via the secret chain).
    """

    def __init__(self, inner: ColdStorageBackend, key_b64: str | None = None) -> None:
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except ImportError as err:
            raise ImportError(
                "EncryptedColdStorage requires the 'cryptography' package "
                "(pip install tribune[security]). Cold observations were NOT written."
            ) from err
        if key_b64 is None:
            from ..security.secrets import resolve_secret

            key_b64 = resolve_secret("TRIBUNE_RETENTION_COLD_ENCRYPT_KEY") or ""
        if not key_b64:
            raise ValueError(
                "EncryptedColdStorage needs TRIBUNE_RETENTION_COLD_ENCRYPT_KEY "
                "(or _FILE/_CMD). Refusing to run unencrypted."
            )
        self._fernet = Fernet(key_b64.encode("utf-8"))
        self._inner = inner

    def write(self, obs: EphemeralObservation) -> str:
        sealed = EphemeralObservation(
            event_id=obs.event_id,
            stdout=self._fernet.encrypt(obs.stdout.encode("utf-8")).decode("ascii"),
            stderr=self._fernet.encrypt(obs.stderr.encode("utf-8")).decode("ascii"),
            raw_json=self._fernet.encrypt(
                _stable_json(obs.raw_json).encode("utf-8")
            ).decode("ascii"),
            payload_size=obs.payload_size,
            created_at=obs.created_at,
            content_hash=obs.content_hash,
            cold_uri=obs.cold_uri,
            serializer="fernet/v1+" + obs.serializer,
        )
        return self._inner.write(sealed)

    def read(self, content_hash: str) -> EphemeralObservation | None:
        sealed = self._inner.read(content_hash)
        if sealed is None:
            return None
        try:
            stdout = self._fernet.decrypt(sealed.stdout.encode("ascii")).decode("utf-8")
            stderr = self._fernet.decrypt(sealed.stderr.encode("ascii")).decode("utf-8")
            raw_json = json.loads(
                self._fernet.decrypt(
                    sealed.raw_json.encode("ascii")  # type: ignore[arg-type]
                ).decode("utf-8")
            )
        except Exception as err:
            raise ValueError(f"Cold record failed to decrypt for '{content_hash}'.") from err
        return EphemeralObservation(
            event_id=sealed.event_id,
            stdout=stdout,
            stderr=stderr,
            raw_json=raw_json,
            payload_size=sealed.payload_size,
            created_at=sealed.created_at,
            content_hash=sealed.content_hash,
            cold_uri=sealed.cold_uri,
            serializer=sealed.serializer.replace("fernet/v1+", ""),
        )


def build_cold_storage(
    policy: RetentionPolicy, key_b64: str | None = None
) -> ColdStorageBackend:
    """Build the configured cold backend: encrypted when a key is provided."""
    inner: ColdStorageBackend = FileColdStorage(policy.cold_storage_path)
    if (
        key_b64
        or os.getenv("TRIBUNE_RETENTION_COLD_ENCRYPT_KEY", "")
        or os.getenv("TRIBUNE_RETENTION_COLD_ENCRYPT_KEY_FILE", "")
        or os.getenv("TRIBUNE_RETENTION_COLD_ENCRYPT_KEY_CMD", "")
    ):
        return EncryptedColdStorage(inner, key_b64)
    return inner


def _normalize_call_sig(tool_name: str, args: Any) -> str:
    norm = unicodedata.normalize("NFKC", f"{tool_name.strip().lower()}:{_stable_json(args)}")
    return _sha256_hex(norm.encode("utf-8"))


@dataclass
class ToolLoopSignal:
    kind: str  # "exact_repeat" | "no_new_delta" | "oscillation"
    tool_name: str
    count: int
    message: str
    blocked: bool = False


class ToolLoopDetector:
    """Detects repeated / oscillating tool calls and no-op deltas."""

    def __init__(
        self,
        repeat_threshold: int = 3,
        similarity_threshold: float = 0.95,
        oscillation_window: int = 6,
    ) -> None:
        self.repeat_threshold = repeat_threshold
        self.similarity_threshold = similarity_threshold
        self.oscillation_window = oscillation_window
        self._history: list[tuple[str, str, str]] = []  # (sig, tool, delta_hash)
        self._lock = threading.RLock()
        self.detections: list[ToolLoopSignal] = []
        self.repeated_call_count = 0

    def _args_similar(self, a: Any, b: Any) -> bool:
        sa, sb = _stable_json(a), _stable_json(b)
        if sa == sb:
            return True
        # cheap Jaccard over char 3-grams
        def grams(s: str) -> set[str]:
            return {s[i : i + 3] for i in range(max(0, len(s) - 2))} or {s}

        ga, gb = grams(sa), grams(sb)
        if not ga or not gb:
            return False
        jacc = len(ga & gb) / max(1, len(ga | gb))
        return jacc >= self.similarity_threshold

    def observe(
        self,
        tool_name: str,
        args: Any,
        delta: dict[str, Any] | None,
    ) -> list[ToolLoopSignal]:
        signals: list[ToolLoopSignal] = []
        with self._lock:
            sig = _normalize_call_sig(tool_name, args)
            delta_hash = _sha256_hex(_stable_json(delta or {}).encode())
            self._history.append((sig, tool_name, delta_hash))
            window = self._history[-self.oscillation_window :]

            # exact / near-identical repeats
            same_tool_recent = [
                h for h in self._history[-self.repeat_threshold :] if h[1] == tool_name
            ]
            if len(same_tool_recent) >= self.repeat_threshold and all(
                h[0] == sig for h in same_tool_recent
            ):
                self.repeated_call_count += 1
                s = ToolLoopSignal(
                    kind="exact_repeat",
                    tool_name=tool_name,
                    count=len(same_tool_recent),
                    message=(
                        f"Tool '{tool_name}' repeated identically "
                        f"{len(same_tool_recent)}x; possible tool-call loop."
                    ),
                )
                signals.append(s)
                self.detections.append(s)

            # no-new-delta repeats
            if len(self._history) >= 2 and delta is not None and not delta:
                prev = self._history[-2]
                if prev[1] == tool_name:
                    s = ToolLoopSignal(
                        kind="no_new_delta",
                        tool_name=tool_name,
                        count=2,
                        message=(
                            f"Tool '{tool_name}' produced no new StateDelta on "
                            "consecutive calls."
                        ),
                    )
                    signals.append(s)
                    self.detections.append(s)

            # oscillation A,B,A,B...
            if len(window) >= 4:
                tools = [h[1] for h in window[-4:]]
                if tools[0] == tools[2] and tools[1] == tools[3] and tools[0] != tools[1]:
                    s = ToolLoopSignal(
                        kind="oscillation",
                        tool_name=tool_name,
                        count=4,
                        message=(
                            f"Oscillating tool pattern detected: "
                            f"{tools[0]} <-> {tools[1]}."
                        ),
                    )
                    signals.append(s)
                    self.detections.append(s)
        return signals


@dataclass
class RetentionMetrics:
    events_split: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_flushed: int = 0
    cold_writes: int = 0
    loop_detections: int = 0
    blocked_calls: int = 0
    active_context_tokens: int = 0

    def context_reduction_ratio(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return max(0.0, (self.tokens_before - self.tokens_after) / self.tokens_before)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_split": self.events_split,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_flushed": self.tokens_flushed,
            "cold_writes": self.cold_writes,
            "loop_detections": self.loop_detections,
            "blocked_calls": self.blocked_calls,
            "active_context_tokens": self.active_context_tokens,
            "context_reduction_ratio": round(self.context_reduction_ratio(), 4),
        }


__all__ = [
    "SCHEMA_VERSION",
    "StateDelta",
    "EphemeralObservation",
    "RetentionPolicy",
    "ColdStorageBackend",
    "FileColdStorage",
    "InMemoryColdStorage",
    "EncryptedColdStorage",
    "build_cold_storage",
    "ToolLoopSignal",
    "ToolLoopDetector",
    "RetentionMetrics",
    "estimate_tokens",
]
