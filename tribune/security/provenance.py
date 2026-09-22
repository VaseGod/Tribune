"""Cryptographic provenance: HMAC signing + tamper-evident audit hash chain.

Signature = HMAC_K(canonical(SourceEpisodicIDs) || canonical(SchemaPayload) || Timestamp)

- SHA-256 or stronger (HMAC-SHA256).
- Key from secure config / env / secret manager; never hardcoded.
- Key IDs for rotation; old nodes verify against retired keys.
- Canonical JSON: sorted keys, no pretty-print ambiguity.
- UTC ISO-8601 timestamps.
- Append-only audit log with hash chain:
    entry_digest = hash(previous_hash || event_type || payload_digest || signature || timestamp)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = "tribune.provenance/v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_digest_hex(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_source_ids(source_ids: list[str]) -> str:
    return canonical_json(sorted(set(source_ids)))


@dataclass
class HMACKeyRing:
    """Secret key holder with rotation support. Never logs secret material."""

    active_key_id: str = "k1"
    keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        key_env: str = "TRIBUNE_HMAC_SECRET",
        key_id_env: str = "TRIBUNE_HMAC_KEY_ID",
    ) -> HMACKeyRing:
        from .secrets import resolve_secret

        key_id = os.getenv(key_id_env, "k1")
        secret = resolve_secret(key_env) or ""
        keys: dict[str, str] = {}
        if secret:
            keys[key_id] = secret
        # Rotated keys via direct env (TRIBUNE_HMAC_SECRET_<KEYID>) or secret
        # refs (TRIBUNE_HMAC_SECRET_<KEYID>_FILE / _CMD, any case).
        prefix = key_env + "_"
        for k, v in os.environ.items():
            if not k.startswith(prefix) or not v:
                continue
            if k.upper().endswith(("_FILE", "_CMD")):
                continue
            keys[k[len(prefix) :].lower()] = v
        for k in os.environ:
            upper = k.upper()
            if not k.startswith(prefix) or not upper.endswith(("_FILE", "_CMD")):
                continue
            stem = k[len(prefix) :]
            base = key_env + "_" + stem[: -len("_FILE" if upper.endswith("_FILE") else "_CMD")]
            resolved = resolve_secret(base)
            if resolved:
                keys[base[len(prefix) :].lower()] = resolved
        if not keys:
            # Dev-only ephemeral key. Production must set TRIBUNE_HMAC_SECRET.
            logger.warning(
                "[PROVENANCE] No HMAC secret configured; using ephemeral dev key. "
                "Set TRIBUNE_HMAC_SECRET (or _FILE/_CMD) in production."
            )
            keys[key_id] = hashlib.sha256(f"ephemeral-{time.time()}".encode()).hexdigest()
        return cls(active_key_id=key_id, keys=keys)

    def secret_for(self, key_id: str) -> str | None:
        return self.keys.get(key_id)

    def rotate(self, new_key_id: str, secret: str) -> None:
        self.keys[new_key_id] = secret
        self.active_key_id = new_key_id


def sign_consolidation(
    source_ids: list[str],
    schema_payload: dict[str, Any],
    timestamp: str,
    keyring: HMACKeyRing,
    key_id: str | None = None,
) -> dict[str, str]:
    """Compute HMAC-SHA256 signature over canonical message."""
    kid = key_id or keyring.active_key_id
    secret = keyring.secret_for(kid)
    if not secret:
        raise PermissionError(f"Unknown HMAC key id '{kid}'. Failing closed.")
    message = (
        canonical_source_ids(source_ids)
        + canonical_json(schema_payload)
        + timestamp
    ).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return {"signature": sig, "key_id": kid}


def verify_consolidation_signature(
    source_ids: list[str],
    schema_payload: dict[str, Any],
    timestamp: str,
    signature: str,
    keyring: HMACKeyRing,
    key_id: str,
) -> bool:
    secret = keyring.secret_for(key_id)
    if not secret:
        return False
    message = (
        canonical_source_ids(source_ids)
        + canonical_json(schema_payload)
        + timestamp
    ).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@dataclass
class ProvenanceAuditEntry:
    sequence: int
    event_type: str
    timestamp: str
    key_id: str
    source_ids: list[str]
    payload_digest: str
    signature: str
    previous_hash: str
    entry_digest: str = ""
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_entry_digest(
    previous_hash: str, event_type: str, payload_digest: str, signature: str, timestamp: str
) -> str:
    msg = (previous_hash + event_type + payload_digest + signature + timestamp).encode("utf-8")
    return hashlib.sha256(msg).hexdigest()


class ProvenanceAuditLog:
    """Append-only tamper-evident log for consolidated-node provenance."""

    def __init__(self, keyring: HMACKeyRing | None = None) -> None:
        self._entries: list[ProvenanceAuditEntry] = []
        self._lock = threading.RLock()
        self._keyring = keyring or HMACKeyRing.from_env()
        self.verification_failures = 0
        self.blocked_unsigned = 0

    @property
    def keyring(self) -> HMACKeyRing:
        return self._keyring

    def sign_and_append(
        self,
        source_ids: list[str],
        schema_payload: dict[str, Any],
        event_type: str = "CONSOLIDATION_SIGNED",
        timestamp: str | None = None,
    ) -> ProvenanceAuditEntry:
        with self._lock:
            ts = timestamp or utc_now_iso()
            digest = payload_digest_hex(schema_payload)
            signed = sign_consolidation(source_ids, schema_payload, ts, self._keyring)
            prev = self._entries[-1].entry_digest if self._entries else ""
            entry_digest = compute_entry_digest(
                prev, event_type, digest, signed["signature"], ts
            )
            entry = ProvenanceAuditEntry(
                sequence=len(self._entries),
                event_type=event_type,
                timestamp=ts,
                key_id=signed["key_id"],
                source_ids=sorted(set(source_ids)),
                payload_digest=digest,
                signature=signed["signature"],
                previous_hash=prev,
                entry_digest=entry_digest,
            )
            self._entries.append(entry)
            return entry

    def verify_chain(self) -> bool:
        with self._lock:
            prev = ""
            for e in self._entries:
                if e.previous_hash != prev:
                    return False
                expected = compute_entry_digest(
                    e.previous_hash, e.event_type, e.payload_digest, e.signature, e.timestamp
                )
                if expected != e.entry_digest:
                    return False
                prev = e.entry_digest
            return True

    def verify_entry_signature(
        self,
        source_ids: list[str],
        schema_payload: dict[str, Any],
        timestamp: str,
        signature: str,
        key_id: str,
    ) -> bool:
        ok = verify_consolidation_signature(
            source_ids, schema_payload, timestamp, signature, self._keyring, key_id
        )
        with self._lock:
            if not ok:
                self.verification_failures += 1
        return ok

    def lookup(self, payload_digest: str) -> ProvenanceAuditEntry | None:
        with self._lock:
            for e in self._entries:
                if e.payload_digest == payload_digest:
                    return e
        return None

    def entries(self) -> list[ProvenanceAuditEntry]:
        with self._lock:
            return list(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "chain_valid": self.verify_chain(),
                "verification_failures": self.verification_failures,
                "blocked_unsigned": self.blocked_unsigned,
                "active_key_id": self._keyring.active_key_id,
            }


_GLOBAL_PROVENANCE_LOG = ProvenanceAuditLog()


def get_provenance_log() -> ProvenanceAuditLog:
    return _GLOBAL_PROVENANCE_LOG


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "HMACKeyRing",
    "ProvenanceAuditEntry",
    "ProvenanceAuditLog",
    "get_provenance_log",
    "sign_consolidation",
    "verify_consolidation_signature",
    "compute_entry_digest",
    "canonical_json",
    "payload_digest_hex",
    "canonical_source_ids",
    "utc_now_iso",
]
