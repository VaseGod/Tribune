"""Structured consolidation schema IR + adversarial syntax sanitizer.

Replaces free-form NL summaries with a strictly typed JSON intermediate
representation. Any payload containing imperative instructions, system prompt
markers, or second-person directives is rejected and quarantined — never
allowed into HDM centroids.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

CONSOLIDATION_SCHEMA_VERSION = "tribune.consolidation/v1"

MAX_SUMMARY_CHARS = 500
MAX_ENTITIES = 64
MAX_ATTRS_PER_ENTITY = 32
MAX_FIELD_CHARS = 2000


class OperationalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    CONTRADICTED = "CONTRADICTED"


FORBIDDEN_MARKERS = [
    "SYSTEM:",
    "ALWAYS",
    "OVERRIDE",
    "IGNORE PREVIOUS",
    "IGNORE ALL PREVIOUS",
    "YOU MUST",
    "DO THIS",
    "ADMIN:",
    "ROOT:",
    "### SYSTEM",
    "<SYSTEM>",
    "[SYSTEM]",
    "PROMPT INJECTION",
    "DISREGARD",
    "BYPASS",
]

# second-person imperative: "you should/must...", "do X", etc.
IMPERATIVE_PATTERNS = [
    re.compile(r"\byou\s+(must|should|will|shall|need to|have to)\b", re.IGNORECASE),
    re.compile(r"^\s*(do|run|execute|delete|ignore|disregard|override|bypass)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b(obey|comply|follow)\s+(these|this|my)\s+(instructions|orders)\b", re.IGNORECASE),
]


def _norm(text: str) -> str:
    # NFKC normalization defeats homoglyph/width evasion before checks.
    return unicodedata.normalize("NFKC", text or "")


@dataclass
class AttributeMutation:
    attribute: str
    old_value: Any
    new_value: Any
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ConsolidatedEntity:
    entity_id: str
    entity_type: str
    mutations: list[AttributeMutation] = field(default_factory=list)
    status: OperationalStatus = OperationalStatus.ACTIVE


@dataclass
class ConsolidationIR:
    """Strictly typed schema intermediate representation."""

    schema_version: str = CONSOLIDATION_SCHEMA_VERSION
    source_episodic_ids: list[str] = field(default_factory=list)
    entities: list[ConsolidatedEntity] = field(default_factory=list)
    declarative_summary: str = ""
    generated_at: str = ""
    consolidator_id: str = "tribune-consolidator/v1"
    contradiction: dict[str, Any] | None = None
    decay: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_episodic_ids": list(self.source_episodic_ids),
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "status": e.status.value,
                    "mutations": [
                        {
                            "attribute": m.attribute,
                            "old_value": m.old_value,
                            "new_value": m.new_value,
                            "evidence_ids": list(m.evidence_ids),
                            "confidence": m.confidence,
                        }
                        for m in e.mutations
                    ],
                }
                for e in self.entities
            ],
            "declarative_summary": self.declarative_summary,
            "generated_at": self.generated_at,
            "consolidator_id": self.consolidator_id,
            "contradiction": self.contradiction,
            "decay": self.decay,
        }


class ConsolidationRejected(ValueError):
    """Raised when a consolidation payload fails schema/sanitizer validation."""


class ConsolidationSanitizer:
    """Validates schema IR and rejects instruction-injection content."""

    def __init__(self, max_summary_chars: int = MAX_SUMMARY_CHARS) -> None:
        self.max_summary_chars = max_summary_chars
        self._lock = threading.RLock()
        self.accepted = 0
        self.rejected = 0
        self.quarantine: list[dict[str, Any]] = []

    def _check_text(self, text: str, field_name: str) -> None:
        norm = _norm(text)
        upper = norm.upper()
        for marker in FORBIDDEN_MARKERS:
            if marker in upper:
                raise ConsolidationRejected(
                    f"Forbidden marker '{marker}' in field '{field_name}'."
                )
        for pat in IMPERATIVE_PATTERNS:
            if pat.search(norm):
                raise ConsolidationRejected(
                    f"Imperative/second-person directive in field '{field_name}'."
                )
        if len(norm) > MAX_FIELD_CHARS and field_name != "declarative_summary":
            raise ConsolidationRejected(f"Field '{field_name}' exceeds max length.")

    def validate(self, payload: dict[str, Any]) -> ConsolidationIR:
        with self._lock:
            if not isinstance(payload, dict):
                raise ConsolidationRejected("Payload must be a dict.")
            if payload.get("schema_version") != CONSOLIDATION_SCHEMA_VERSION:
                raise ConsolidationRejected("Unsupported or missing schema_version.")
            src = payload.get("source_episodic_ids")
            if not isinstance(src, list) or not src or not all(isinstance(s, str) for s in src):
                raise ConsolidationRejected("source_episodic_ids must be a non-empty list[str].")
            entities = payload.get("entities")
            if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
                raise ConsolidationRejected("entities must be a list within bounds.")
            parsed_entities: list[ConsolidatedEntity] = []
            for ent in entities:
                if not isinstance(ent, dict):
                    raise ConsolidationRejected("entity must be a dict.")
                eid = ent.get("entity_id")
                etype = ent.get("entity_type")
                if not isinstance(eid, str) or not eid or len(eid) > 256:
                    raise ConsolidationRejected("invalid entity_id.")
                if not isinstance(etype, str) or not etype or len(etype) > 128:
                    raise ConsolidationRejected("invalid entity_type.")
                self._check_text(eid, "entity_id")
                self._check_text(etype, "entity_type")
                try:
                    status = OperationalStatus(str(ent.get("status", "ACTIVE")))
                except ValueError as err:
                    raise ConsolidationRejected("invalid operational status.") from err
                muts = ent.get("mutations", [])
                if not isinstance(muts, list) or len(muts) > MAX_ATTRS_PER_ENTITY:
                    raise ConsolidationRejected("mutations out of bounds.")
                parsed_muts: list[AttributeMutation] = []
                for m in muts:
                    if not isinstance(m, dict):
                        raise ConsolidationRejected("mutation must be a dict.")
                    attr = m.get("attribute")
                    if not isinstance(attr, str) or not attr:
                        raise ConsolidationRejected("invalid mutation attribute.")
                    self._check_text(attr, "mutation.attribute")
                    conf = float(m.get("confidence", 1.0))
                    if not (0.0 <= conf <= 1.0):
                        raise ConsolidationRejected("confidence must be in [0.0, 1.0].")
                    ev = m.get("evidence_ids", [])
                    if not isinstance(ev, list) or not all(isinstance(x, str) for x in ev):
                        raise ConsolidationRejected("evidence_ids must be list[str].")
                    for ev_id in ev:
                        self._check_text(ev_id, "evidence_id")
                    parsed_muts.append(
                        AttributeMutation(
                            attribute=attr,
                            old_value=m.get("old_value"),
                            new_value=m.get("new_value"),
                            evidence_ids=list(ev),
                            confidence=conf,
                        )
                    )
                parsed_entities.append(
                    ConsolidatedEntity(
                        entity_id=eid, entity_type=etype, mutations=parsed_muts, status=status
                    )
                )
            summary = payload.get("declarative_summary", "")
            if not isinstance(summary, str):
                raise ConsolidationRejected("declarative_summary must be str.")
            if len(summary) > self.max_summary_chars:
                raise ConsolidationRejected("declarative_summary exceeds max length.")
            self._check_text(summary, "declarative_summary")
            # Reject any unexpected instruction-like top-level keys.
            allowed = {
                "schema_version",
                "source_episodic_ids",
                "entities",
                "declarative_summary",
                "generated_at",
                "consolidator_id",
                "contradiction",
                "decay",
            }
            extra = set(payload.keys()) - allowed
            if extra:
                raise ConsolidationRejected(f"Unexpected top-level fields: {sorted(extra)}.")
            for opt_key in ("contradiction", "decay"):
                val = payload.get(opt_key)
                if val is not None and not isinstance(val, dict):
                    raise ConsolidationRejected(f"{opt_key} must be a dict or null.")
                if isinstance(val, dict):
                    self._check_text(str(val), opt_key)
            gen_at = str(payload.get("generated_at", ""))
            cid = str(payload.get("consolidator_id", "tribune-consolidator/v1"))
            self._check_text(cid, "consolidator_id")
            ir = ConsolidationIR(
                source_episodic_ids=list(src),
                entities=parsed_entities,
                declarative_summary=summary,
                generated_at=gen_at,
                consolidator_id=cid,
                contradiction=payload.get("contradiction"),
                decay=payload.get("decay"),
            )
            self.accepted += 1
            return ir

    def sanitize_or_reject(self, payload: dict[str, Any]) -> ConsolidationIR:
        try:
            return self.validate(payload)
        except ConsolidationRejected as exc:
            with self._lock:
                self.rejected += 1
                self.quarantine.append(
                    {"payload": payload, "reason": str(exc), "at": time.time()}
                )
            logger.warning("[CONSOLIDATION-SANITIZER] Rejected payload: %s", exc)
            raise

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "accepted": self.accepted,
                "rejected": self.rejected,
                "quarantined": len(self.quarantine),
            }


__all__ = [
    "CONSOLIDATION_SCHEMA_VERSION",
    "OperationalStatus",
    "AttributeMutation",
    "ConsolidatedEntity",
    "ConsolidationIR",
    "ConsolidationRejected",
    "ConsolidationSanitizer",
    "FORBIDDEN_MARKERS",
    "MAX_SUMMARY_CHARS",
]
