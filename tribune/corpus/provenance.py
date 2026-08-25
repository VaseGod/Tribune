"""Anonymization + provenance tagging applied at ingest time.

A person's PII never needs to leave the deploying organization, but even within
it we minimize what is retained: free-text extracted from documents is scrubbed of
obvious direct identifiers before it is stored as evidence, and every piece of
evidence carries a :class:`Provenance` record (source document, ingest method,
content hash, anonymization flag).
"""

from __future__ import annotations

import hashlib
import re

from ..types import IngestMethod, Provenance

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]


def anonymize(text: str) -> tuple[str, list[str]]:
    """Redact obvious direct identifiers. Returns ``(clean_text, found_kinds)``.

    This is deliberately conservative and not a substitute for a full de-id
    pipeline; it removes the high-risk direct identifiers (SSN, email, phone,
    card numbers) that should never be persisted in a rule/evidence trace.
    """
    found: list[str] = []
    clean = text
    for kind, pattern in _PII_PATTERNS:
        if pattern.search(clean):
            found.append(kind)
            clean = pattern.sub(f"[REDACTED-{kind}]", clean)
    return clean, found


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_provenance(
    source_doc_id: str,
    ingest_method: IngestMethod,
    text: str,
    anonymized: bool,
    notes: str = "",
) -> Provenance:
    return Provenance(
        source_doc_id=source_doc_id,
        ingest_method=ingest_method,
        anonymized=anonymized,
        content_hash=content_hash(text),
        notes=notes,
    )


from datetime import datetime, timezone
from typing import Any
from pydantic import Field
from ..types import StrictModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FactInjectionRecord(StrictModel):
    """Provenance-tracked deterministic statutory fact injected dynamically from Engram RAM into agent workspace."""

    fact_key: str
    exact_value: Any
    citation_id: str
    statute_source: str
    content_hash: str
    provenance: Provenance
    notes: str = ""
    injected_at: datetime = Field(default_factory=_utcnow)


def make_injected_fact(
    fact_key: str,
    exact_value: Any,
    citation_id: str,
    statute_source: str,
    notes: str = "",
) -> FactInjectionRecord:
    """Create a provenance-tracked fact injection record for Engram RAM lookups."""
    raw_repr = f"{fact_key}:{exact_value}:{citation_id}:{statute_source}"
    prov = make_provenance(
        source_doc_id=f"engram_ram::{statute_source.replace(' ', '_')}",
        ingest_method=IngestMethod.STRUCTURED,
        text=raw_repr,
        anonymized=False,
        notes=f"Injected from Engram RAM table for {fact_key}",
    )
    return FactInjectionRecord(
        fact_key=fact_key,
        exact_value=exact_value,
        citation_id=citation_id,
        statute_source=statute_source,
        content_hash=content_hash(raw_repr),
        provenance=prov,
        notes=notes,
    )


__all__ = [
    "anonymize",
    "content_hash",
    "make_provenance",
    "FactInjectionRecord",
    "make_injected_fact",
]

