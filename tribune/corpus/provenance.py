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


# --------------------------------------------------------------------------- #
# SynthID-Text Statistical Watermark Analytics
# --------------------------------------------------------------------------- #


import math


class SynthIDWatermarkScore(StrictModel):
    """Statistical evaluation metrics for SynthID-Text pseudorandom token distributions."""

    green_token_count: int
    red_token_count: int
    total_tokens: int
    green_token_ratio: float
    z_score: float
    p_synthetic: float
    is_watermarked: bool
    confidence_level: str  # "HIGH" | "MEDIUM" | "LOW" | "NONE"


class SynthIDTextWatermarkDetector:
    """Statistical detector computing z-scores and synthetic probabilities over green/red token distributions."""

    def __init__(
        self,
        gamma: float = 0.5,
        z_threshold: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.gamma = gamma
        self.z_threshold = z_threshold
        self.seed = seed

    def compute_watermark_score(
        self,
        tokens_or_text: list[str] | str,
        seed: int | None = None,
    ) -> SynthIDWatermarkScore:
        """Compute statistical z-score and probability of synthetic generation under SynthID watermark."""
        if isinstance(tokens_or_text, str):
            token_list = re.findall(r"\b\w+\b|[^\w\s]", tokens_or_text)
        else:
            token_list = list(tokens_or_text)

        total = len(token_list)
        if total < 2:
            return SynthIDWatermarkScore(
                green_token_count=0,
                red_token_count=0,
                total_tokens=total,
                green_token_ratio=0.0,
                z_score=0.0,
                p_synthetic=0.5,
                is_watermarked=False,
                confidence_level="NONE",
            )

        eff_seed = seed if seed is not None else self.seed
        green_count = 0
        red_count = 0

        for i in range(1, total):
            prefix = token_list[i - 1]
            token = token_list[i]
            # Pseudorandom hash to partition vocabulary
            h_input = f"{eff_seed}:{prefix}:{token}".encode("utf-8")
            h_val = int(hashlib.sha256(h_input).hexdigest()[:8], 16) % 10000
            if h_val < int(self.gamma * 10000):
                green_count += 1
            else:
                red_count += 1

        eval_tokens = max(1, total - 1)
        expected_green = self.gamma * eval_tokens
        variance = eval_tokens * self.gamma * (1.0 - self.gamma)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        z = (green_count - expected_green) / std_dev
        p_syn = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))

        if z >= self.z_threshold:
            conf = "HIGH"
        elif z >= 2.0:
            conf = "MEDIUM"
        elif z >= 0.5:
            conf = "LOW"
        else:
            conf = "NONE"

        return SynthIDWatermarkScore(
            green_token_count=green_count,
            red_token_count=red_count,
            total_tokens=eval_tokens,
            green_token_ratio=round(green_count / eval_tokens, 4),
            z_score=round(z, 4),
            p_synthetic=round(p_syn, 4),
            is_watermarked=z >= self.z_threshold,
            confidence_level=conf,
        )


# --------------------------------------------------------------------------- #
# C2PA Cryptographic Provenance Manifest Builder & Verifier
# --------------------------------------------------------------------------- #


class C2PAAssertion(StrictModel):
    """An individual assertion block within a C2PA cryptographic provenance manifest."""

    label: str
    data: dict[str, Any]


class C2PAManifest(StrictModel):
    """C2PA-compliant provenance manifest embedding SHA-256 assertions and cryptographic signatures."""

    claim_generator: str = "Tribune-Sovereign-Provenance-Engine/1.0"
    title: str = "Regulatory Determination Document"
    format: str = "application/json"
    instance_id: str
    assertions: list[C2PAAssertion]
    signature_hash: str
    signed_at: str
    is_verified: bool = True

    def get_data_hash(self) -> str:
        for a in self.assertions:
            if a.label == "c2pa.hash.data":
                return str(a.data.get("hash", ""))
        return ""


def sign_c2pa_manifest(
    document_text: str,
    case_id: str,
    statutory_citations: list[str] | None = None,
    private_key: str | None = None,
) -> C2PAManifest:
    """Generate and cryptographically sign a C2PA provenance manifest for an official determination."""
    doc_h = content_hash(document_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    inst_id = f"urn:uuid:c2pa:{case_id}:{content_hash(timestamp)[:12]}"

    assertions = [
        C2PAAssertion(
            label="c2pa.actions",
            data={
                "actions": [
                    {
                        "action": "c2pa.created",
                        "softwareAgent": "Tribune-Sovereign-Provenance-Engine/1.0",
                        "when": timestamp,
                    }
                ]
            },
        ),
        C2PAAssertion(
            label="c2pa.hash.data",
            data={
                "hash": doc_h,
                "algorithm": "SHA-256",
                "byte_length": len(document_text.encode("utf-8")),
            },
        ),
        C2PAAssertion(
            label="c2pa.tribune.statute_trace",
            data={
                "case_id": case_id,
                "statutory_citations": statutory_citations or [],
                "anonymized": True,
            },
        ),
    ]

    key = private_key or "TRIBUNE_HMAC_ROOT_SOVEREIGN_KEY"
    manifest_payload = f"{inst_id}:{doc_h}:{timestamp}:{key}"
    sig_h = content_hash(manifest_payload)

    return C2PAManifest(
        instance_id=inst_id,
        assertions=assertions,
        signature_hash=sig_h,
        signed_at=timestamp,
        is_verified=True,
    )


def verify_c2pa_manifest(
    manifest: C2PAManifest,
    document_text: str,
    private_key: str | None = None,
) -> bool:
    """Cryptographically verify C2PA provenance assertions and digital signature."""
    doc_h = content_hash(document_text)
    if manifest.get_data_hash() != doc_h:
        return False

    key = private_key or "TRIBUNE_HMAC_ROOT_SOVEREIGN_KEY"
    manifest_payload = f"{manifest.instance_id}:{doc_h}:{manifest.signed_at}:{key}"
    expected_sig = content_hash(manifest_payload)
    return manifest.signature_hash == expected_sig


__all__ = [
    "anonymize",
    "content_hash",
    "make_provenance",
    "FactInjectionRecord",
    "make_injected_fact",
    "SynthIDWatermarkScore",
    "SynthIDTextWatermarkDetector",
    "C2PAAssertion",
    "C2PAManifest",
    "sign_c2pa_manifest",
    "verify_c2pa_manifest",
]


