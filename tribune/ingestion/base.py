"""The DocIngest interface and the shared field->evidence mapping.

Ingestion turns a document (a structured record, or text extracted from a scan) into
typed, provenance-tagged :class:`~tribune.types.Evidence`. PII is scrubbed before
anything is persisted.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..corpus.provenance import anonymize, make_provenance
from ..types import Evidence, EvidenceType, IngestMethod, RawDocument

_NUMERIC: set[EvidenceType] = {
    EvidenceType.HOUSEHOLD_SIZE,
    EvidenceType.MONTHLY_INCOME,
    EvidenceType.ANNUAL_INCOME,
    EvidenceType.LIQUID_ASSETS,
    EvidenceType.AGE,
    EvidenceType.BASE_PERIOD_EARNINGS,
    EvidenceType.WEEKS_WORKED,
    EvidenceType.MONTHLY_RENT,
    EvidenceType.DAYS_SINCE_DENIAL,
}

_BOOL: set[EvidenceType] = {
    EvidenceType.RESIDENT,
    EvidenceType.DISABLED,
    EvidenceType.PREGNANT,
    EvidenceType.HAS_DEPENDENT_CHILD,
    EvidenceType.ABLE_AND_AVAILABLE,
}


def coerce_value(etype: EvidenceType, raw: str) -> float | bool | str | None:
    raw = raw.strip()
    if raw == "":
        return None
    if etype in _NUMERIC:
        try:
            return float(raw.replace(",", "").replace("$", ""))
        except ValueError:
            return None
    if etype in _BOOL:
        return raw.lower() in ("true", "yes", "y", "1")
    return raw


def fields_to_evidence(
    doc: RawDocument, method: IngestMethod
) -> list[Evidence]:
    """Map a document's structured fields to typed evidence with provenance."""
    clean_text, _found = anonymize(doc.text or "")
    prov = make_provenance(
        source_doc_id=doc.doc_id,
        ingest_method=method,
        text=clean_text or doc.doc_type,
        anonymized=True,
        notes=f"doc_type={doc.doc_type}",
    )
    out: list[Evidence] = []
    for key, raw in doc.fields.items():
        try:
            etype = EvidenceType(key)
        except ValueError:
            continue  # unknown field key -> skip rather than guess
        value = coerce_value(etype, raw)
        if value is None:
            continue
        out.append(
            Evidence(
                evidence_id=f"{doc.doc_id}:{etype.value}",
                type=etype,
                value=value,
                provenance=prov,
            )
        )
    return out


@runtime_checkable
class DocIngest(Protocol):
    name: str

    def ingest(self, doc: RawDocument) -> list[Evidence]: ...

    def ingest_many(self, docs: list[RawDocument]) -> list[Evidence]: ...


def make_doc_ingest(settings=None) -> DocIngest:
    from ..config import get_settings

    settings = settings or get_settings()
    if settings.doc_ingest == "ocr":
        from .ocr import OcrIngest

        return OcrIngest(settings=settings)
    from .structured import StructuredIngest

    return StructuredIngest()
