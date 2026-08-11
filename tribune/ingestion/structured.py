"""Structured document ingestion (the default).

Reads the typed fields off a :class:`~tribune.types.RawDocument` and emits
provenance-tagged evidence. This is the path used by the synthetic generator and
by any caller that already has structured data (e.g. an intake form).
"""

from __future__ import annotations

from ..types import Evidence, IngestMethod, RawDocument
from .base import fields_to_evidence
from .ocr import fast_heuristic_parse


class StructuredIngest:
    name = "structured"

    def ingest(self, doc: RawDocument) -> list[Evidence]:
        fields = dict(doc.fields)
        if doc.text:
            fast_fields = fast_heuristic_parse(doc.text, doc.doc_type)
            for k, v in fast_fields.items():
                fields.setdefault(k, v)
        updated_doc = RawDocument(doc_id=doc.doc_id, doc_type=doc.doc_type, text=doc.text, fields=fields)
        return fields_to_evidence(updated_doc, IngestMethod.STRUCTURED)

    def ingest_many(self, docs: list[RawDocument]) -> list[Evidence]:
        out: list[Evidence] = []
        for d in docs:
            out.extend(self.ingest(d))
        return out

