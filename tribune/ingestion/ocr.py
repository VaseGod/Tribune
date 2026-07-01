"""OCR-based document intake.

Turns a photo/scan of a benefit letter, pay stub, or form into structured,
provenance-tagged evidence. The OCR engine sits behind this adapter:

* If ``TRIBUNE_OCR_ENDPOINT`` is set and a document has an ``image_path``, the
  adapter posts the image to a self-hosted OCR server (e.g. an Unlimited-OCR
  deployment) and parses the returned text.
* Otherwise it falls back to parsing the document's already-extracted ``text``
  (``key: value`` lines), so the OCR code path runs end-to-end offline.

Either way the extracted text is anonymized before evidence is created.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from ..config import TribuneSettings
from ..types import Evidence, EvidenceType, IngestMethod, RawDocument
from .base import coerce_value, fields_to_evidence

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*[:=]\s*(.+?)\s*$")


def _normalize_key(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def parse_text_to_fields(text: str) -> dict[str, str]:
    """Parse ``key: value`` lines from OCR'd text into known evidence fields."""
    fields: dict[str, str] = {}
    known = {e.value for e in EvidenceType}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        key = _normalize_key(m.group(1))
        if key in known:
            fields[key] = m.group(2)
    return fields


class OcrIngest:
    name = "ocr"

    def __init__(self, settings: TribuneSettings) -> None:
        self._endpoint = settings.ocr_endpoint
        self._timeout = settings.request_timeout_s

    def _ocr_image(self, image_path: str) -> str:  # pragma: no cover - needs endpoint
        payload = json.dumps({"image_path": image_path}).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint.rstrip("/") + "/ocr",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"OCR endpoint {self._endpoint} failed: {exc}") from exc
        return body.get("text", "")

    def ingest(self, doc: RawDocument) -> list[Evidence]:
        # Path 1: a real scan + configured OCR server.
        if self._endpoint and doc.image_path:  # pragma: no cover - needs endpoint
            text = self._ocr_image(doc.image_path)
            parsed = RawDocument(
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                text=text,
                fields=parse_text_to_fields(text),
            )
            return fields_to_evidence(parsed, IngestMethod.OCR)

        # Path 2 (offline fallback): parse already-extracted text, or, if the
        # document carries structured fields, use those directly.
        fields = dict(doc.fields)
        if not fields and doc.text:
            fields = parse_text_to_fields(doc.text)
        else:
            # Validate/normalize any text-derived values that are present.
            fields = {k: v for k, v in fields.items() if coerce_value(_safe_type(k), v) is not None}
        parsed = RawDocument(doc_id=doc.doc_id, doc_type=doc.doc_type, text=doc.text, fields=fields)
        return fields_to_evidence(parsed, IngestMethod.OCR)

    def ingest_many(self, docs: list[RawDocument]) -> list[Evidence]:
        out: list[Evidence] = []
        for d in docs:
            out.extend(self.ingest(d))
        return out


def _safe_type(key: str) -> EvidenceType:
    try:
        return EvidenceType(key)
    except ValueError:
        return EvidenceType.RESIDENCY_STATE  # treated as text; harmless for filtering
