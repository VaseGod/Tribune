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
import time
import urllib.error
import urllib.request

from ..config import TribuneSettings
from ..types import Evidence, EvidenceType, IngestMethod, RawDocument
from .base import coerce_value, fields_to_evidence

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*[:=]\s*(.+?)\s*$")


def _normalize_key(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


_ALIAS_MAP: dict[str, str] = {
    "gross_income": EvidenceType.MONTHLY_INCOME.value,
    "gross_pay": EvidenceType.MONTHLY_INCOME.value,
    "monthly_pay": EvidenceType.MONTHLY_INCOME.value,
    "total_wages": EvidenceType.MONTHLY_INCOME.value,
    "base_earnings": EvidenceType.BASE_PERIOD_EARNINGS.value,
    "earnings": EvidenceType.BASE_PERIOD_EARNINGS.value,
    "rent": EvidenceType.MONTHLY_RENT.value,
    "rent_amount": EvidenceType.MONTHLY_RENT.value,
    "lease_amount": EvidenceType.MONTHLY_RENT.value,
    "days_denied": EvidenceType.DAYS_SINCE_DENIAL.value,
    "denial_days": EvidenceType.DAYS_SINCE_DENIAL.value,
    "household": EvidenceType.HOUSEHOLD_SIZE.value,
    "family_size": EvidenceType.HOUSEHOLD_SIZE.value,
    "state": EvidenceType.RESIDENCY_STATE.value,
    "jurisdiction": EvidenceType.RESIDENCY_STATE.value,
}

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:gross\s+(?:income|pay)|monthly\s+pay|total\s+wages)\s*[:=$-]\s*\$?([0-9,]+(?:\.[0-9]+)?)", re.I), EvidenceType.MONTHLY_INCOME.value),
    (re.compile(r"(?:annual\s+income|yearly\s+pay)\s*[:=$-]\s*\$?([0-9,]+(?:\.[0-9]+)?)", re.I), EvidenceType.ANNUAL_INCOME.value),
    (re.compile(r"(?:base\s+period\s+earnings|wages)\s*[:=$-]\s*\$?([0-9,]+(?:\.[0-9]+)?)", re.I), EvidenceType.BASE_PERIOD_EARNINGS.value),
    (re.compile(r"(?:monthly\s+rent|rent\s+amount|rent)\s*[:=$-]\s*\$?([0-9,]+(?:\.[0-9]+)?)", re.I), EvidenceType.MONTHLY_RENT.value),
    (re.compile(r"(?:days\s+since\s+denial|denial\s+days|days\s+elapsed)\s*[:=$-]\s*([0-9]+)", re.I), EvidenceType.DAYS_SINCE_DENIAL.value),
    (re.compile(r"(?:household\s+size|family\s+size|members)\s*[:=$-]\s*([0-9]+)", re.I), EvidenceType.HOUSEHOLD_SIZE.value),
    (re.compile(r"(?:liquid\s+assets|bank\s+balance|assets)\s*[:=$-]\s*\$?([0-9,]+(?:\.[0-9]+)?)", re.I), EvidenceType.LIQUID_ASSETS.value),
    (re.compile(r"(?:residency\s+state|state|jurisdiction)\s*[:=$-]\s*([A-Za-z]{2})", re.I), EvidenceType.RESIDENCY_STATE.value),
]


def fast_heuristic_parse(text: str, doc_type: str = "") -> dict[str, str]:
    """Lightweight, high-speed heuristic parsing pipeline using layout extraction regexes.

    Completes initial text parsing before falling back to heavy VLM or OCR processing.
    Handles wage stubs, lease agreements, and decision notices.
    """
    fields: dict[str, str] = {}
    if not text:
        return fields

    known = {e.value for e in EvidenceType}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m:
            key = _normalize_key(m.group(1))
            val = m.group(2).strip()
            mapped_key = _ALIAS_MAP.get(key, key)
            if mapped_key in known:
                fields[mapped_key] = val

    for pattern, etype_val in _PATTERNS:
        if etype_val not in fields:
            match = pattern.search(text)
            if match:
                fields[etype_val] = match.group(1)

    return fields


def parse_text_to_fields(text: str) -> dict[str, str]:
    """Parse ``key: value`` lines from text into known evidence fields via fast heuristic path."""
    return fast_heuristic_parse(text)


class OcrIngest:
    name = "ocr"

    def __init__(self, settings: TribuneSettings) -> None:
        self._endpoint = settings.ocr_endpoint
        self._timeout = settings.request_timeout_s
        self.last_latency_ms: float = 0.0

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
        start_t = time.perf_counter()
        fields = dict(doc.fields)
        if doc.text:
            fast_fields = fast_heuristic_parse(doc.text, doc.doc_type)
            for k, v in fast_fields.items():
                fields.setdefault(k, v)

        # Path 1: If text or fields fast path completed, parse directly
        if fields:
            fields = {k: v for k, v in fields.items() if coerce_value(_safe_type(k), str(v)) is not None}
            parsed = RawDocument(doc_id=doc.doc_id, doc_type=doc.doc_type, text=doc.text, fields=fields)
            res = fields_to_evidence(parsed, IngestMethod.OCR)
            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return res

        # Path 2: Heavier OCR endpoint processing fallback for images without extracted text
        if self._endpoint and doc.image_path:  # pragma: no cover - needs endpoint
            text = self._ocr_image(doc.image_path)
            fast_fields = fast_heuristic_parse(text, doc.doc_type)
            parsed = RawDocument(
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                text=text,
                fields=fast_fields,
            )
            res = fields_to_evidence(parsed, IngestMethod.OCR)
            self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
            return res

        fields = {k: v for k, v in fields.items() if coerce_value(_safe_type(k), str(v)) is not None}
        parsed = RawDocument(doc_id=doc.doc_id, doc_type=doc.doc_type, text=doc.text, fields=fields)
        res = fields_to_evidence(parsed, IngestMethod.OCR)
        self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
        return res

    def ingest_many(self, docs: list[RawDocument]) -> list[Evidence]:
        start_t = time.perf_counter()
        out: list[Evidence] = []
        for d in docs:
            out.extend(self.ingest(d))
        self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
        return out


def _safe_type(key: str) -> EvidenceType:
    try:
        return EvidenceType(key)
    except ValueError:
        return EvidenceType.RESIDENCY_STATE  # treated as text; harmless for filtering
