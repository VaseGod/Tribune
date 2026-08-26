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
from dataclasses import dataclass, field
from typing import Any

from ..config import TribuneSettings
from ..types import Evidence, EvidenceType, IngestMethod, RawDocument
from .base import coerce_value, fields_to_evidence


# --------------------------------------------------------------------------- #
# Visual Layout Embeddings, Bounding Boxes, & Reading-Order DAGs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box coordinates (0.0 to 1.0) on a 2D document plane."""

    x0: float
    y0: float
    x1: float
    y1: float

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def overlaps(self, other: BoundingBox) -> bool:
        return not (
            self.x1 < other.x0 or self.x0 > other.x1 or self.y1 < other.y0 or self.y0 > other.y1
        )


@dataclass
class VisualLayoutToken:
    """A visual layout element with bounding box coordinates and semantic layout type."""

    token_id: str
    text: str
    bbox: BoundingBox
    token_type: str  # "HEADER" | "KEY_VALUE" | "PARAGRAPH" | "TABLE_CELL" | "SIGNATURE_BLOCK" | "NOTICE_DATE" | "SEAL"
    confidence: float = 1.0
    reading_order_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualLayoutDAG:
    """Directed Acyclic Graph modeling 2D spatial and hierarchical reading order flow."""

    nodes: list[VisualLayoutToken] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (from_token_id, to_token_id)

    def topological_reading_order(self) -> list[VisualLayoutToken]:
        """Compute strict topological reading order across multi-column DAG nodes."""
        node_map = {n.token_id: n for n in self.nodes}
        indeg = {n.token_id: 0 for n in self.nodes}
        for u, v in self.edges:
            if v in indeg:
                indeg[v] += 1

        # Priority queue / sort by (center_y, center_x) when in-degree is zero
        queue = [nid for nid, deg in indeg.items() if deg == 0]
        queue.sort(key=lambda nid: (node_map[nid].bbox.y0, node_map[nid].bbox.x0))

        order: list[VisualLayoutToken] = []
        while queue:
            curr_id = queue.pop(0)
            order.append(node_map[curr_id])
            for u, v in self.edges:
                if u == curr_id and v in indeg:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        queue.append(v)
            queue.sort(key=lambda nid: (node_map[nid].bbox.y0, node_map[nid].bbox.x0))

        if len(order) != len(self.nodes):
            # Fallback for unlinked or cyclic nodes
            remaining = [n for n in self.nodes if n not in order]
            remaining.sort(key=lambda n: (n.bbox.y0, n.bbox.x0))
            order.extend(remaining)

        return order


@dataclass
class VisualDocumentLayout:
    """Complete visual document layout structure containing layout tokens and reading-order DAG."""

    doc_id: str
    doc_type: str = "generic_notice"
    page_count: int = 1
    tokens: list[VisualLayoutToken] = field(default_factory=list)
    layout_dag: VisualLayoutDAG = field(default_factory=VisualLayoutDAG)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "tokens": [
                {
                    "token_id": t.token_id,
                    "text": t.text,
                    "bbox": [t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1],
                    "token_type": t.token_type,
                    "confidence": t.confidence,
                    "reading_order_index": t.reading_order_index,
                }
                for t in self.tokens
            ],
            "edges": list(self.layout_dag.edges),
            "metadata": self.metadata,
        }


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

    def extract_layout(self, doc: RawDocument) -> VisualDocumentLayout:
        """Extract visual bounding boxes and reading-order directed acyclic graph (DAG) from document."""
        text = doc.text or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        total_lines = max(1, len(lines))

        tokens: list[VisualLayoutToken] = []
        edges: list[tuple[str, str]] = []

        # If empty text, generate fallback token from document fields
        if not lines and doc.fields:
            lines = [f"{k}: {v}" for k, v in doc.fields.items()]
            total_lines = len(lines)

        prev_token_id: str | None = None
        for idx, line in enumerate(lines):
            token_id = f"tok_{doc.doc_id}_{idx + 1}"
            y0 = round(0.05 + (idx / total_lines) * 0.85, 3)
            y1 = round(y0 + (0.85 / total_lines), 3)

            # Heuristic token classification & column coordinate calculation
            if idx == 0 or any(h in line.upper() for h in ("NOTICE", "DETERMINATION", "DEPARTMENT", "STATE OF", "AGENCY")):
                token_type = "HEADER"
                x0, x1 = 0.10, 0.90
            elif any(s in line.lower() for s in ("signature", "authorized by", "caseworker:", "signed:")):
                token_type = "SIGNATURE_BLOCK"
                x0, x1 = 0.50, 0.90
            elif re.search(r"\b(?:date|dated|issued)\s*[:=]\s*\d+", line, re.IGNORECASE):
                token_type = "NOTICE_DATE"
                x0, x1 = 0.60, 0.90
            elif _LINE_RE.match(line) or any(c in line for c in (":", "=")):
                token_type = "KEY_VALUE"
                x0, x1 = 0.10, 0.50
            else:
                token_type = "PARAGRAPH"
                x0, x1 = 0.10, 0.90

            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
            tok = VisualLayoutToken(
                token_id=token_id,
                text=line,
                bbox=bbox,
                token_type=token_type,
                confidence=0.98,
                reading_order_index=idx + 1,
            )
            tokens.append(tok)

            if prev_token_id is not None:
                edges.append((prev_token_id, token_id))
            prev_token_id = token_id

        layout_dag = VisualLayoutDAG(nodes=tokens, edges=edges)
        return VisualDocumentLayout(
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            page_count=1,
            tokens=tokens,
            layout_dag=layout_dag,
            metadata={"extracted_at": time.time()},
        )

    def ingest_with_layout(self, doc: RawDocument) -> tuple[list[Evidence], VisualDocumentLayout]:
        """Ingest document returning both structured evidence and visual layout graph."""
        evidence = self.ingest(doc)
        layout = self.extract_layout(doc)
        return evidence, layout

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


__all__ = [
    "BoundingBox",
    "VisualLayoutToken",
    "VisualLayoutDAG",
    "VisualDocumentLayout",
    "fast_heuristic_parse",
    "parse_text_to_fields",
    "OcrIngest",
]

