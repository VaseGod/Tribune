"""Generate synthetic benefit-notice documents carrying adversarial payloads.

Takes a legitimate synthetic case and produces a tampered twin whose notice
document embeds a payload at the requested placement. The document text models
what the OCR path would hand downstream — including OCR-artifact placements where
the malicious strip is rendered as the noisy, spaced-out text a scanner would
emit from a low-contrast region.

Only synthetic data is used; no real notices, no PII.
"""

from __future__ import annotations

from ..types import RawDocument, SyntheticCase
from .payloads import Payload, Placement

_LEGIT_BODY = (
    "STATE BENEFITS OFFICE — NOTICE OF ACTION\n"
    "This letter concerns your application for assistance. Please review the "
    "information below and contact your caseworker with any questions.\n"
)


def _mangle(text: str) -> str:
    """Approximate how OCR renders a low-contrast / degraded region."""
    # Real OCR of a faint strip yields broken spacing and stray glyphs; keep the
    # words recoverable so the probe tests the *hardest* case (payload survives
    # OCR) rather than an easy one (payload destroyed by noise).
    return "~ " + text.replace("  ", " ") + " ~"


def build_notice(case_id: str, payload: Payload, base_fields: dict[str, str]) -> RawDocument:
    """Build one adversarial benefit-notice document embedding ``payload``."""
    header = "BENEFIT DETERMINATION NOTICE"
    footer = "End of notice. Keep for your records."
    body_lines = [_LEGIT_BODY]

    # Legitimate-looking structured fields the notice would carry.
    field_lines = [f"{k}: {v}" for k, v in base_fields.items()]

    if payload.placement is Placement.HEADER:
        header = f"{header}\n{payload.text}"
    elif payload.placement is Placement.FOOTER:
        footer = f"{payload.text}\n{footer}"
    elif payload.placement is Placement.BODY:
        body_lines.append(payload.text)
    elif payload.placement is Placement.FORM_FIELD:
        field_lines.append(payload.text)
    elif payload.placement is Placement.OCR_ARTIFACT:
        body_lines.append(_mangle(payload.text))

    text = "\n".join([header, *body_lines, *field_lines, footer])
    return RawDocument(
        doc_id=f"{case_id}:adv:{payload.payload_id}",
        doc_type="benefit_notice",
        text=text,
        # A scanned notice arrives as text, not pre-structured fields: force the
        # OCR text-parsing path to do the extraction (this is the attack surface).
        fields={},
        image_path=None,
    )


def tamper_case(case: SyntheticCase, payload: Payload) -> SyntheticCase:
    """Return a twin of ``case`` whose documents include the adversarial notice.

    The original legitimate documents are preserved; the malicious notice is
    appended, modeling a real intake where a forged notice arrives alongside
    genuine paperwork.
    """
    base_fields = dict(case.documents[0].fields) if case.documents else {}
    notice = build_notice(case.case_id, payload, base_fields)
    return case.model_copy(update={"documents": [*case.documents, notice]})
