"""Translation scaffolding for multilingual synthetic cases.

``translate_case`` builds a language twin of a synthetic case: the *document
text* is rendered in the target language via the glossary (so token counts and
any served model see target-language text), while the structured ``fields`` keep
their canonical keys so ingestion and ground truth remain identical — the twin
differs only in language, which is exactly what a parity audit needs.

Machine translation of legal/eligibility terms is never silent: every glossary
entry carries ``human_reviewed`` and the parity report lists unreviewed terms as
blockers for any real Spanish-facing deployment.
"""

from __future__ import annotations

from ...types import RawDocument, SyntheticCase
from . import es

_GLOSSARIES = {"es": es}

SUPPORTED_LANGUAGES = ("en", *_GLOSSARIES.keys())


def _translate_document(doc: RawDocument, glossary) -> RawDocument:
    lines = [glossary.label_for("intake_header")]
    for key, value in doc.fields.items():
        lines.append(f"{glossary.label_for(key)}: {value}")
    return RawDocument(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        text="\n".join(lines),
        fields=dict(doc.fields),  # canonical keys: ingestion behavior is identical
        image_path=doc.image_path,
    )


def translate_case(case: SyntheticCase, language: str) -> SyntheticCase:
    if language == "en":
        return case
    glossary = _GLOSSARIES.get(language)
    if glossary is None:
        raise ValueError(f"no glossary for language '{language}'; add tribune/casegen/i18n/{language}.py")
    documents = [_translate_document(d, glossary) for d in case.documents]
    return case.model_copy(update={"language": language, "documents": documents})


def unreviewed_legal_terms(language: str):
    glossary = _GLOSSARIES.get(language)
    return list(glossary.unreviewed_legal_terms()) if glossary else []
