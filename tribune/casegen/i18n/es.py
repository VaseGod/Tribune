"""Spanish (es) glossary for synthetic case documents.

Every entry is machine-drafted and starts with ``human_reviewed=False``.
**Legal/eligibility terms must never be silently machine-translated**: entries
with ``legal_term=True`` are surfaced by the parity report as requiring review
by a bilingual navigator, benefits counselor, or legal-aid staffer before any
Spanish-facing deployment. Flipping ``human_reviewed`` to True is a reviewed,
attributable change (it will show in git blame) — do not flip it in bulk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    en: str
    es: str
    legal_term: bool = False
    human_reviewed: bool = False  # flips only after bilingual domain review


# Keyed by the canonical English label (EvidenceType values + document phrases).
GLOSSARY: dict[str, GlossaryEntry] = {
    e.en: e
    for e in [
        # Household / income facts.
        GlossaryEntry("household_size", "tamaño del hogar"),
        GlossaryEntry("monthly_income", "ingreso mensual", legal_term=True),
        GlossaryEntry("annual_income", "ingreso anual", legal_term=True),
        GlossaryEntry("liquid_assets", "activos líquidos", legal_term=True),
        GlossaryEntry("residency_state", "estado de residencia", legal_term=True),
        GlossaryEntry("resident", "residente", legal_term=True),
        GlossaryEntry("citizenship_status", "estatus de ciudadanía", legal_term=True),
        GlossaryEntry("age", "edad"),
        GlossaryEntry("disabled", "con discapacidad", legal_term=True),
        GlossaryEntry("pregnant", "embarazada"),
        GlossaryEntry("has_dependent_child", "tiene hijo/a dependiente", legal_term=True),
        # Employment / unemployment insurance.
        GlossaryEntry("employment_status", "situación laboral", legal_term=True),
        GlossaryEntry("separation_reason", "motivo de separación laboral", legal_term=True),
        GlossaryEntry("base_period_earnings", "ingresos del período base", legal_term=True),
        GlossaryEntry("weeks_worked", "semanas trabajadas", legal_term=True),
        GlossaryEntry("able_and_available", "capaz y disponible para trabajar", legal_term=True),
        # Housing.
        GlossaryEntry("monthly_rent", "alquiler mensual"),
        GlossaryEntry("waitlist_status", "estado de la lista de espera", legal_term=True),
        # Appeals.
        GlossaryEntry("denial_date", "fecha de denegación", legal_term=True),
        GlossaryEntry("days_since_denial", "días desde la denegación", legal_term=True),
        GlossaryEntry("appeal_grounds", "fundamentos de la apelación", legal_term=True),
        # Document phrases.
        GlossaryEntry("application_intake", "formulario de solicitud"),
        GlossaryEntry(
            "intake_header",
            "Datos declarados por la persona solicitante (documento sintético)",
        ),
    ]
}


def label_for(en_label: str) -> str:
    entry = GLOSSARY.get(en_label)
    return entry.es if entry else en_label


def unreviewed_legal_terms() -> list[GlossaryEntry]:
    return [e for e in GLOSSARY.values() if e.legal_term and not e.human_reviewed]


def unreviewed_terms() -> list[GlossaryEntry]:
    return [e for e in GLOSSARY.values() if not e.human_reviewed]
