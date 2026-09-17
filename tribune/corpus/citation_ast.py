"""Deterministic Citation Abstract Syntax Tree (AST) Parser & Verifier.

Parses statutory codes, federal regulations, case law reporters, and administrative
rules into a structured AST representation, validating citations deterministically
against official reporter registries, jurisdictional rules, and rule stores
WITHOUT incurring LLM evaluation costs.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class CitationType(str, enum.Enum):
    """Categorical taxonomy of legal and statutory citations."""

    STATUTE_USC = "STATUTE_USC"
    REGULATION_CFR = "REGULATION_CFR"
    CASE_LAW_REPORTER = "CASE_LAW_REPORTER"
    STATE_ADMIN_RULE = "STATE_ADMIN_RULE"
    CONSTITUTIONAL = "CONSTITUTIONAL"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(str, enum.Enum):
    """Deterministic validation status for a citation."""

    VALID = "VALID"
    MALFORMED = "MALFORMED"
    UNKNOWN_REPORTER = "UNKNOWN_REPORTER"
    INVALID_JURISDICTION = "INVALID_JURISDICTION"
    UNGROUNDED = "UNGROUNDED"


@dataclass
class CitationASTNode:
    """Structured AST representation of a parsed citation."""

    raw_text: str
    normalized_citation: str
    citation_type: CitationType
    title_or_volume: str = ""
    reporter_or_statute: str = ""
    section_or_page: str = ""
    year: int | None = None
    jurisdiction: str = "US"
    pin_cite: str | None = None
    subdivisions: list[str] = field(default_factory=list)


@dataclass
class CitationVerificationGateResult:
    """Structured output emitted by the deterministic citation verifier gate."""

    is_valid: bool
    citation_text: str
    normalized_citation: str
    citation_type: str
    jurisdiction: str
    reporter_or_statute_family: str
    validation_status: ValidationStatus
    failure_reason: str | None = None
    remediation_hint: str | None = None
    node: CitationASTNode | None = None


# --------------------------------------------------------------------------- #
# Official Reporter & Statutory Registries
# --------------------------------------------------------------------------- #

# Recognized Federal & State Case Law Reporters
OFFICIAL_REPORTERS: dict[str, dict[str, str]] = {
    # US Supreme Court
    "U.S.": {"jurisdiction": "US", "family": "SCOTUS", "name": "United States Reports"},
    "US": {"jurisdiction": "US", "family": "SCOTUS", "name": "United States Reports"},
    "S. Ct.": {"jurisdiction": "US", "family": "SCOTUS", "name": "Supreme Court Reporter"},
    "S.Ct.": {"jurisdiction": "US", "family": "SCOTUS", "name": "Supreme Court Reporter"},
    "L. Ed.": {"jurisdiction": "US", "family": "SCOTUS", "name": "Lawyers' Edition"},
    "L. Ed. 2d": {"jurisdiction": "US", "family": "SCOTUS", "name": "Lawyers' Edition Second"},
    # Federal Courts of Appeals
    "F.": {"jurisdiction": "US_FED", "family": "CIRCUIT", "name": "Federal Reporter"},
    "F.2d": {"jurisdiction": "US_FED", "family": "CIRCUIT", "name": "Federal Reporter Second"},
    "F.3d": {"jurisdiction": "US_FED", "family": "CIRCUIT", "name": "Federal Reporter Third"},
    "F.4th": {"jurisdiction": "US_FED", "family": "CIRCUIT", "name": "Federal Reporter Fourth"},
    # Federal District Courts
    "F. Supp.": {"jurisdiction": "US_DIST", "family": "DISTRICT", "name": "Federal Supplement"},
    "F. Supp. 2d": {"jurisdiction": "US_DIST", "family": "DISTRICT", "name": "Federal Supplement Second"},
    "F. Supp. 3d": {"jurisdiction": "US_DIST", "family": "DISTRICT", "name": "Federal Supplement Third"},
    # Regional Reporters
    "A.": {"jurisdiction": "ATLANTIC", "family": "REGIONAL", "name": "Atlantic Reporter"},
    "A.2d": {"jurisdiction": "ATLANTIC", "family": "REGIONAL", "name": "Atlantic Reporter Second"},
    "A.3d": {"jurisdiction": "ATLANTIC", "family": "REGIONAL", "name": "Atlantic Reporter Third"},
    "N.E.": {"jurisdiction": "NORTH_EAST", "family": "REGIONAL", "name": "North Eastern Reporter"},
    "N.E.2d": {"jurisdiction": "NORTH_EAST", "family": "REGIONAL", "name": "North Eastern Reporter Second"},
    "N.E.3d": {"jurisdiction": "NORTH_EAST", "family": "REGIONAL", "name": "North Eastern Reporter Third"},
    "N.W.": {"jurisdiction": "NORTH_WEST", "family": "REGIONAL", "name": "North Western Reporter"},
    "N.W.2d": {"jurisdiction": "NORTH_WEST", "family": "REGIONAL", "name": "North Western Reporter Second"},
    "P.": {"jurisdiction": "PACIFIC", "family": "REGIONAL", "name": "Pacific Reporter"},
    "P.2d": {"jurisdiction": "PACIFIC", "family": "REGIONAL", "name": "Pacific Reporter Second"},
    "P.3d": {"jurisdiction": "PACIFIC", "family": "REGIONAL", "name": "Pacific Reporter Third"},
    "S.E.": {"jurisdiction": "SOUTH_EAST", "family": "REGIONAL", "name": "South Eastern Reporter"},
    "S.E.2d": {"jurisdiction": "SOUTH_EAST", "family": "REGIONAL", "name": "South Eastern Reporter Second"},
    "S.W.": {"jurisdiction": "SOUTH_WEST", "family": "REGIONAL", "name": "South Western Reporter"},
    "S.W.2d": {"jurisdiction": "SOUTH_WEST", "family": "REGIONAL", "name": "South Western Reporter Second"},
    "S.W.3d": {"jurisdiction": "SOUTH_WEST", "family": "REGIONAL", "name": "South Western Reporter Third"},
    "So.": {"jurisdiction": "SOUTHERN", "family": "REGIONAL", "name": "Southern Reporter"},
    "So. 2d": {"jurisdiction": "SOUTHERN", "family": "REGIONAL", "name": "Southern Reporter Second"},
    "So. 3d": {"jurisdiction": "SOUTHERN", "family": "REGIONAL", "name": "Southern Reporter Third"},
}

# Recognized Statutory Titles / Subjects
KNOWN_USC_TITLES: set[str] = {
    "7",   # Agriculture (SNAP)
    "20",  # Education
    "26",  # Internal Revenue Code
    "29",  # Labor
    "42",  # Public Health and Welfare (Medicaid, Housing, Social Security)
    "45",  # Railroads
}

KNOWN_CFR_TITLES: set[str] = {
    "7",   # Agriculture (SNAP regulations: 7 CFR 273)
    "20",  # Employees' Benefits (Unemployment Insurance: 20 CFR 604)
    "24",  # Housing and Urban Development (Section 8)
    "42",  # Public Health (Medicaid: 42 CFR 435)
    "45",  # Public Welfare
}

# Recognized State Jurisdictions in Tribune
KNOWN_JURISDICTIONS: set[str] = {
    "US", "FEDERAL", "EX", "NX", "CA", "NY", "TX", "FL", "IL", "PA", "OH", "MI"
}


# --------------------------------------------------------------------------- #
# AST Parser Engine
# --------------------------------------------------------------------------- #


class CitationASTParser:
    """Parses citation strings into structured CitationASTNode instances."""

    # Regex for Federal Regulations: e.g. "7 C.F.R. § 273.9", "7 CFR 273.9", "42 C.F.R. 435.603"
    _CFR_PATTERN = re.compile(
        r"\b(?P<title>\d+)\s*(?:C\.?\s*F\.?\s*R\.?)\s*(?:§+|sec\.?|section)?\s*(?P<section>\d+(?:\.\d+)?(?:\([a-zA-Z0-9]+\))*)",
        re.IGNORECASE,
    )

    # Regex for United States Code: e.g. "42 U.S.C. § 1396a", "7 USC 2014", "42 U.S.C. 1437f"
    _USC_PATTERN = re.compile(
        r"\b(?P<title>\d+)\s*(?:U\.?\s*S\.?\s*C\.?(?:\s*App\.?)?)\s*(?:§+|sec\.?|section)?\s*(?P<section>\d+[a-z]*(?:\([a-zA-Z0-9]+\))*)",
        re.IGNORECASE,
    )

    # Regex for Case Law Reporters: e.g. "397 U.S. 254", "424 U.S. 319", "123 F.3d 456 (2d Cir. 1997)", "100 Fake. Rep. 200"
    _REPORTER_PATTERN = re.compile(
        r"\b(?P<volume>\d+)\s+(?P<reporter>[A-Za-z][A-Za-z0-9.]*(?:\s+[A-Za-z0-9.]+)*?)\s+(?P<page>\d+)(?:\s*,\s*(?P<pin>\d+))?(?:\s*\((?:[^)]*?(?P<year>\d{4}))\))?",
        re.IGNORECASE,
    )

    # Regex for State Administrative Rules: e.g. "EX-SNAP-001", "EX-UI-004", "EX-MEDICAID-002", "NYCRR 358-1"
    _STATE_RULE_PATTERN = re.compile(
        r"\b(?P<jurisdiction>[A-Z]{2})-(?P<program>[A-Z]+)-(?P<rule_id>\d{3,4})\b|"
        r"\b(?P<state>[A-Z]{2,4})\s+(?:Admin\.?\s*Code|CRR)\s*(?:§+|sec\.?)?\s*(?P<sec>[0-9.-]+)",
        re.IGNORECASE,
    )

    # Regex for Constitutional References: e.g. "U.S. Const. amend. XIV", "US Const. Art. III"
    _CONST_PATTERN = re.compile(
        r"\b(?:U\.?S\.?\s*Const\.?)\s*(?:amend\.?\s*(?P<amend>[IVXLCDM0-9]+)|art\.?\s*(?P<art>[IVXLCDM0-9]+))",
        re.IGNORECASE,
    )

    def parse_single(self, text: str) -> CitationASTNode:
        """Parse a single citation string into an AST node."""
        clean = text.strip()

        # 1. Test CFR
        m_cfr = self._CFR_PATTERN.search(clean)
        if m_cfr:
            title = m_cfr.group("title")
            section = m_cfr.group("section")
            normalized = f"{title} C.F.R. § {section}"
            return CitationASTNode(
                raw_text=clean,
                normalized_citation=normalized,
                citation_type=CitationType.REGULATION_CFR,
                title_or_volume=title,
                reporter_or_statute="C.F.R.",
                section_or_page=section,
                jurisdiction="US",
            )

        # 2. Test USC
        m_usc = self._USC_PATTERN.search(clean)
        if m_usc:
            title = m_usc.group("title")
            section = m_usc.group("section")
            normalized = f"{title} U.S.C. § {section}"
            return CitationASTNode(
                raw_text=clean,
                normalized_citation=normalized,
                citation_type=CitationType.STATUTE_USC,
                title_or_volume=title,
                reporter_or_statute="U.S.C.",
                section_or_page=section,
                jurisdiction="US",
            )

        # 3. Test State Administrative Rules
        m_state = self._STATE_RULE_PATTERN.search(clean)
        if m_state:
            if m_state.group("jurisdiction"):
                jur = m_state.group("jurisdiction").upper()
                prog = m_state.group("program").upper()
                rid = m_state.group("rule_id")
                normalized = f"{jur}-{prog}-{rid}"
                return CitationASTNode(
                    raw_text=clean,
                    normalized_citation=normalized,
                    citation_type=CitationType.STATE_ADMIN_RULE,
                    title_or_volume=prog,
                    reporter_or_statute="STATE_RULE",
                    section_or_page=rid,
                    jurisdiction=jur,
                )
            else:
                st = m_state.group("state").upper()
                sec = m_state.group("sec")
                normalized = f"{st} Admin. Code § {sec}"
                return CitationASTNode(
                    raw_text=clean,
                    normalized_citation=normalized,
                    citation_type=CitationType.STATE_ADMIN_RULE,
                    title_or_volume=st,
                    reporter_or_statute="ADMIN_CODE",
                    section_or_page=sec,
                    jurisdiction=st[:2],
                )

        # 4. Test Case Law Reporters
        m_rep = self._REPORTER_PATTERN.search(clean)
        if m_rep:
            vol = m_rep.group("volume")
            rep_raw = m_rep.group("reporter").strip()
            # Normalize reporter string
            rep_norm = rep_raw.replace(" ", "")
            page = m_rep.group("page")
            pin = m_rep.group("pin")
            year_str = m_rep.group("year")
            year = int(year_str) if year_str else None

            # Standardize reporter key
            rep_clean = "U.S." if rep_norm in ("U.S.", "US") else rep_raw
            normalized = f"{vol} {rep_clean} {page}"
            if pin:
                normalized += f", {pin}"
            if year:
                normalized += f" ({year})"

            rep_meta = OFFICIAL_REPORTERS.get(rep_clean, OFFICIAL_REPORTERS.get(rep_norm, {}))
            jur = rep_meta.get("jurisdiction", "US")

            return CitationASTNode(
                raw_text=clean,
                normalized_citation=normalized,
                citation_type=CitationType.CASE_LAW_REPORTER,
                title_or_volume=vol,
                reporter_or_statute=rep_clean,
                section_or_page=page,
                year=year,
                jurisdiction=jur,
                pin_cite=pin,
            )

        # 5. Test Constitutional References
        m_const = self._CONST_PATTERN.search(clean)
        if m_const:
            detail = f"amend. {m_const.group('amend')}" if m_const.group("amend") else f"Art. {m_const.group('art')}"
            normalized = f"U.S. Const. {detail}"
            return CitationASTNode(
                raw_text=clean,
                normalized_citation=normalized,
                citation_type=CitationType.CONSTITUTIONAL,
                reporter_or_statute="U.S. Const.",
                jurisdiction="US",
            )

        # Unrecognized / Malformed
        return CitationASTNode(
            raw_text=clean,
            normalized_citation=clean,
            citation_type=CitationType.UNKNOWN,
            jurisdiction="UNKNOWN",
        )


# --------------------------------------------------------------------------- #
# Deterministic Citation Verifier Gate
# --------------------------------------------------------------------------- #


class CitationVerifierGate:
    """Deterministic verifier gate validating citations against official standards."""

    def __init__(
        self,
        allowed_jurisdictions: set[str] | None = None,
        rule_store: Any | None = None,
        strict_mode: bool = True,
    ) -> None:
        self.allowed_jurisdictions = allowed_jurisdictions or KNOWN_JURISDICTIONS
        self.rule_store = rule_store
        self.strict_mode = strict_mode
        self.parser = CitationASTParser()

    def verify_citation(
        self,
        citation_text: str,
        expected_jurisdiction: str | None = None,
    ) -> CitationVerificationGateResult:
        """Deterministically verify citation validity, reporter legitimacy, and jurisdictional fit."""
        if not citation_text or not citation_text.strip():
            return CitationVerificationGateResult(
                is_valid=False,
                citation_text=citation_text,
                normalized_citation="",
                citation_type=CitationType.UNKNOWN.value,
                jurisdiction="UNKNOWN",
                reporter_or_statute_family="NONE",
                validation_status=ValidationStatus.MALFORMED,
                failure_reason="Empty citation string provided.",
                remediation_hint="Provide a recognized legal or statutory citation.",
            )

        node = self.parser.parse_single(citation_text)

        # 1. Reject unknown or malformed patterns
        if node.citation_type == CitationType.UNKNOWN:
            return CitationVerificationGateResult(
                is_valid=False,
                citation_text=citation_text,
                normalized_citation=citation_text,
                citation_type=CitationType.UNKNOWN.value,
                jurisdiction="UNKNOWN",
                reporter_or_statute_family="UNKNOWN",
                validation_status=ValidationStatus.MALFORMED,
                failure_reason=f"Citation '{citation_text}' does not match any valid statutory or reporter grammar.",
                remediation_hint="Ensure citation follows standard formats (e.g. '7 CFR § 273.9', '42 USC § 1396a', '397 U.S. 254', 'EX-SNAP-001').",
                node=node,
            )

        # 2. Validate Case Law Reporters against official registry
        if node.citation_type == CitationType.CASE_LAW_REPORTER:
            norm_rep = node.reporter_or_statute.strip()
            # Try lookup with standard variations
            matched_meta = OFFICIAL_REPORTERS.get(norm_rep)
            if not matched_meta:
                matched_meta = OFFICIAL_REPORTERS.get(norm_rep.replace(" ", ""))

            if not matched_meta:
                return CitationVerificationGateResult(
                    is_valid=False,
                    citation_text=citation_text,
                    normalized_citation=node.normalized_citation,
                    citation_type=node.citation_type.value,
                    jurisdiction=node.jurisdiction,
                    reporter_or_statute_family="UNKNOWN",
                    validation_status=ValidationStatus.UNKNOWN_REPORTER,
                    failure_reason=f"Reporter '{norm_rep}' is not an official federal or regional reporter.",
                    remediation_hint="Use an official reporter citation: U.S., S. Ct., F.3d, F.4th, F. Supp. 3d, etc.",
                    node=node,
                )

        # 3. Validate CFR / USC Titles
        if node.citation_type == CitationType.REGULATION_CFR:
            if node.title_or_volume not in KNOWN_CFR_TITLES:
                return CitationVerificationGateResult(
                    is_valid=False,
                    citation_text=citation_text,
                    normalized_citation=node.normalized_citation,
                    citation_type=node.citation_type.value,
                    jurisdiction=node.jurisdiction,
                    reporter_or_statute_family="C.F.R.",
                    validation_status=ValidationStatus.MALFORMED,
                    failure_reason=f"CFR Title {node.title_or_volume} is not a valid public benefits / labor title.",
                    remediation_hint=f"Valid benefits CFR titles are {sorted(list(KNOWN_CFR_TITLES))}.",
                    node=node,
                )

        if node.citation_type == CitationType.STATUTE_USC:
            if node.title_or_volume not in KNOWN_USC_TITLES:
                return CitationVerificationGateResult(
                    is_valid=False,
                    citation_text=citation_text,
                    normalized_citation=node.normalized_citation,
                    citation_type=node.citation_type.value,
                    jurisdiction=node.jurisdiction,
                    reporter_or_statute_family="U.S.C.",
                    validation_status=ValidationStatus.MALFORMED,
                    failure_reason=f"USC Title {node.title_or_volume} is not a recognized public assistance title.",
                    remediation_hint=f"Valid benefits USC titles are {sorted(list(KNOWN_USC_TITLES))}.",
                    node=node,
                )

        # 4. Validate Jurisdictional Boundaries
        target_jur = expected_jurisdiction or node.jurisdiction
        if target_jur not in self.allowed_jurisdictions and target_jur not in ("US", "US_FED", "FEDERAL"):
            return CitationVerificationGateResult(
                is_valid=False,
                citation_text=citation_text,
                normalized_citation=node.normalized_citation,
                citation_type=node.citation_type.value,
                jurisdiction=node.jurisdiction,
                reporter_or_statute_family=node.reporter_or_statute,
                validation_status=ValidationStatus.INVALID_JURISDICTION,
                failure_reason=f"Jurisdiction '{node.jurisdiction}' is outside allowed jurisdictions: {self.allowed_jurisdictions}",
                remediation_hint="Limit state citations to approved jurisdictions.",
                node=node,
            )
        if expected_jurisdiction and node.jurisdiction not in ("US", "US_FED", "FEDERAL") and node.jurisdiction != expected_jurisdiction:
            return CitationVerificationGateResult(
                is_valid=False,
                citation_text=citation_text,
                normalized_citation=node.normalized_citation,
                citation_type=node.citation_type.value,
                jurisdiction=node.jurisdiction,
                reporter_or_statute_family=node.reporter_or_statute,
                validation_status=ValidationStatus.INVALID_JURISDICTION,
                failure_reason=f"Citation jurisdiction '{node.jurisdiction}' does not match expected jurisdiction '{expected_jurisdiction}'",
                remediation_hint=f"Use statutory authorities specifically for jurisdiction '{expected_jurisdiction}'.",
                node=node,
            )

        # Valid citation passes all deterministic verifier gates
        return CitationVerificationGateResult(
            is_valid=True,
            citation_text=citation_text,
            normalized_citation=node.normalized_citation,
            citation_type=node.citation_type.value,
            jurisdiction=node.jurisdiction,
            reporter_or_statute_family=node.reporter_or_statute,
            validation_status=ValidationStatus.VALID,
            node=node,
        )

    def verify_all(
        self,
        citations: list[str],
        expected_jurisdiction: str | None = None,
    ) -> list[CitationVerificationGateResult]:
        """Verify multiple citations, evaluating each against gate policies."""
        return [self.verify_citation(c, expected_jurisdiction) for c in citations]
