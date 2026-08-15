"""Storage / retrieval / update / consolidation / lifecycle for case memory.

Evidence and assessments are written into a case partition as they are produced.
Consolidation deduplicates evidence (keeping the most recent / highest-confidence
value per evidence type) and writes a compact summary. Lifecycle management purges
records past their TTL — useful for a deployment that wants case data to expire.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..types import Assessment, Evidence, EvidenceType
from .partitions import CasePartition

EVIDENCE = "evidence"
ASSESSMENT = "assessment"
SUMMARY = "summary"
CONSTRAINTS = "constraints"


@dataclass(frozen=True)
class StatutoryConstraintBlock:
    """Immutable block containing extracted factual bounds and statutory parameters.

    Injected into system-prompt headers prior to agent dispatch to prevent
    constraint degradation during memory summarization and context compaction.
    """

    household_size: int | None = None
    income_thresholds: dict[str, float] = field(default_factory=dict)
    gross_income_limit: float | None = None
    net_income_limit: float | None = None
    countable_income: float | None = None
    medical_offsets: dict[str, float] = field(default_factory=dict)
    medical_deductions: float | None = None
    appeal_deadlines: dict[str, Any] = field(default_factory=dict)
    jurisdictional_timebars: dict[str, Any] = field(default_factory=dict)
    statutory_bounds: dict[str, Any] = field(default_factory=dict)

    def to_system_header(self) -> str:
        lines = [
            "=== STATUTORY CONSTRAINT HEADER (IMMUTABLE BOUNDS) ===",
            f"Household Size: {self.household_size if self.household_size is not None else 'UNSPECIFIED'}",
        ]
        if self.gross_income_limit is not None:
            lines.append(f"Gross Income Limit: ${self.gross_income_limit:.2f}")
        if self.net_income_limit is not None:
            lines.append(f"Net Income Limit: ${self.net_income_limit:.2f}")
        if self.countable_income is not None:
            lines.append(f"Countable Income: ${self.countable_income:.2f}")
        if self.income_thresholds:
            lines.append(f"Income Thresholds: {json.dumps(self.income_thresholds, sort_keys=True)}")
        if self.medical_deductions is not None:
            lines.append(f"Medical Deductions: ${self.medical_deductions:.2f}")
        if self.medical_offsets:
            lines.append(f"Medical Expenditure Offsets: {json.dumps(self.medical_offsets, sort_keys=True)}")
        if self.appeal_deadlines:
            lines.append(f"Statutory Appeal Filing Deadlines: {json.dumps(self.appeal_deadlines, sort_keys=True)}")
        if self.jurisdictional_timebars:
            lines.append(f"Jurisdictional Timebars: {json.dumps(self.jurisdictional_timebars, sort_keys=True)}")
        if self.statutory_bounds:
            lines.append(f"Statutory Bounds: {json.dumps(self.statutory_bounds, sort_keys=True)}")
        lines.append("=== END STATUTORY CONSTRAINT HEADER ===")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "household_size": self.household_size,
            "income_thresholds": self.income_thresholds,
            "gross_income_limit": self.gross_income_limit,
            "net_income_limit": self.net_income_limit,
            "countable_income": self.countable_income,
            "medical_offsets": self.medical_offsets,
            "medical_deductions": self.medical_deductions,
            "appeal_deadlines": self.appeal_deadlines,
            "jurisdictional_timebars": self.jurisdictional_timebars,
            "statutory_bounds": self.statutory_bounds,
        }


def extract_statutory_constraints(history: list[Any]) -> StatutoryConstraintBlock:
    """Pre-compaction filter parsing and locking critical statutory constraints.

    Extracts household income thresholds, countable gross/net income limits,
    medical expenditure offsets, and statutory appeal filing deadlines from
    evidence, raw text messages, or assessment histories.
    """
    hh_size: int | None = None
    income_thresholds: dict[str, float] = {}
    gross_limit: float | None = None
    net_limit: float | None = None
    countable_income: float | None = None
    medical_offsets: dict[str, float] = {}
    medical_deductions: float | None = None
    appeal_deadlines: dict[str, Any] = {}
    jurisdictional_timebars: dict[str, Any] = {}
    statutory_bounds: dict[str, Any] = {}

    for item in history:
        # Case 1: Evidence object
        if isinstance(item, Evidence):
            etype = item.type
            if etype == EvidenceType.HOUSEHOLD_SIZE:
                try:
                    hh_size = int(float(item.value))
                except (ValueError, TypeError):
                    pass
            elif etype in (EvidenceType.MONTHLY_INCOME, EvidenceType.ANNUAL_INCOME, EvidenceType.BASE_PERIOD_EARNINGS):
                try:
                    val = float(item.value)
                    income_thresholds[etype.value] = val
                    if etype == EvidenceType.MONTHLY_INCOME:
                        countable_income = val
                except (ValueError, TypeError):
                    pass
            elif etype == EvidenceType.DAYS_SINCE_DENIAL:
                try:
                    appeal_deadlines["days_since_denial"] = int(float(item.value))
                except (ValueError, TypeError):
                    pass
            elif etype == EvidenceType.DENIAL_DATE:
                appeal_deadlines["denial_date"] = str(item.value)
            elif etype == EvidenceType.LIQUID_ASSETS:
                try:
                    statutory_bounds["liquid_assets"] = float(item.value)
                except (ValueError, TypeError):
                    pass

        # Case 2: Assessment object
        elif isinstance(item, Assessment):
            for crit in item.criteria:
                if "income" in crit.criterion_id or "gross" in crit.criterion_id or "net" in crit.criterion_id:
                    statutory_bounds[f"criterion_{crit.criterion_id}"] = crit.outcome.value

        # Case 3: Dict or raw string representation (e.g. message logs, turn dictionaries)
        else:
            text = str(item)
            # Household size extraction
            hh_m = re.search(r"(?:household_size|family_size|household)\s*[:=]\s*(\d+)", text, re.I)
            if hh_m:
                hh_size = int(hh_m.group(1))

            # Income thresholds
            for m in re.finditer(r"(?:gross_income|monthly_income|gross_limit|net_limit|income_threshold)\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                val = float(m.group(1).replace(",", ""))
                income_thresholds[k] = val
                if "gross" in k:
                    gross_limit = val
                elif "net" in k:
                    net_limit = val
                elif "monthly" in k:
                    countable_income = val

            # Medical deductions / offsets
            for m in re.finditer(r"(?:medical_deduction|medical_offset|medical_expense|deduction)\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                val = float(m.group(1).replace(",", ""))
                medical_offsets[k] = val
                medical_deductions = val

            # Appeal filing deadlines
            for m in re.finditer(r"(?:appeal_deadline|filing_deadline|timebar|days_to_appeal|days_since_denial)\s*[:=]\s*([0-9]+(?:\s*days)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                appeal_deadlines[k] = m.group(1).strip()

            # Jurisdictional timebars
            timebar_m = re.search(r"(\d+)[- ]day statutory (?:window|timebar|limit)", text, re.I)
            if timebar_m:
                jurisdictional_timebars["statutory_appeal_window_days"] = int(timebar_m.group(1))

    return StatutoryConstraintBlock(
        household_size=hh_size,
        income_thresholds=income_thresholds,
        gross_income_limit=gross_limit,
        net_income_limit=net_limit,
        countable_income=countable_income,
        medical_offsets=medical_offsets,
        medical_deductions=medical_deductions,
        appeal_deadlines=appeal_deadlines,
        jurisdictional_timebars=jurisdictional_timebars,
        statutory_bounds=statutory_bounds,
    )


class MemoryConsolidator:
    def __init__(self, partition: CasePartition) -> None:
        self.partition = partition

    # -- storage ------------------------------------------------------------ #

    def store_evidence(self, evidence: list[Evidence], ttl_s: float | None = None) -> None:
        for ev in evidence:
            self.partition.write(
                kind=EVIDENCE,
                key=ev.evidence_id,
                record_type="Evidence",
                payload=ev.model_dump(mode="json"),
                ttl_s=ttl_s,
            )

    def store_assessment(self, assessment: Assessment, ttl_s: float | None = None) -> None:
        self.partition.write(
            kind=ASSESSMENT,
            key=f"{assessment.program.value}:{assessment.attempt}",
            record_type="Assessment",
            payload=assessment.model_dump(mode="json"),
            ttl_s=ttl_s,
        )

    def store_constraints(self, constraints: StatutoryConstraintBlock, ttl_s: float | None = None) -> None:
        self.partition.write(
            kind=CONSTRAINTS,
            key="statutory_constraints",
            record_type="StatutoryConstraintBlock",
            payload=constraints.to_dict(),
            ttl_s=ttl_s,
        )

    # -- retrieval ---------------------------------------------------------- #

    def read_evidence(self) -> list[Evidence]:
        return [Evidence.model_validate(r.payload) for r in self.partition.read_all(EVIDENCE)]

    def read_assessments(self) -> list[Assessment]:
        return [Assessment.model_validate(r.payload) for r in self.partition.read_all(ASSESSMENT)]

    def extract_constraints(self) -> StatutoryConstraintBlock:
        records = self.partition.read_all(EVIDENCE)
        evidence = [Evidence.model_validate(r.payload) for r in records]
        assessments = self.read_assessments()
        constraints = extract_statutory_constraints(evidence + assessments)
        self.store_constraints(constraints)
        return constraints

    # -- consolidation & context compaction --------------------------------- #

    def consolidate_evidence(self) -> list[Evidence]:
        """Deduplicate by evidence type, preferring higher confidence then recency.

        Returns the consolidated set and writes a summary record. The records read
        back from the partition preserve insertion order, so a later, equally
        confident value supersedes an earlier one.
        """
        best: dict[str, Evidence] = {}
        for ev in self.read_evidence():
            key = ev.type.value
            current = best.get(key)
            if current is None or ev.confidence >= current.confidence:
                best[key] = ev
        consolidated = list(best.values())
        self.partition.write(
            kind=SUMMARY,
            key="evidence_summary",
            record_type="EvidenceSummary",
            payload={"types_present": sorted(best.keys()), "count": str(len(consolidated))},
        )
        # Lock statutory constraints upon consolidation
        self.extract_constraints()
        return consolidated

    def compact_context(self, history: list[Any]) -> tuple[StatutoryConstraintBlock, str]:
        """Compact lengthy interaction histories while guaranteeing statutory bounds retention.

        Extracts critical statutory constraints, produces a compacted summary,
        and prepends the immutable statutory constraint header.
        """
        constraints = extract_statutory_constraints(history)
        self.store_constraints(constraints)

        # Generate summary of facts and transitions
        summary_lines = [
            f"Compacted History: {len(history)} interaction turn(s) consolidated.",
            f"Active Evidence Count: {len(self.read_evidence())}",
        ]
        compacted_body = "\n".join(summary_lines)
        system_header = constraints.to_system_header()
        full_compacted = f"{system_header}\n\n{compacted_body}"
        return constraints, full_compacted

    # -- lifecycle ---------------------------------------------------------- #

    def expire(self) -> int:
        return self.partition.purge_expired()

