"""RuleStore: retrieve governing rules (with citations) for a program.

Integrates:
1. ``LocalRuleStore``: In-repo rule store backed by both token-level late-interaction
   retrieval (ColBERT MaxSim) and payload-constrained ``FilterableHNSWIndex`` with
   explicit metadata attribute bridging edges.
2. ``HostedRuleStore``: Extension point for hosted vector backends with seamless local fallback.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..config import TribuneSettings, get_settings
from ..context.graph_builder import FilterableHNSWIndex
from ..types import Citation, ProgramId
from . import programs as program_registry
from .citations import LateInteractionRetriever, embed_dense
from .programs.base import Rule


@dataclass(frozen=True)
class RetrievedRule:
    rule: Rule
    citation: Citation
    score: float


@runtime_checkable
class RuleStore(Protocol):
    def retrieve(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]: ...

    def required_criteria(self, program: ProgramId) -> list[str]: ...

    def citation_for(
        self, program: ProgramId, jurisdiction: str, criterion_id: str
    ) -> Citation | None: ...

    def all_citations(self, program: ProgramId, jurisdiction: str) -> list[Citation]: ...

    def get_scoped_schema(self, program: ProgramId, jurisdiction: str) -> dict: ...

    def get_program_tools(self, program: ProgramId, jurisdiction: str) -> str: ...

    def lookup_engram(self, table_name: str, key: str | int, jurisdiction: str = "EX") -> Any: ...

    def inject_engram_facts(
        self, program: ProgramId, jurisdiction: str, evidence: list[Any] | None = None
    ) -> list[Any]: ...


def _doc_text(rule: Rule) -> str:
    return f"{rule.title}. {rule.description} {rule.text}"


class LocalRuleStore:
    """In-repo rule store backed by deterministic late-interaction retrieval and filterable HNSW indexing."""

    name = "local"

    def __init__(self) -> None:
        self._retriever = LateInteractionRetriever()
        self._hnsw_index = FilterableHNSWIndex(dim=96)
        self._rule_lookup: dict[str, Rule] = {}
        self.engram_store = StatutoryEngramRAMStore()
        self._cleared = False

        for program in program_registry.all_programs():
            ruleset = program_registry.get_ruleset(program)
            for rule in ruleset.rules:
                doc_key = f"{program.value}:{rule.criterion_id}"
                text = _doc_text(rule)
                self._rule_lookup[doc_key] = rule
                self._retriever.index(doc_key, text)

                # Index in filterable HNSW with rich statutory metadata
                dense_vec = embed_dense(text)
                self._hnsw_index.insert(
                    node_id=doc_key,
                    vector=dense_vec,
                    metadata={
                        "program": program.value,
                        "benefit_program": program.value,
                        "criterion_id": rule.criterion_id,
                        "required": rule.required,
                        "statutory_level": "federal" if "CFR" in rule.source or "USC" in rule.source else "state",
                        "effective_year": 2026,
                        "title": rule.title,
                    },
                )

    def clear(self) -> None:
        """Clear all indexed rules to serve as an unskilled/empty baseline store."""
        self._cleared = True
        self._rule_lookup.clear()
        self._retriever = LateInteractionRetriever()
        self._hnsw_index = FilterableHNSWIndex(dim=96)

    def retrieve(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:
        if self._cleared:
            return []
        rules = program_registry.get_ruleset(program).rules
        docs = {rule.criterion_id: _doc_text(rule) for rule in rules}
        scored = self._retriever.rank(query, docs, k=max(1, k))
        by_id = {rule.criterion_id: rule for rule in rules}
        out: list[RetrievedRule] = []
        for sd in scored:
            rule = by_id[sd.doc_id]
            out.append(
                RetrievedRule(
                    rule=rule,
                    citation=rule.citation(program, jurisdiction),
                    score=sd.score,
                )
            )
        return out

    def retrieve_filtered(
        self,
        query: str,
        program: ProgramId,
        jurisdiction: str,
        k: int = 8,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[RetrievedRule]:
        """Perform payload-constrained vector retrieval using FilterableHNSWIndex."""
        query_dense = embed_dense(query)

        def combined_filter(meta: dict[str, Any]) -> bool:
            if meta.get("program") != program.value and meta.get("benefit_program") != program.value:
                return False
            if filter_fn is not None and not filter_fn(meta):
                return False
            return True

        hits = self._hnsw_index.search(
            query_vector=query_dense,
            k=max(1, k),
            filter_fn=combined_filter,
        )

        rules = program_registry.get_ruleset(program).rules
        by_id = {rule.criterion_id: rule for rule in rules}
        out: list[RetrievedRule] = []

        for _node_id, score, meta in hits:
            crit_id = meta.get("criterion_id")
            if crit_id in by_id:
                rule = by_id[crit_id]
                out.append(
                    RetrievedRule(
                        rule=rule,
                        citation=rule.citation(program, jurisdiction),
                        score=score,
                    )
                )

        # Fallback to standard retrieve if HNSW hit list is empty
        if not out:
            return self.retrieve(query, program, jurisdiction, k)
        return out

    def required_criteria(self, program: ProgramId) -> list[str]:
        if self._cleared:
            return []
        return program_registry.get_ruleset(program).required_ids

    def citation_for(
        self, program: ProgramId, jurisdiction: str, criterion_id: str
    ) -> Citation | None:
        if self._cleared:
            return None
        rule = program_registry.get_ruleset(program).get(criterion_id)
        return rule.citation(program, jurisdiction) if rule else None

    def all_citations(self, program: ProgramId, jurisdiction: str) -> list[Citation]:
        if self._cleared:
            return []
        return [
            r.citation(program, jurisdiction) for r in program_registry.get_ruleset(program).rules
        ]

    def get_scoped_schema(self, program: ProgramId, jurisdiction: str) -> dict:
        """Extract schema and statutory criteria specifically for the target program, pruning others."""
        if self._cleared:
            return {
                "program": program.value,
                "jurisdiction": jurisdiction,
                "required_criteria": [],
                "rules": [],
                "citations": [],
            }
        ruleset = program_registry.get_ruleset(program)
        citations = self.all_citations(program, jurisdiction)
        return {
            "program": program.value,
            "jurisdiction": jurisdiction,
            "required_criteria": ruleset.required_ids,
            "rules": [
                {
                    "criterion_id": r.criterion_id,
                    "title": r.title,
                    "description": r.description,
                    "required": r.required,
                }
                for r in ruleset.rules
            ],
            "citations": [c.source for c in citations],
        }

    def get_program_tools(self, program: ProgramId, jurisdiction: str) -> str:
        """Generate typed Python signatures strictly scoped to the target program domain."""
        if program == ProgramId.SNAP:
            return (
                f"class Programmatic{program.value.capitalize()}Tools:\n"
                f"    @staticmethod\n"
                f"    def calculate_snap_gross_income(monthly_earned: float, monthly_unearned: float, household_size: int) -> dict: ...\n"
                f"    @staticmethod\n"
                f"    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
            )
        if program == ProgramId.MEDICAID:
            return (
                f"class Programmatic{program.value.capitalize()}Tools:\n"
                f"    @staticmethod\n"
                f"    def evaluate_medicaid_magi_income(monthly_income: float, household_size: int, pathway: str) -> dict: ...\n"
                f"    @staticmethod\n"
                f"    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
            )
        if program == ProgramId.HOUSING:
            return (
                f"class Programmatic{program.value.capitalize()}Tools:\n"
                f"    @staticmethod\n"
                f"    def evaluate_housing_ami_limit(annual_income: float, household_size: int, jurisdiction: str) -> dict: ...\n"
                f"    @staticmethod\n"
                f"    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
            )
        if program == ProgramId.UNEMPLOYMENT:
            return (
                f"class Programmatic{program.value.capitalize()}Tools:\n"
                f"    @staticmethod\n"
                f"    def evaluate_base_period_earnings(earnings: float, separation_reason: str) -> dict: ...\n"
                f"    @staticmethod\n"
                f"    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
            )
        if program == ProgramId.APPEALS:
            return (
                f"class Programmatic{program.value.capitalize()}Tools:\n"
                f"    @staticmethod\n"
                f"    def evaluate_appeal_timeliness(days_since_denial: int, statutory_window: int = 90) -> dict: ...\n"
                f"    @staticmethod\n"
                f"    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
            )
        return (
            "class ProgrammaticEligibilityTools:\n"
            "    @staticmethod\n"
            "    def evaluate_criterion(evidence_value: float | str | bool, statutory_threshold: float | str | bool, operator: str = '<=') -> dict: ...\n"
        )

    def lookup_engram(self, table_name: str, key: str | int, jurisdiction: str = "EX") -> Any:
        """Lookup deterministic numeric or formula value from in-memory Engram RAM table."""
        return self.engram_store.lookup(table_name, key, jurisdiction)

    def inject_engram_facts(
        self,
        program: ProgramId,
        jurisdiction: str,
        evidence: list[Any] | None = None,
    ) -> list[Any]:
        """Dynamically inject exact deterministic statutory values with provenance tracking."""
        from .provenance import FactInjectionRecord, make_injected_fact
        from ..types import EvidenceType

        injected: list[FactInjectionRecord] = []
        hh_size = 1
        if evidence:
            for ev in evidence:
                ev_type = getattr(ev, "type", ev.get("type") if isinstance(ev, dict) else None)
                ev_val = getattr(ev, "value", ev.get("value") if isinstance(ev, dict) else None)
                if (ev_type == EvidenceType.HOUSEHOLD_SIZE or str(ev_type) == "household_size") and ev_val is not None:
                    try:
                        hh_size = int(ev_val)
                    except (ValueError, TypeError):
                        pass

        matrix = self.engram_store.query_matrix(
            table_name=program.value,
            jurisdiction=jurisdiction,
            criteria={"household_size": hh_size},
        )

        cit_id = f"{program.value}:{jurisdiction}:statutory_engram"
        source = matrix.get("citation", f"Statutory Engram Table {program.value.upper()}")

        for k, v in matrix.items():
            if k != "citation":
                rec = make_injected_fact(
                    fact_key=f"{program.value}::{k}",
                    exact_value=v,
                    citation_id=cit_id,
                    statute_source=source,
                    notes=f"Injected for {program.value} (jurisdiction={jurisdiction}, hh={hh_size})",
                )
                injected.append(rec)

        return injected



class StatutoryEngramRAMStore:
    """In-memory deterministic statutory knowledge store.

    Extracts static numeric matrices, Federal Poverty Level (FPL) tables, state income limits,
    asset caps, and formula definitions out of prompt context into an in-memory indexed store.
    Enables zero-hallucination, sub-millisecond statutory table lookups.
    """

    def __init__(self) -> None:
        # Federal Poverty Level (annual basis, 48 contiguous states) for 2026
        self.fpl_tables = {
            "2026_annual": {
                1: 15650.0,
                2: 21150.0,
                3: 26650.0,
                4: 32150.0,
                5: 37650.0,
                6: 43150.0,
                7: 48650.0,
                8: 54150.0,
            },
            "2026_monthly": {
                1: 1304.17,
                2: 1762.50,
                3: 2220.83,
                4: 2679.17,
                5: 3137.50,
                6: 3595.83,
                7: 4054.17,
                8: 4512.50,
            },
        }

        # SNAP matrices
        self.snap_matrices = {
            "gross_income_fpl_pct": 1.30,  # 130% FPL
            "net_income_fpl_pct": 1.00,    # 100% FPL
            "standard_deduction": {
                1: 198.0, 2: 198.0, 3: 198.0, 4: 208.0, 5: 244.0, 6: 279.0, 7: 279.0, 8: 279.0,
            },
            "asset_limit_standard": 3000.0,
            "asset_limit_elderly_disabled": 4500.0,
            "bbce_asset_waiver": {"EX": True, "NX": False},
        }

        # Medicaid MAGI matrices (% of FPL)
        self.medicaid_magi_pct = {
            "expansion_adult": 1.38,       # 138% FPL
            "non_expansion_parent": 0.40,  # 40% FPL
            "pregnant_women": 2.13,        # 213% FPL
            "children": 2.55,              # 255% FPL
        }

        # Housing AMI factors (HUD 4-person base $100,000)
        self.housing_ami = {
            "ami_base_4person": 100000.0,
            "factors": {1: 0.70, 2: 0.80, 3: 0.90, 4: 1.00, 5: 1.08, 6: 1.16, 7: 1.24, 8: 1.32},
            "extremely_low_pct": 0.30,
            "very_low_pct": 0.50,
            "low_pct": 0.80,
        }

        # Unemployment Insurance thresholds
        self.ui_matrices = {
            "EX": {"min_base_earnings": 5000.0, "min_weeks_worked": 1, "max_wba": 600.0},
            "NX": {"min_base_earnings": 4500.0, "min_weeks_worked": 2, "max_wba": 450.0},
        }

        # Administrative Appeals windows
        self.appeals_deadlines = {
            "EX": {"window_days": 90, "good_cause_extension_days": 30},
            "NX": {"window_days": 60, "good_cause_extension_days": 15},
        }

    def lookup(self, table_name: str, key: str | int, jurisdiction: str = "EX") -> Any:
        """Lookup an exact deterministic numeric threshold or matrix value."""
        if table_name == "fpl_annual":
            hh = int(key)
            if hh in self.fpl_tables["2026_annual"]:
                return self.fpl_tables["2026_annual"][hh]
            return 15650.0 + (hh - 1) * 5500.0

        if table_name == "fpl_monthly":
            hh = int(key)
            if hh in self.fpl_tables["2026_monthly"]:
                return self.fpl_tables["2026_monthly"][hh]
            return round((15650.0 + (hh - 1) * 5500.0) / 12.0, 2)

        if table_name == "snap_gross_limit":
            hh = int(key)
            monthly_fpl = self.lookup("fpl_monthly", hh, jurisdiction)
            return round(monthly_fpl * self.snap_matrices["gross_income_fpl_pct"], 2)

        if table_name == "medicaid_magi_limit":
            pathway = str(key)
            pct = self.medicaid_magi_pct.get(pathway, 1.38)
            return pct

        if table_name == "housing_ami_limit":
            parts = str(key).split("_", 1)
            hh = int(parts[0]) if parts[0].isdigit() else 4
            tier = parts[1] if len(parts) > 1 else "very_low"
            factor = self.housing_ami["factors"].get(hh, 1.0 + 0.08 * (hh - 4))
            tier_pct = self.housing_ami.get(f"{tier}_pct", 0.50)
            return round(self.housing_ami["ami_base_4person"] * factor * tier_pct, 2)

        if table_name == "ui":
            jur_data = self.ui_matrices.get(jurisdiction, self.ui_matrices["EX"])
            return jur_data.get(str(key))

        if table_name == "appeals":
            jur_data = self.appeals_deadlines.get(jurisdiction, self.appeals_deadlines["EX"])
            return jur_data.get(str(key), 90)

        return None

    def query_matrix(
        self, table_name: str, jurisdiction: str = "EX", criteria: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Query complete statutory matrix for dynamic workspace fact injection."""
        criteria = criteria or {}
        hh = int(criteria.get("household_size", 1))
        fpl_mo = self.lookup("fpl_monthly", hh, jurisdiction)

        if table_name == "snap":
            return {
                "household_size": hh,
                "monthly_fpl": fpl_mo,
                "gross_income_limit": self.lookup("snap_gross_limit", hh, jurisdiction),
                "asset_limit": self.snap_matrices["asset_limit_standard"],
                "bbce_waives_assets": self.snap_matrices["bbce_asset_waiver"].get(jurisdiction, True),
                "standard_deduction": self.snap_matrices["standard_deduction"].get(hh, 198.0),
                "citation": "7 CFR 273.9(a)",
            }
        elif table_name == "medicaid":
            return {
                "household_size": hh,
                "monthly_fpl": fpl_mo,
                "adult_expansion_limit": round(fpl_mo * self.medicaid_magi_pct["expansion_adult"], 2),
                "pregnant_limit": round(fpl_mo * self.medicaid_magi_pct["pregnant_women"], 2),
                "child_limit": round(fpl_mo * self.medicaid_magi_pct["children"], 2),
                "citation": "42 CFR 435.119",
            }
        elif table_name == "housing":
            return {
                "household_size": hh,
                "extremely_low_limit": self.lookup("housing_ami_limit", f"{hh}_extremely_low", jurisdiction),
                "very_low_limit": self.lookup("housing_ami_limit", f"{hh}_very_low", jurisdiction),
                "low_limit": self.lookup("housing_ami_limit", f"{hh}_low", jurisdiction),
                "citation": "24 CFR 982.201",
            }
        elif table_name == "unemployment":
            ui_data = self.ui_matrices.get(jurisdiction, self.ui_matrices["EX"])
            return {
                "min_base_period_earnings": ui_data["min_base_earnings"],
                "min_weeks_worked": ui_data["min_weeks_worked"],
                "max_weekly_benefit": ui_data["max_wba"],
                "citation": "20 CFR 604.3",
            }
        elif table_name == "appeals":
            app_data = self.appeals_deadlines.get(jurisdiction, self.appeals_deadlines["EX"])
            return {
                "appeal_window_days": app_data["window_days"],
                "good_cause_extension_days": app_data["good_cause_extension_days"],
                "citation": "42 CFR 431.221",
            }
        return {}


class HostedRuleStore(LocalRuleStore):
    """Pluggable hosted-vector-store backend with seamless local fallback."""

    name = "hosted"

    def __init__(self, settings: TribuneSettings) -> None:
        super().__init__()
        self._endpoint = settings.hosted_vector_url
        self._collection = settings.hosted_vector_collection

    def retrieve(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:
        if not self._endpoint:
            return super().retrieve(query, program, jurisdiction, k)
        return self._retrieve_hosted(query, program, jurisdiction, k)

    def _retrieve_hosted(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:  # pragma: no cover - requires a live backend
        raise NotImplementedError(
            "Connect your hosted vector backend here. The local store is the working "
            "fallback; see HostedRuleStore docstring."
        )


def make_rule_store(settings: TribuneSettings | None = None) -> RuleStore:
    settings = settings or get_settings()
    if settings.rule_store == "hosted":
        return HostedRuleStore(settings)
    return LocalRuleStore()


def ruleset_fingerprint(program: ProgramId, jurisdiction: str) -> dict[str, str]:
    """Stable per-criterion hash of citation source + rule text."""
    store = LocalRuleStore()
    fp: dict[str, str] = {}
    for citation in store.all_citations(program, jurisdiction):
        digest = hashlib.sha256(
            f"{citation.source}||{citation.text}||{citation.effective_date}".encode()
        ).hexdigest()
        fp[citation.citation_id] = digest
    return fp


__all__ = [
    "RetrievedRule",
    "RuleStore",
    "LocalRuleStore",
    "HostedRuleStore",
    "StatutoryEngramRAMStore",
    "make_rule_store",
    "ruleset_fingerprint",
]

