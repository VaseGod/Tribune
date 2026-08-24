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


def _doc_text(rule: Rule) -> str:
    return f"{rule.title}. {rule.description} {rule.text}"


class LocalRuleStore:
    """In-repo rule store backed by deterministic late-interaction retrieval and filterable HNSW indexing."""

    name = "local"

    def __init__(self) -> None:
        self._retriever = LateInteractionRetriever()
        self._hnsw_index = FilterableHNSWIndex(dim=96)
        self._rule_lookup: dict[str, Rule] = {}

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

    def retrieve(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:
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
        return program_registry.get_ruleset(program).required_ids

    def citation_for(
        self, program: ProgramId, jurisdiction: str, criterion_id: str
    ) -> Citation | None:
        rule = program_registry.get_ruleset(program).get(criterion_id)
        return rule.citation(program, jurisdiction) if rule else None

    def all_citations(self, program: ProgramId, jurisdiction: str) -> list[Citation]:
        return [r.citation(program, jurisdiction) for r in program_registry.get_ruleset(program).rules]

    def get_scoped_schema(self, program: ProgramId, jurisdiction: str) -> dict:
        """Extract schema and statutory criteria specifically for the target program, pruning others."""
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
    "make_rule_store",
    "ruleset_fingerprint",
]
