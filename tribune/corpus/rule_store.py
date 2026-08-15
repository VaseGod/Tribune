"""RuleStore: retrieve governing rules (with citations) for a program.

``LocalRuleStore`` is the default: it indexes the in-repo rule sets and ranks them
with the late-interaction retriever — a real ColBERT-style MaxSim search that runs
offline. ``HostedRuleStore`` is the extension point for a hosted vector backend
(e.g. a managed ColBERT/embedding service); it shares the same contract and falls
back to local if no endpoint is configured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import TribuneSettings, get_settings
from ..types import Citation, ProgramId
from . import programs as program_registry
from .citations import LateInteractionRetriever
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
    """In-repo rule store backed by deterministic late-interaction retrieval."""

    name = "local"

    def __init__(self) -> None:
        self._retriever = LateInteractionRetriever()
        for program in program_registry.all_programs():
            for rule in program_registry.get_ruleset(program).rules:
                self._retriever.index(f"{program.value}:{rule.criterion_id}", _doc_text(rule))

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
    """Pluggable hosted-vector-store backend.

    Subclasses ``LocalRuleStore`` so the rule metadata, citations, and required-
    criteria contracts are always available locally. Only the *ranking* would be
    delegated to a hosted vector/ColBERT service. With no endpoint configured it
    transparently behaves like the local store, which keeps the system runnable.
    """

    name = "hosted"

    def __init__(self, settings: TribuneSettings) -> None:
        super().__init__()
        self._endpoint = settings.hosted_vector_url
        self._collection = settings.hosted_vector_collection

    def retrieve(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:
        if not self._endpoint:
            # No hosted backend configured -> local late-interaction ranking.
            return super().retrieve(query, program, jurisdiction, k)
        return self._retrieve_hosted(query, program, jurisdiction, k)

    def _retrieve_hosted(
        self, query: str, program: ProgramId, jurisdiction: str, k: int
    ) -> list[RetrievedRule]:  # pragma: no cover - requires a live backend
        # Extension point: call the configured vector service, map returned doc ids
        # back to in-repo rules so citations and predicates stay authoritative.
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
    """Stable per-criterion hash of citation source + rule text.

    The canary compares this against a frozen baseline to detect rule-citation
    drift (a program's rules changed but the system's cited logic has not been
    updated — a live hazard for any benefits tool).
    """
    store = LocalRuleStore()
    fp: dict[str, str] = {}
    for citation in store.all_citations(program, jurisdiction):
        digest = hashlib.sha256(
            f"{citation.source}||{citation.text}||{citation.effective_date}".encode()
        ).hexdigest()
        fp[citation.citation_id] = digest
    return fp
