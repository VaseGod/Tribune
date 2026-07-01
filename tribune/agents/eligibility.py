"""Eligibility proposer.

Derives a fully-cited :class:`Assessment` for one program. Eligibility is treated
as a *derivation*, never an oracle: retrieve the governing rules, evaluate each
rule's predicate against the documented evidence, and let the provider synthesize a
status, recommended action, and rationale from those cited criterion results. It
also computes the diagnostics the calibrator needs (coverage, resolved fraction,
margin-to-threshold, structural ambiguity signals).
"""

from __future__ import annotations

from ..abstention.calibration import AssessmentDiagnostics
from ..corpus import programs as program_registry
from ..corpus.programs.jurisdictions import get_profile
from ..corpus.rule_store import RuleStore
from ..providers.base import ModelProvider, SynthesisRequest
from ..types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    Evidence,
    EvidenceType,
    EvidenceView,
    ProgramId,
    WaitlistStatus,
)

_QUERY_TERMS = {
    ProgramId.SNAP: "SNAP food assistance eligibility income limit household residency citizenship assets",
    ProgramId.UNEMPLOYMENT: "unemployment insurance monetary earnings separation reason able available",
    ProgramId.MEDICAID: "medicaid health coverage MAGI income pathway residency citizenship expansion",
    ProgramId.HOUSING: "housing choice voucher income limit area median income citizenship residency",
    ProgramId.APPEALS: "appeal fair hearing timely request grounds deadline",
}


class EligibilityProposer:
    def __init__(self, provider: ModelProvider, rule_store: RuleStore) -> None:
        self.provider = provider
        self.rule_store = rule_store

    def _query(self, program: ProgramId, jurisdiction: str) -> str:
        return f"{_QUERY_TERMS[program]} {jurisdiction}"

    def assess(
        self,
        case_id: str,
        jurisdiction: str,
        program: ProgramId,
        evidence: list[Evidence],
        k: int,
        attempt: int,
    ) -> tuple[Assessment, AssessmentDiagnostics]:
        profile = get_profile(jurisdiction)
        view = EvidenceView(evidence)
        retrieved = self.rule_store.retrieve(self._query(program, jurisdiction), program, jurisdiction, k)

        criteria: list[CriterionResult] = []
        citations: list[Citation] = []
        seen: set[str] = set()
        margins: list[float] = []

        for rr in retrieved:
            rule = rr.rule
            citation = rr.citation
            if citation.citation_id not in seen:
                citations.append(citation)
                seen.add(citation.citation_id)
            outcome = rule.predicate(view, profile)
            if rule.margin_fn is not None:
                m = rule.margin_fn(view, profile)
                if m is not None:
                    margins.append(m)
            criteria.append(
                CriterionResult(
                    criterion_id=rule.criterion_id,
                    description=rule.description,
                    outcome=outcome,
                    required=rule.required,
                    citation_ids=[citation.citation_id],
                    evidence_ids=rule.evidence_ids(view),
                    note=rule.source,
                )
            )

        required_total = len(self.rule_store.required_criteria(program))
        evaluated_required = sum(1 for c in criteria if c.required)
        unknown_required = sum(
            1 for c in criteria if c.required and c.outcome is CriterionOutcome.UNKNOWN
        )
        resolved_required = evaluated_required - unknown_required
        coverage_complete = evaluated_required >= required_total

        synth = self.provider.synthesize_assessment(
            SynthesisRequest(
                program=program,
                jurisdiction=jurisdiction,
                criteria=criteria,
                required_total=required_total,
                coverage_complete=coverage_complete,
                evidence_summary=self._summary(evidence),
                citations=citations,
            )
        )

        waitlist = None
        if program is ProgramId.HOUSING:
            raw = view.text(EvidenceType.WAITLIST_STATUS)
            waitlist = (
                WaitlistStatus(raw) if raw in {e.value for e in WaitlistStatus} else WaitlistStatus.UNKNOWN
            )

        assessment = Assessment(
            assessment_id=f"{case_id}:{program.value}:a{attempt}",
            case_id=case_id,
            program=program,
            jurisdiction=jurisdiction,
            status=synth.status,
            criteria=criteria,
            citations=citations,
            evidence_ids=[e.evidence_id for e in evidence],
            recommended_action=synth.recommended_action,
            self_confidence=synth.self_confidence,
            rationale=synth.rationale,
            waitlist_status=waitlist,
            attempt=attempt,
        )

        diagnostics = AssessmentDiagnostics(
            required_total=required_total,
            evaluated_required=evaluated_required,
            unknown_required=unknown_required,
            coverage=(evaluated_required / required_total) if required_total else 0.0,
            resolved_fraction=(resolved_required / required_total) if required_total else 0.0,
            min_margin=min(margins) if margins else 1.0,
            ambiguity_signals=program_registry.get_ruleset(program).ambiguity_signals(view, profile),
        )
        return assessment, diagnostics

    @staticmethod
    def _summary(evidence: list[Evidence]) -> str:
        return ", ".join(f"{ev.type.value}={ev.value}" for ev in evidence[:12])
