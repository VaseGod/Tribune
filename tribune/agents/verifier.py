"""Independent verifier.

The verifier re-derives the assessment from the *cited* rules rather than trusting
the proposer's criterion results. It checks three things and can reject -> REPLAN:

1. **Citation integrity** — every resolved criterion cites the rule that governs it.
2. **Coverage** — every required criterion of the program was actually assessed
   (catches the proposer concluding eligibility from a partial, retrieval-limited
   view; this is what drives the REPLAN path).
3. **Support** — the asserted status matches the status obtained by re-deriving
   over *all* required rules; no unsupported leaps.

A served deployment can run this on a stronger model; the provider's model-side
review is folded in alongside the structural checks.
"""

from __future__ import annotations

from ..corpus import programs as program_registry
from ..corpus.programs.jurisdictions import get_profile
from ..corpus.rule_store import RuleStore
from ..providers.base import ModelProvider, ReviewRequest, derive_status
from ..types import (
    Assessment,
    CriterionOutcome,
    CriterionResult,
    Evidence,
    EvidenceView,
    VerifierVerdict,
)


class ProgrammaticVerifierTools:
    """Typed Python stubs executed directly in-code by verifier agent loops."""

    @staticmethod
    def verify_citation_mapping(criterion_id: str, citation_id: str, rule_citations: list[str]) -> dict:
        """Verify that criterion citation matches expected rule citation."""
        is_valid = citation_id in rule_citations
        return {
            "criterion_id": criterion_id,
            "citation_id": citation_id,
            "is_valid": is_valid,
        }

    @staticmethod
    def rederive_status(criteria_outcomes: list[str]) -> dict:
        """Re-derive overall eligibility status from criterion outcome strings."""
        if any(o == "ineligible" for o in criteria_outcomes):
            status = "likely_ineligible"
        elif any(o == "unknown" for o in criteria_outcomes):
            status = "indeterminate"
        else:
            status = "likely_eligible"
        return {"rederived_status": status}

    @classmethod
    def get_tool_signatures(cls) -> str:
        """Expose typed Python signatures for model prompt generation."""
        return (
            "class ProgrammaticVerifierTools:\n"
            "    @staticmethod\n"
            "    def verify_citation_mapping(criterion_id: str, citation_id: str, rule_citations: list[str]) -> dict: ...\n"
            "    @staticmethod\n"
            "    def rederive_status(criteria_outcomes: list[str]) -> dict: ...\n"
        )


class Verifier:
    routing_intent: str = "multi_step_verification"
    target_engine: str = "Grok 4.6"

    def __init__(self, provider: ModelProvider, rule_store: RuleStore) -> None:
        self.provider = provider
        self.rule_store = rule_store
        self.tools = ProgrammaticVerifierTools()

    @staticmethod
    def parse_visible_response(text: str) -> str:
        """Parse only explicit, visible model text responses, ignoring unverified thinking monologues or block metadata."""
        if not isinstance(text, str):
            return text
        import re
        clean = re.sub(r"<(?:think|thought|reasoning)[^>]*>.*?</(?:think|thought|reasoning)>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()

    def generate_prompt(self, assessment: Assessment) -> str:
        """Generate prompt incorporating programmatic Python tool signatures."""
        return (
            f"You are the verifier for assessment '{assessment.assessment_id}'.\n"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticVerifierTools.get_tool_signatures()}\n"
            "Use these tools directly to check citation mappings and re-derive statuses."
        )

    def generate_self_testing_prompt(self, assessment: Assessment, jurisdiction: str) -> str:
        """Leverage Grok 4.6 capabilities to prompt for explicit multi-step self-testing trajectories."""
        return (
            f"You are Grok 4.6 performing independent verification for assessment '{assessment.assessment_id}' "
            f"under {jurisdiction} statutory rules.\n"
            "Execute the following explicit multi-step self-testing trajectory:\n"
            "1. Citation Integrity Check: Verify that all cited statutory rules exist, are active, and directly ground the claim.\n"
            "2. Predicate Re-derivation: Re-evaluate each rule predicate independently against claimant evidence.\n"
            "3. Global Coverage Assertion: Verify whether all statutory required criteria for the program are fully resolved.\n"
            "4. Cross-Statute Coherence: Cross-evaluate intermediate determinations to ensure absence of legal contradictions.\n\n"
            f"{ProgrammaticVerifierTools.get_tool_signatures()}\n"
            "Emit your trajectory outcome in strict structured JSON format."
        )

    def execute_self_testing_trajectory(
        self, assessment: Assessment, evidence: list[Evidence], jurisdiction: str
    ) -> tuple[float, list[dict]]:
        """Execute multi-step self-testing trajectory and cross-evaluate against statutory citation rules."""
        from ..corpus.citations import LateInteractionRetriever, cross_evaluate_citations

        program = assessment.program
        profile = get_profile(jurisdiction)
        view = EvidenceView(evidence)
        ruleset = program_registry.get_ruleset(program)
        active_citations = self.rule_store.all_citations(program, jurisdiction)
        active_cids = {c.citation_id for c in active_citations}

        steps: list[dict] = []

        # Milestone 1: Citation Mapping & Cross-Evaluation
        citation_valid = bool(assessment.citations) and all(c.citation_id in active_cids for c in assessment.citations)
        query_ctx = f"{program.value} eligibility verification {jurisdiction} {assessment.status.value}"
        cit_score = cross_evaluate_citations(assessment.citations, query_ctx) if assessment.citations else 0.0
        m1_score = (1.0 if citation_valid else 0.0) * 0.5 + cit_score * 0.5
        steps.append({
            "step": 1,
            "milestone": "Statutory Citation Mapping Verification",
            "passed": citation_valid and (cit_score > 0.0 or not assessment.citations),
            "score": round(m1_score, 4),
            "citations_evaluated": len(assessment.citations),
        })

        # Milestone 2: Evidence Predicate Consistency
        predicate_matches = 0
        total_eval = 0
        for crit in assessment.criteria:
            rule = ruleset.get(crit.criterion_id)
            if rule is not None:
                recomputed = rule.predicate(view, profile)
                total_eval += 1
                if recomputed is crit.outcome:
                    predicate_matches += 1
        m2_score = (predicate_matches / total_eval) if total_eval > 0 else 1.0
        steps.append({
            "step": 2,
            "milestone": "Evidence Predicate Consistency",
            "passed": m2_score == 1.0,
            "score": round(m2_score, 4),
            "matches": f"{predicate_matches}/{total_eval}",
        })

        # Milestone 3: Statutory Coverage Completeness
        assessed_req = {c.criterion_id for c in assessment.criteria if c.required}
        req_total = len(ruleset.required_ids)
        m3_score = (len(assessed_req) / req_total) if req_total > 0 else 1.0
        steps.append({
            "step": 3,
            "milestone": "Statutory Coverage Completeness",
            "passed": m3_score >= 1.0,
            "score": round(m3_score, 4),
            "covered": len(assessed_req),
            "required_total": req_total,
        })

        # Milestone 4: Cross-Statute Coherence & Re-derivation Check
        full_recompute = [
            CriterionResult(
                criterion_id=rule.criterion_id,
                description=rule.description,
                outcome=rule.predicate(view, profile),
                required=rule.required,
                citation_ids=[rule.citation(program, jurisdiction).citation_id],
            )
            for rule in ruleset.rules
        ]
        recomputed_status = derive_status(full_recompute, coverage_complete=True)
        status_match = (not assessment.is_assertion) or (recomputed_status is assessment.status)
        m4_score = 1.0 if status_match else 0.0
        steps.append({
            "step": 4,
            "milestone": "Cross-Statute Coherence",
            "passed": status_match,
            "score": m4_score,
            "recomputed_status": recomputed_status.value,
        })

        # Aggregate trajectory confidence
        weights = [0.25, 0.30, 0.20, 0.25]
        total_score = sum(w * s["score"] for w, s in zip(weights, steps))
        return round(total_score, 4), steps

    def verify(
        self, assessment: Assessment, evidence: list[Evidence], jurisdiction: str
    ) -> VerifierVerdict:
        program = assessment.program
        profile = get_profile(jurisdiction)
        view = EvidenceView(evidence)
        ruleset = program_registry.get_ruleset(program)

        missing_citations: list[str] = []
        unsupported_claims: list[str] = []
        reasons: list[str] = []

        # Mandatory citation verification against RuleStore
        active_citations = self.rule_store.all_citations(program, jurisdiction)
        active_cids = {c.citation_id for c in active_citations}
        for cit in assessment.citations:
            if cit.citation_id not in active_cids:
                missing_citations.append(f"invalid statutory citation '{cit.citation_id}'")

        # Re-derive every criterion the assessment took a position on, from its rule.
        recomputed_for_assessment: list[CriterionResult] = []
        for crit in assessment.criteria:
            rule = ruleset.get(crit.criterion_id)
            if rule is None:
                unsupported_claims.append(f"unknown criterion '{crit.criterion_id}'")
                continue
            expected_cid = rule.citation(program, jurisdiction).citation_id
            if crit.outcome is not CriterionOutcome.UNKNOWN:
                if not crit.citation_ids:
                    missing_citations.append(f"uncited claim for criterion '{crit.criterion_id}'")
                elif expected_cid not in crit.citation_ids or any(cid not in active_cids for cid in crit.citation_ids):
                    missing_citations.append(crit.criterion_id)
            recomputed = rule.predicate(view, profile)
            recomputed_for_assessment.append(
                CriterionResult(
                    criterion_id=crit.criterion_id,
                    description=crit.description,
                    outcome=recomputed,
                    required=crit.required,
                    citation_ids=[expected_cid],
                    evidence_ids=crit.evidence_ids,
                )
            )
            if recomputed is not crit.outcome:
                unsupported_claims.append(
                    f"criterion '{crit.criterion_id}': proposer said {crit.outcome.value}, "
                    f"re-derivation says {recomputed.value}"
                )

        # Coverage: were all required criteria assessed?
        assessed_required = {c.criterion_id for c in assessment.criteria if c.required}
        incomplete_coverage = [rid for rid in ruleset.required_ids if rid not in assessed_required]
        if incomplete_coverage:
            reasons.append(
                "assessment did not cover all required criteria: " + ", ".join(incomplete_coverage)
            )

        # Independent status from a full re-derivation over every required rule.
        full_recompute: list[CriterionResult] = []
        for rule in ruleset.rules:
            full_recompute.append(
                CriterionResult(
                    criterion_id=rule.criterion_id,
                    description=rule.description,
                    outcome=rule.predicate(view, profile),
                    required=rule.required,
                    citation_ids=[rule.citation(program, jurisdiction).citation_id],
                )
            )
        recomputed_status = derive_status(full_recompute, coverage_complete=True)
        if assessment.is_assertion and recomputed_status is not assessment.status:
            unsupported_claims.append(
                f"asserted status {assessment.status.value} is not supported by full "
                f"re-derivation ({recomputed_status.value})"
            )

        review = self.provider.review_assessment(
            ReviewRequest(
                assessment=assessment,
                recomputed=recomputed_for_assessment,
                citations=assessment.citations,
            )
        )

        # Execute multi-step self-testing trajectory
        self_testing_score, trajectory_steps = self.execute_self_testing_trajectory(
            assessment, evidence, jurisdiction
        )

        approved = (
            not missing_citations
            and not unsupported_claims
            and not incomplete_coverage
            and review.supported
        )
        if missing_citations:
            reasons.append("missing/incorrect citations for: " + ", ".join(missing_citations))
        for c in review.concerns:
            reasons.append(f"model review concern: {c}")
        if approved:
            reasons.append("independent re-derivation from the cited rules confirms the assessment")

        return VerifierVerdict(
            approved=approved,
            recomputed_status=recomputed_status,
            missing_citations=missing_citations,
            unsupported_claims=unsupported_claims,
            incomplete_coverage=incomplete_coverage,
            reasons=reasons,
            self_testing_score=self_testing_score,
            trajectory_steps=trajectory_steps,
        )
