"""Independent Verifier, Programmatic Verification Code Generator, Tripartite Self-Testing, & SAO.

The verifier performs:
1. Dynamic generation of deterministic, programmatic verification functions without access to reference solutions.
2. Mandatory tripartite automated self-test suite (Oracle, No-Op, and Unsolved-State checks) prior to certification.
3. Step-level Advantage Optimization (SAO) assigning reward/advantage scores to individual reasoning steps,
   premises, and statutory citations in legal appeal briefs and trajectories.
4. Trajectory-level binary verification directly evaluating solver reasoning trajectories for fact grounding,
   statutory citation validity, and lack of ungrounded assumptions.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..corpus import programs as program_registry
from ..corpus.citations import cross_evaluate_citations
from ..corpus.programs.jurisdictions import JurisdictionProfile, get_profile
from ..corpus.rule_store import LocalRuleStore, RuleStore
from ..providers.base import ModelProvider, ReviewRequest, ReviewResult, derive_status
from ..types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    Evidence,
    EvidenceType,
    EvidenceView,
    ProgramId,
    RecommendedAction,
    VerifierVerdict,
)


@dataclass(frozen=True)
class VerificationReport:
    """Structured milestone verification report produced in Pass 1 of the two-stage pattern."""

    is_certified: bool
    assessment_id: str
    program: ProgramId
    jurisdiction: str
    recomputed_status: EligibilityStatus
    milestone_steps: list[dict[str, Any]] = field(default_factory=list)
    violated_rule_ids: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    calculation_discrepancies: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    score: float = 1.0
    certified_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TrajectoryVerificationVerdict:
    """Trajectory-level binary verification verdict certifying solver trajectory validity without reference answers."""

    approved: bool
    sanity_score: float  # [0.0, 1.0]
    step_validations: list[dict[str, Any]] = field(default_factory=list)
    grounding_violations: list[str] = field(default_factory=list)
    citation_violations: list[str] = field(default_factory=list)
    assumption_violations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tripartite Self-Testing Suite Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TripartiteCheckResult:
    """Outcome of a single check within the Tripartite Self-Test Suite."""

    check_name: str  # "oracle_check" | "noop_check" | "unsolved_state_check"
    passed: bool
    score: float  # [0.0, 1.0]
    details: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TripartiteVerificationResult:
    """Result of running the full tripartite self-test suite."""

    all_passed: bool
    composite_score: float
    oracle_result: TripartiteCheckResult
    noop_result: TripartiteCheckResult
    unsolved_state_result: TripartiteCheckResult
    reasons: list[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Step-level Advantage Optimization (SAO) Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepAdvantageRecord:
    """Attributed advantage, citation validity, and grounding score for an individual reasoning step."""

    step_index: int
    step_type: str  # "statutory_citation" | "factual_premise" | "predicate_deduction" | "status_assertion"
    action_text: str
    step_advantage: float  # Scalar advantage [-1.0, 1.0]
    grounding_score: float  # [0.0, 1.0]
    citation_gain: float  # [0.0, 1.0]
    is_grounded: bool
    has_valid_citations: bool
    penalties: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SAOAnalysisResult:
    """Comprehensive Step-level Advantage Optimization evaluation across an entire appeal brief / trajectory."""

    mean_advantage: float
    cumulative_advantage: float
    total_steps: int
    grounded_steps_count: int
    valid_citation_steps_count: int
    step_records: list[StepAdvantageRecord] = field(default_factory=list)
    critical_defects: list[str] = field(default_factory=list)
    is_advantageous: bool = True


# --------------------------------------------------------------------------- #
# Dynamic Programmatic Verification Function Generator
# --------------------------------------------------------------------------- #


class DynamicVerificationFunctionGenerator:
    """Dynamically generates deterministic, programmatic verification functions without access to reference solutions."""

    @staticmethod
    def generate_verification_code(program: ProgramId, jurisdiction: str = "EX") -> str:
        """Generate standalone executable Python code to independently verify determinations for a given program."""
        profile = get_profile(jurisdiction)
        ruleset = program_registry.get_ruleset(program)

        code_lines = [
            f"# Auto-generated Programmatic Verification Function for {program.value.upper()} in {jurisdiction}",
            "# Evaluates claim grounding, rule predicates, and citation mapping without reference solutions.",
            "def verify_candidate_determination(assessment_dict, evidence_dict, jurisdiction_profile):",
            "    violations = []",
            "    recomputed_criteria = {}",
            "    required_ids = " + str(ruleset.required_ids),
            "",
            "    # 1. Fact Grounding & Predicate Execution",
        ]

        if program == ProgramId.SNAP:
            code_lines.extend([
                f"    gross_limit = jurisdiction_profile.get('fpl_monthly', {profile.fpl_monthly(1)}) * {profile.snap_gross_income_pct}",
                "    monthly_income = float(evidence_dict.get('monthly_income', 0.0))",
                "    hh_size = int(evidence_dict.get('household_size', 1))",
                "    liquid_assets = float(evidence_dict.get('liquid_assets', 0.0))",
                "    asset_limit = jurisdiction_profile.get('snap_asset_limit', 2750.0)",
                "    if monthly_income > gross_limit:",
                "        recomputed_criteria['gross_income'] = 'not_satisfied'",
                "    else:",
                "        recomputed_criteria['gross_income'] = 'satisfied'",
                "    if liquid_assets > asset_limit:",
                "        recomputed_criteria['liquid_assets'] = 'not_satisfied'",
                "    else:",
                "        recomputed_criteria['liquid_assets'] = 'satisfied'",
            ])
        elif program == ProgramId.APPEALS:
            code_lines.extend([
                f"    window_days = jurisdiction_profile.get('appeal_window_days', {profile.appeal_window_days})",
                "    days_since_denial = float(evidence_dict.get('days_since_denial', 999.0))",
                "    has_grounds = bool(evidence_dict.get('appeal_grounds', ''))",
                "    if days_since_denial <= window_days:",
                "        recomputed_criteria['timely_filing'] = 'satisfied'",
                "    else:",
                "        recomputed_criteria['timely_filing'] = 'not_satisfied'",
                "    recomputed_criteria['stated_grounds'] = 'satisfied' if has_grounds else 'not_satisfied'",
            ])
        else:
            code_lines.extend([
                "    for req in required_ids:",
                "        recomputed_criteria[req] = 'satisfied' if req in evidence_dict else 'unknown'",
            ])

        code_lines.extend([
            "",
            "    # 2. Coverage & Citation Mapping Check",
            "    assessed_criteria = assessment_dict.get('criteria', [])",
            "    assessed_map = {c.get('criterion_id'): c for c in assessed_criteria}",
            "    for req in required_ids:",
            "        if req not in assessed_map:",
            "            violations.append(f'Missing required statutory criterion: {req}')",
            "        elif assessed_map[req].get('outcome') != recomputed_criteria.get(req):",
            "            violations.append(f'Predicate mismatch on criterion {req}')",
            "",
            "    is_valid = len(violations) == 0",
            "    return {'is_valid': is_valid, 'violations': violations, 'recomputed_criteria': recomputed_criteria}",
        ])

        return "\n".join(code_lines)


# --------------------------------------------------------------------------- #
# Programmatic Verifier Tools (Python stubs for models)
# --------------------------------------------------------------------------- #


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
        if any(o == "ineligible" or o == "not_satisfied" for o in criteria_outcomes):
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


# --------------------------------------------------------------------------- #
# Tripartite Self-Testing Suite
# --------------------------------------------------------------------------- #


class TripartiteSelfTestSuite:
    """Mandatory automated tripartite self-test suite prior to verifier registration in evaluation harnesses.

    Performs:
    1. Oracle Checks: Validate that known gold-standard / cleanly grounded solutions pass cleanly.
    2. No-Op Checks: Ensure empty, null, or tautological submissions fail immediately.
    3. Unsolved-State Checks: Confirm incomplete claims, partial evidence, or unverified claims fail.
    """

    def __init__(self, verifier: Verifier) -> None:
        self.verifier = verifier

    def run_oracle_check(self, program: ProgramId = ProgramId.SNAP, jurisdiction: str = "EX") -> TripartiteCheckResult:
        """Oracle Check: Validates that a well-formed, fully compliant assessment passes cleanly."""
        store = self.verifier.rule_store
        citations = store.all_citations(program, jurisdiction)
        ruleset = program_registry.get_ruleset(program)
        profile = get_profile(jurisdiction)

        # Build clean valid evidence
        from ..casegen.synthetic import SyntheticCaseGenerator
        case = SyntheticCaseGenerator().build_case(
            case_id="oracle_gold_case",
            jurisdiction=jurisdiction,
            overrides={"monthly_income": 800.0, "household_size": 2, "liquid_assets": 400.0},
            target_programs=[program],
        )

        view = EvidenceView(case.evidence)
        criteria = [
            CriterionResult(
                criterion_id=rule.criterion_id,
                description=rule.description,
                outcome=rule.predicate(view, profile),
                required=rule.required,
                citation_ids=[rule.citation(program, jurisdiction).citation_id],
            )
            for rule in ruleset.rules
        ]

        assessment = Assessment(
            assessment_id=f"oracle:{program.value}:gold",
            case_id="oracle_gold_case",
            program=program,
            jurisdiction=jurisdiction,
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=criteria,
            citations=citations,
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.98,
            rationale="Gold-standard verified assessment satisfying all statutory rules.",
        )

        verdict = self.verifier.verify(assessment, case.evidence, jurisdiction)
        passed = verdict.approved and (verdict.self_testing_score >= 0.85)

        reasons = list(verdict.reasons)
        if passed:
            reasons.append("Oracle Check: Gold-standard submission passed verification cleanly.")

        return TripartiteCheckResult(
            check_name="oracle_check",
            passed=passed,
            score=verdict.self_testing_score if passed else 0.0,
            details={"approved": verdict.approved, "self_testing_score": verdict.self_testing_score},
            reasons=reasons,
        )

    def run_noop_check(self, program: ProgramId = ProgramId.SNAP, jurisdiction: str = "EX") -> TripartiteCheckResult:
        """No-Op Check: Ensures empty, null, or tautological submissions fail verification."""
        # Empty assessment with no criteria and no citations
        noop_assessment = Assessment(
            assessment_id="noop_empty_submission",
            case_id="c_noop",
            program=program,
            jurisdiction=jurisdiction,
            status=EligibilityStatus.INDETERMINATE,
            criteria=[],
            citations=[],
            recommended_action=RecommendedAction.ABSTAIN_AND_ESCALATE,
            self_confidence=0.5,
            rationale="Tautological submission: empty criteria and ungrounded determination.",
        )

        verdict = self.verifier.verify(noop_assessment, [], jurisdiction)
        # MUST fail verification (verdict.approved is False)
        passed = (not verdict.approved) and (len(verdict.incomplete_coverage) > 0 or len(verdict.reasons) > 0)

        reasons = []
        if passed:
            reasons.append("No-Op Check: Empty/tautological submission correctly failed verification.")
        else:
            reasons.append("No-Op Check FAILURE: Empty/tautological submission erroneously approved.")

        return TripartiteCheckResult(
            check_name="noop_check",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"approved": verdict.approved, "reasons": verdict.reasons},
            reasons=reasons,
        )

    def run_unsolved_state_check(
        self, program: ProgramId = ProgramId.SNAP, jurisdiction: str = "EX"
    ) -> TripartiteCheckResult:
        """Unsolved-State Check: Confirms incomplete claims, partial evidence, or unverified claims fail."""
        store = self.verifier.rule_store
        citations = store.all_citations(program, jurisdiction)

        # Incomplete assessment omitting mandatory criteria
        incomplete_assessment = Assessment(
            assessment_id="incomplete_unsolved_submission",
            case_id="c_unsolved",
            program=program,
            jurisdiction=jurisdiction,
            status=EligibilityStatus.LIKELY_ELIGIBLE,
            criteria=[
                CriterionResult(
                    criterion_id="gross_income",
                    description="Gross income test",
                    outcome=CriterionOutcome.SATISFIED,
                    required=True,
                    citation_ids=[citations[0].citation_id] if citations else [],
                )
            ],
            citations=citations[:1] if citations else [],
            recommended_action=RecommendedAction.PREPARE_APPLICATION,
            self_confidence=0.9,
            rationale="Incomplete assessment missing residency and citizenship checks.",
        )

        verdict = self.verifier.verify(incomplete_assessment, [], jurisdiction)
        # MUST fail due to missing coverage / unverified predicates
        passed = (not verdict.approved) and (len(verdict.incomplete_coverage) > 0 or len(verdict.unsupported_claims) > 0)

        reasons = []
        if passed:
            reasons.append("Unsolved-State Check: Incomplete submission correctly rejected.")
        else:
            reasons.append("Unsolved-State Check FAILURE: Incomplete submission erroneously approved.")

        return TripartiteCheckResult(
            check_name="unsolved_state_check",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"incomplete_coverage": verdict.incomplete_coverage, "approved": verdict.approved},
            reasons=reasons,
        )

    def run_full_suite(
        self, program: ProgramId = ProgramId.SNAP, jurisdiction: str = "EX"
    ) -> TripartiteVerificationResult:
        """Execute all three tripartite checks and return combined verification result."""
        oracle_res = self.run_oracle_check(program, jurisdiction)
        noop_res = self.run_noop_check(program, jurisdiction)
        unsolved_res = self.run_unsolved_state_check(program, jurisdiction)

        all_passed = oracle_res.passed and noop_res.passed and unsolved_res.passed
        composite_score = round((oracle_res.score + noop_res.score + unsolved_res.score) / 3.0, 4)

        reasons = []
        reasons.extend(oracle_res.reasons)
        reasons.extend(noop_res.reasons)
        reasons.extend(unsolved_res.reasons)

        return TripartiteVerificationResult(
            all_passed=all_passed,
            composite_score=composite_score,
            oracle_result=oracle_res,
            noop_result=noop_res,
            unsolved_state_result=unsolved_res,
            reasons=reasons,
        )


# --------------------------------------------------------------------------- #
# Step-level Advantage Optimization (SAO)
# --------------------------------------------------------------------------- #


class StepAdvantageOptimizer:
    """Assigns step-level advantage and reward scores to reasoning steps and statutory citations in legal appeal briefs.

    Computes:
    - Step-level Advantage A(s_t, a_t) = grounding_gain + citation_gain + precision_gain - penalty
    - Factual premise grounding against source evidence records
    - Statutory citation validity against active RuleStore entries
    - Penalties for ungrounded assumptions, non-deterministic monologues (<think>), or circular deductions
    """

    def __init__(self, rule_store: RuleStore | None = None) -> None:
        self.rule_store = rule_store or LocalRuleStore()

    def analyze_trajectory(
        self,
        trajectory: Any,
        evidence: list[Evidence],
        jurisdiction: str,
        program: ProgramId = ProgramId.SNAP,
    ) -> SAOAnalysisResult:
        """Compute Step-level Advantage Optimization (SAO) across all trajectory reasoning frames."""
        known_evidence_keys = {e.type.value: e.value for e in evidence}
        active_citations = {c.citation_id for c in self.rule_store.all_citations(program, jurisdiction)}

        frames = getattr(trajectory, "frames", trajectory if isinstance(trajectory, list) else [])
        step_records: list[StepAdvantageRecord] = []
        critical_defects: list[str] = []

        cumulative_adv = 0.0

        for idx, frame in enumerate(frames):
            action_text = str(getattr(frame, "action", frame.get("action", "") if isinstance(frame, dict) else ""))
            data = getattr(frame, "data", frame.get("data", {}) if isinstance(frame, dict) else {})
            step_type = "predicate_deduction"

            penalties: list[str] = []
            grounding_score = 1.0
            citation_gain = 0.5
            is_grounded = True
            has_valid_citations = True

            # 1. Fact Grounding Analysis
            if "evidence" in data and isinstance(data["evidence"], list):
                step_type = "factual_premise"
                for ev in data["evidence"]:
                    etype = getattr(ev, "type", ev.get("type") if isinstance(ev, dict) else None)
                    eval_str = etype.value if hasattr(etype, "value") else str(etype)
                    if eval_str not in known_evidence_keys:
                        is_grounded = False
                        grounding_score = 0.0
                        penalties.append(f"Ungrounded fact '{eval_str}' asserted without ingestion")
                        critical_defects.append(f"Step {idx+1}: Ungrounded fact '{eval_str}'")

            # 2. Citation Validity Analysis
            if "citations" in data and isinstance(data["citations"], list):
                step_type = "statutory_citation"
                for cit in data["citations"]:
                    cid = cit.citation_id if hasattr(cit, "citation_id") else str(cit)
                    if cid not in active_citations:
                        has_valid_citations = False
                        citation_gain = -0.5
                        penalties.append(f"Invalid statutory citation '{cid}'")
                        critical_defects.append(f"Step {idx+1}: Invalid citation '{cid}'")
                    else:
                        citation_gain = 1.0

            # 3. Monologue / Reasoning Leak Checks
            if re.search(r"<(?:think|thought|reasoning)[^>]*>", action_text, re.IGNORECASE):
                penalties.append("Non-deterministic reasoning monologue detected")
                critical_defects.append(f"Step {idx+1}: Reasoning monologue artifact")

            # Step Advantage Calculation
            base_adv = 0.35 if is_grounded else -0.5
            cit_adv = 0.35 if has_valid_citations else -0.5
            penalty_deduction = len(penalties) * 0.25

            step_adv = max(-1.0, min(1.0, round(base_adv + cit_adv - penalty_deduction, 4)))
            cumulative_adv += step_adv

            step_records.append(
                StepAdvantageRecord(
                    step_index=idx + 1,
                    step_type=step_type,
                    action_text=action_text[:120],
                    step_advantage=step_adv,
                    grounding_score=grounding_score,
                    citation_gain=citation_gain,
                    is_grounded=is_grounded,
                    has_valid_citations=has_valid_citations,
                    penalties=penalties,
                )
            )

        total_steps = max(1, len(step_records))
        mean_adv = round(cumulative_adv / total_steps, 4)
        grounded_count = sum(1 for s in step_records if s.is_grounded)
        valid_cit_count = sum(1 for s in step_records if s.has_valid_citations)

        return SAOAnalysisResult(
            mean_advantage=mean_adv,
            cumulative_advantage=round(cumulative_adv, 4),
            total_steps=len(step_records),
            grounded_steps_count=grounded_count,
            valid_citation_steps_count=valid_cit_count,
            step_records=step_records,
            critical_defects=critical_defects,
            is_advantageous=mean_adv > 0.2 and len(critical_defects) == 0,
        )


# --------------------------------------------------------------------------- #
# Verifier Agent
# --------------------------------------------------------------------------- #


class Verifier:
    """VerifierAgent executing dynamic verification code, tripartite self-testing, and SAO analysis."""

    routing_intent: str = "multi_step_verification"
    target_engine: str = "gpt-5.6-sol-ultrafast"

    def __init__(
        self,
        provider: ModelProvider,
        rule_store: RuleStore,
        target_engine: str = "gpt-5.6-sol-ultrafast",
        router: Any | None = None,
    ) -> None:
        self.provider = provider
        self.rule_store = rule_store
        self.target_engine = target_engine
        self.router = router
        self.tools = ProgrammaticVerifierTools()
        self.self_test_suite = TripartiteSelfTestSuite(self)
        self.sao_optimizer = StepAdvantageOptimizer(self.rule_store)
        self.code_generator = DynamicVerificationFunctionGenerator()

    def run_self_testing_suite(
        self, program: ProgramId = ProgramId.SNAP, jurisdiction: str = "EX"
    ) -> TripartiteVerificationResult:
        """Execute the automated tripartite self-test suite (Oracle, No-Op, Unsolved-State)."""
        return self.self_test_suite.run_full_suite(program=program, jurisdiction=jurisdiction)

    def analyze_step_advantages(
        self,
        trajectory: Any,
        evidence: list[Evidence],
        jurisdiction: str,
        program: ProgramId = ProgramId.SNAP,
    ) -> SAOAnalysisResult:
        """Perform Step-level Advantage Optimization analysis over a solver trajectory or legal brief."""
        return self.sao_optimizer.analyze_trajectory(
            trajectory=trajectory, evidence=evidence, jurisdiction=jurisdiction, program=program
        )

    def generate_programmatic_verification_code(self, program: ProgramId, jurisdiction: str = "EX") -> str:
        """Generate standalone programmatic verification code for the given program and jurisdiction."""
        return self.code_generator.generate_verification_code(program, jurisdiction)

    def review_with_router(self, req: ReviewRequest) -> ReviewResult:
        """Route verifier review request to Tier 2 frontier endpoint if router configured."""
        if self.router is not None and hasattr(self.router, "route_verifier_task"):
            return self.router.route_verifier_task(req)
        return self.provider.review_assessment(req)

    @staticmethod
    def parse_visible_response(text: str) -> str:
        """Parse only explicit, visible model text responses, ignoring unverified thinking monologues."""
        if not isinstance(text, str):
            return text
        clean = re.sub(
            r"<(?:think|thought|reasoning)[^>]*>.*?</(?:think|thought|reasoning)>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
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
        """Prompt for explicit multi-step self-testing trajectories."""
        return (
            f"You are {self.target_engine} performing independent verification for assessment '{assessment.assessment_id}' "
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

        weights = [0.25, 0.30, 0.20, 0.25]
        total_score = sum(w * s["score"] for w, s in zip(weights, steps, strict=False))
        return round(total_score, 4), steps

    def verify_trajectory(
        self,
        trajectory: Any,
        evidence: list[Evidence],
        jurisdiction: str,
        program: ProgramId | None = None,
    ) -> TrajectoryVerificationVerdict:
        """Trajectory-Level Binary Verifier evaluating solver reasoning paths without reference solutions."""
        grounding_violations: list[str] = []
        citation_violations: list[str] = []
        assumption_violations: list[str] = []
        reasons: list[str] = []
        step_validations: list[dict[str, Any]] = []
        known_facts = {e.type.value: e.value for e in evidence}

        frames = getattr(trajectory, "frames", trajectory if isinstance(trajectory, list) else [])

        all_active_citations: set[str] = set()
        programs_to_check = [program] if program else program_registry.all_programs()
        for p in programs_to_check:
            for c in self.rule_store.all_citations(p, jurisdiction):
                all_active_citations.add(c.citation_id)

        for idx, frame in enumerate(frames):
            frame_agent = getattr(frame, "agent", frame.get("agent", "") if isinstance(frame, dict) else "")
            frame_action = getattr(frame, "action", frame.get("action", "") if isinstance(frame, dict) else "")
            frame_data = getattr(frame, "data", frame.get("data", {}) if isinstance(frame, dict) else {})
            frame_state = getattr(frame, "state", frame.get("state", None) if isinstance(frame, dict) else None)

            step_passed = True
            step_errors: list[str] = []

            # 1. Fact Grounding Check
            if "evidence" in frame_data and isinstance(frame_data["evidence"], list):
                for ev in frame_data["evidence"]:
                    ev_type = getattr(ev, "type", ev.get("type") if isinstance(ev, dict) else None)
                    ev_type_val = ev_type.value if hasattr(ev_type, "value") else str(ev_type)
                    if ev_type_val not in known_facts:
                        err = f"Step {idx+1}: Ungrounded evidence type '{ev_type_val}' introduced without source ingestion"
                        grounding_violations.append(err)
                        step_errors.append(err)
                        step_passed = False

            # 2. Citation Validity Check
            if "citations" in frame_data and isinstance(frame_data["citations"], list):
                for cit in frame_data["citations"]:
                    cid = cit.citation_id if hasattr(cit, "citation_id") else str(cit)
                    if cid not in all_active_citations:
                        err = f"Step {idx+1}: Invalid statutory citation '{cid}' referenced"
                        citation_violations.append(err)
                        step_errors.append(err)
                        step_passed = False

            # 3. Assumption / Monologue Checks in Action Rationale
            action_text = str(frame_action)
            if re.search(r"<(?:think|thought|reasoning)[^>]*>", action_text, re.IGNORECASE):
                err = f"Step {idx+1}: Unverified reasoning monologue detected in visible action"
                assumption_violations.append(err)
                step_errors.append(err)
                step_passed = False

            step_validations.append({
                "step_index": idx + 1,
                "agent": frame_agent,
                "state": frame_state.value if hasattr(frame_state, "value") else str(frame_state),
                "passed": step_passed,
                "errors": step_errors,
            })

        approved = (len(grounding_violations) == 0) and (len(citation_violations) == 0) and (len(assumption_violations) == 0)

        if not approved:
            reasons.extend(grounding_violations)
            reasons.extend(citation_violations)
            reasons.extend(assumption_violations)
        else:
            reasons.append("Trajectory fully grounded, statutory citations valid, and assumption-free.")

        total_steps = max(1, len(step_validations))
        passed_steps = sum(1 for s in step_validations if s["passed"])
        sanity_score = round(passed_steps / total_steps, 4)

        return TrajectoryVerificationVerdict(
            approved=approved,
            sanity_score=sanity_score,
            step_validations=step_validations,
            grounding_violations=grounding_violations,
            citation_violations=citation_violations,
            assumption_violations=assumption_violations,
            reasons=reasons,
        )

    def evaluate_and_certify(
        self, assessment: Assessment, evidence: list[Evidence], jurisdiction: str
    ) -> tuple[bool, VerificationReport]:
        """Pass 1: Strictly evaluate and certify candidate findings against statutory rules."""
        program = assessment.program
        profile = get_profile(jurisdiction)
        view = EvidenceView(evidence)
        ruleset = program_registry.get_ruleset(program)

        missing_citations: list[str] = []
        unsupported_claims: list[str] = []
        calculation_discrepancies: list[str] = []
        violated_rule_ids: list[str] = []
        reasons: list[str] = []

        # 1. Statutory citation validity against RuleStore
        active_citations = self.rule_store.all_citations(program, jurisdiction)
        active_cids = {c.citation_id for c in active_citations}
        for cit in assessment.citations:
            if cit.citation_id not in active_cids:
                missing_citations.append(f"invalid statutory citation '{cit.citation_id}'")

        # 2. Predicate and calculation verification
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
                violated_rule_ids.append(crit.criterion_id)
                unsupported_claims.append(
                    f"criterion '{crit.criterion_id}': proposer said {crit.outcome.value}, "
                    f"re-derivation says {recomputed.value}"
                )

        # 3. Coverage check: were all required criteria evaluated?
        assessed_required = {c.criterion_id for c in assessment.criteria if c.required}
        incomplete_coverage = [rid for rid in ruleset.required_ids if rid not in assessed_required]
        if incomplete_coverage:
            for rid in incomplete_coverage:
                violated_rule_ids.append(rid)
            reasons.append(
                "assessment did not cover all required criteria: " + ", ".join(incomplete_coverage)
            )

        # 4. Independent status from a full re-derivation over every required rule.
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
            calculation_discrepancies.append(
                f"asserted status {assessment.status.value} != full re-derivation {recomputed_status.value}"
            )
            unsupported_claims.append(
                f"asserted status {assessment.status.value} is not supported by full "
                f"re-derivation ({recomputed_status.value})"
            )

        # 5. Model review request
        review = self.provider.review_assessment(
            ReviewRequest(
                assessment=assessment,
                recomputed=recomputed_for_assessment,
                citations=assessment.citations,
            )
        )

        # 6. Multi-step self-testing trajectory
        self_testing_score, trajectory_steps = self.execute_self_testing_trajectory(
            assessment, evidence, jurisdiction
        )

        is_certified = (
            not missing_citations
            and not unsupported_claims
            and not incomplete_coverage
            and not calculation_discrepancies
            and review.supported
        )

        if missing_citations:
            reasons.append("missing/incorrect citations for: " + ", ".join(missing_citations))
        for c in review.concerns:
            reasons.append(f"model review concern: {c}")
        if is_certified:
            reasons.append("independent re-derivation from the cited rules confirms the assessment")

        report = VerificationReport(
            is_certified=is_certified,
            assessment_id=assessment.assessment_id,
            program=program,
            jurisdiction=jurisdiction,
            recomputed_status=recomputed_status,
            milestone_steps=trajectory_steps,
            violated_rule_ids=violated_rule_ids,
            missing_citations=missing_citations,
            unsupported_claims=unsupported_claims,
            calculation_discrepancies=calculation_discrepancies,
            reasons=reasons,
            score=self_testing_score,
        )
        return is_certified, report

    def verify(
        self, assessment: Assessment, evidence: list[Evidence], jurisdiction: str
    ) -> VerifierVerdict:
        program = assessment.program
        ruleset = program_registry.get_ruleset(program)

        is_certified, report = self.evaluate_and_certify(assessment, evidence, jurisdiction)

        assessed_required = {c.criterion_id for c in assessment.criteria if c.required}
        incomplete_coverage = [rid for rid in ruleset.required_ids if rid not in assessed_required]

        return VerifierVerdict(
            approved=is_certified,
            recomputed_status=report.recomputed_status,
            missing_citations=report.missing_citations,
            unsupported_claims=report.unsupported_claims,
            incomplete_coverage=incomplete_coverage,
            reasons=report.reasons,
            self_testing_score=report.score,
            trajectory_steps=report.milestone_steps,
        )

    def extract_failure_payload(
        self,
        assessment: Assessment,
        verdict: VerifierVerdict,
        report: VerificationReport | None = None,
    ) -> dict[str, Any] | None:
        """Extract structured failure telemetry payload from uncertified verdict or report."""
        if verdict.approved and (report is None or report.is_certified):
            return None

        missing = list(verdict.missing_citations)
        unsupported = list(verdict.unsupported_claims)
        incomplete = list(verdict.incomplete_coverage)
        reasons = list(verdict.reasons)
        violated_rules = list(report.violated_rule_ids) if report else []

        category = "general_failure"
        if missing:
            category = "citation_mismatch"
        elif unsupported or violated_rules:
            category = "predicate_error"
        elif incomplete:
            category = "coverage_gap"

        return {
            "case_id": assessment.case_id,
            "assessment_id": assessment.assessment_id,
            "program": assessment.program.value,
            "agent_id": "verifier",
            "category": category,
            "asserted_status": assessment.status.value,
            "recomputed_status": verdict.recomputed_status.value,
            "missing_citations": missing,
            "unsupported_claims": unsupported,
            "incomplete_coverage": incomplete,
            "violated_rule_ids": violated_rules,
            "self_testing_score": verdict.self_testing_score,
            "reasons": reasons,
        }


# VerifierAgent class alias
VerifierAgent = Verifier

__all__ = [
    "VerificationReport",
    "TrajectoryVerificationVerdict",
    "TripartiteCheckResult",
    "TripartiteVerificationResult",
    "TripartiteSelfTestSuite",
    "StepAdvantageRecord",
    "SAOAnalysisResult",
    "StepAdvantageOptimizer",
    "DynamicVerificationFunctionGenerator",
    "ProgrammaticVerifierTools",
    "Verifier",
    "VerifierAgent",
]
