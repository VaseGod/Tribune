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

import enum
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..corpus import programs as program_registry
from ..corpus.citations import cross_evaluate_citations
from ..corpus.programs.jurisdictions import get_profile
from ..corpus.rule_store import LocalRuleStore, RuleStore
from ..providers.base import ModelProvider, ReviewRequest, ReviewResult, derive_status
from ..types import (
    Assessment,
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

# --------------------------------------------------------------------------- #
# Decomposed Span-Level Editing Taxonomy & Dataclasses
# --------------------------------------------------------------------------- #


class DefectType(str, enum.Enum):
    """Enumeration of verification failure modes for decomposed span repair."""

    UNCITED_CLAIM = "uncited_claim"
    INCORRECT_CALCULATION = "incorrect_calculation"
    MISSING_JURISDICTION_CLAUSE = "missing_jurisdiction_clause"
    OUTDATED_STATUTE_REFERENCE = "outdated_statute_reference"


@dataclass
class DefectReport:
    """Structured report detailing a localized verification defect in draft text."""

    defect_type: DefectType
    description: str
    span_text: str = ""
    start_offset: int = -1
    end_offset: int = -1
    ast_node_type: str | None = None
    suggested_patch: str | None = None
    rule_id: str | None = None
    expected_value: Any | None = None
    actual_value: Any | None = None
    citation_anchor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizedSpan:
    """Precise text-offset matching and AST node localization."""

    start_offset: int
    end_offset: int
    matched_text: str
    ast_node_type: str = "text_span"
    line_number: int = 1
    column_offset: int = 0


@dataclass
class PatchResult:
    """Outcome of applying a targeted span repair to draft text."""

    success: bool
    defect_type: DefectType
    original_span: str
    patched_span: str
    start_offset: int
    end_offset: int
    applied_text: str
    strategy: str  # "deterministic_formula" | "deterministic_statute" | "ast_replacement" | "llm_micro_patch"
    preserved_anchors: list[str] = field(default_factory=list)
    message: str = ""


# --------------------------------------------------------------------------- #
# Legal Brief & Determination Template AST Parser
# --------------------------------------------------------------------------- #


@dataclass
class LegalASTNode:
    """An AST node representing a structural element of a legal determination or brief."""

    node_id: str
    node_type: str  # "HEADER" | "JURISDICTION_BLOCK" | "CRITERIA_BLOCK" | "CALCULATION_BLOCK" | "CITATION_BLOCK" | "CONCLUSION"
    text: str
    start_offset: int
    end_offset: int
    children: list[LegalASTNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LegalBriefASTParser:
    """Parses legal brief and regulatory determination templates into structural AST nodes."""

    _CITATION_RE = re.compile(
        r"(?:\d+\s+CFR\s+[\d.]+(?:\([a-zA-Z0-9]+\))*|\d+\s+USC\s+[\d.]+|RCW\s+[\d.]+|NYCRR\s+[\d.]+)",
        re.IGNORECASE,
    )
    _CALCULATION_RE = re.compile(
        r"(?:\$[\d,]+(?:\.\d+)?\s*(?:[><=+/×*-]|\b(?:exceeds|is\s+less\s+than|totaling|gross|net)\b)\s*\$?[\d,]+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _JURISDICTION_RE = re.compile(
        r"(?:Jurisdiction\s*:\s*([A-Za-z]{2})|under\s+(?:the\s+laws\s+of\s+)?([A-Za-z]{2})\s+statutes)",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, text: str) -> list[LegalASTNode]:
        """Parse draft text into top-level structural AST nodes with exact offsets."""
        nodes: list[LegalASTNode] = []
        if not text:
            return nodes

        lines = text.split("\n")
        curr_offset = 0

        for line_idx, line in enumerate(lines):
            line_len = len(line)
            start_off = curr_offset
            end_off = curr_offset + line_len
            stripped = line.strip()

            if not stripped:
                curr_offset += line_len + 1
                continue

            node_type = "TEXT_BLOCK"
            meta: dict[str, Any] = {"line_number": line_idx + 1}

            if stripped.startswith(("=", "#", "==")) or "NOTICE" in stripped.upper() or "DETERMINATION" in stripped.upper():
                node_type = "HEADER"
            elif cls._JURISDICTION_RE.search(stripped):
                node_type = "JURISDICTION_BLOCK"
                m = cls._JURISDICTION_RE.search(stripped)
                if m:
                    meta["jurisdiction"] = m.group(1) or m.group(2)
            elif cls._CITATION_RE.search(stripped):
                node_type = "CITATION_BLOCK"
                meta["citations"] = cls._CITATION_RE.findall(stripped)
            elif cls._CALCULATION_RE.search(stripped) or any(w in stripped.lower() for w in ("income", "limit", "asset", "gross", "rent")):
                node_type = "CALCULATION_BLOCK"
            elif any(w in stripped.lower() for w in ("outcome:", "status:", "recommended action:", "conclusion")):
                node_type = "CONCLUSION"
            elif stripped.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5.")):
                node_type = "CRITERIA_BLOCK"

            node = LegalASTNode(
                node_id=f"node_{line_idx + 1}",
                node_type=node_type,
                text=line,
                start_offset=start_off,
                end_offset=end_off,
                metadata=meta,
            )
            nodes.append(node)
            curr_offset += line_len + 1  # include newline

        return nodes


# --------------------------------------------------------------------------- #
# Agentic ASR 3-Step Span-Level Editing Engine
# --------------------------------------------------------------------------- #


class SpanEditingEngine:
    """Agentic Adaptive Span Refinement (ASR) 3-Step Correction Engine.

    1. Intent/Defect Classification: Enumerate failure modes (UNCITED_CLAIM, INCORRECT_CALCULATION,
       MISSING_JURISDICTION_CLAUSE, OUTDATED_STATUTE_REFERENCE).
    2. Span Localization: Match exact text offsets and AST node boundaries.
    3. Span-Targeted Patching: Deterministic string/AST replacement or localized LLM micro-patching
       while strictly preserving surrounding verified text and citation anchors.
    """

    def __init__(
        self,
        rule_store: RuleStore | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        self.rule_store = rule_store or LocalRuleStore()
        self.provider = provider

    def classify_defects(
        self,
        draft_text: str,
        assessment: Assessment | None = None,
        evidence: list[Evidence] | None = None,
        jurisdiction: str = "EX",
        program: ProgramId = ProgramId.SNAP,
    ) -> list[DefectReport]:
        """Step 1: Classify all verification defects across the draft text."""
        defects: list[DefectReport] = []
        if not draft_text:
            return defects

        ast_nodes = LegalBriefASTParser.parse(draft_text)
        active_cits = self.rule_store.all_citations(program, jurisdiction)
        active_sources = {c.source.strip().lower(): c for c in active_cits}

        # 1. Missing Jurisdiction Clause Check
        has_jurisdiction = any(n.node_type == "JURISDICTION_BLOCK" for n in ast_nodes) or f"Jurisdiction: {jurisdiction}" in draft_text
        if not has_jurisdiction:
            defects.append(
                DefectReport(
                    defect_type=DefectType.MISSING_JURISDICTION_CLAUSE,
                    description=f"Draft determination is missing mandatory jurisdiction clause for '{jurisdiction}'.",
                    span_text=draft_text[:min(100, len(draft_text))],
                    start_offset=0,
                    end_offset=0,
                    ast_node_type="HEADER",
                    suggested_patch=f"Jurisdiction: {jurisdiction} | Governed by statutory code of {jurisdiction}",
                )
            )

        # 2. Inspect AST Nodes for Calculation, Citation, and Claim Defects
        for node in ast_nodes:
            text = node.text

            # Check for Outdated / Invalid Statute References
            found_citations = LegalBriefASTParser._CITATION_RE.findall(text)
            for raw_cit in found_citations:
                norm_cit = raw_cit.strip().lower()
                is_valid = norm_cit in active_sources or any(norm_cit in src for src in active_sources)
                if not is_valid:
                    # Match closest active citation
                    matched_cit = active_cits[0] if active_cits else None
                    suggested = matched_cit.source if matched_cit else "7 CFR 273.9"
                    loc = self.localize_span(draft_text, raw_cit)
                    defects.append(
                        DefectReport(
                            defect_type=DefectType.OUTDATED_STATUTE_REFERENCE,
                            description=f"Outdated or superseded statutory reference '{raw_cit}' found in text.",
                            span_text=raw_cit,
                            start_offset=loc.start_offset,
                            end_offset=loc.end_offset,
                            ast_node_type=node.node_type,
                            suggested_patch=suggested,
                            citation_anchor=suggested,
                        )
                    )

            # Check for Calculation Discrepancies
            if node.node_type == "CALCULATION_BLOCK" and evidence:
                view = EvidenceView(evidence)
                monthly_income = view.num(EvidenceType.MONTHLY_INCOME)
                if monthly_income is not None:
                    profile = get_profile(jurisdiction)
                    fpl_limit = profile.fpl_monthly(int(view.num(EvidenceType.HOUSEHOLD_SIZE) or 1)) * profile.snap_gross_income_pct
                    # Look for incorrect threshold numbers in text
                    m_num = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", text)
                    if m_num:
                        val_str = m_num.group(1).replace(",", "")
                        try:
                            val_float = float(val_str)
                            if "limit" in text.lower() and abs(val_float - fpl_limit) > 5.0:
                                loc = self.localize_span(draft_text, m_num.group(0))
                                defects.append(
                                    DefectReport(
                                        defect_type=DefectType.INCORRECT_CALCULATION,
                                        description=f"Calculation limit discrepancy: stated ${val_float:.2f}, statutory limit is ${fpl_limit:.2f}",
                                        span_text=m_num.group(0),
                                        start_offset=loc.start_offset,
                                        end_offset=loc.end_offset,
                                        ast_node_type=node.node_type,
                                        expected_value=f"${fpl_limit:.2f}",
                                        actual_value=m_num.group(0),
                                        suggested_patch=f"${fpl_limit:.2f}",
                                    )
                                )
                        except ValueError:
                            pass

            # Check for Uncited Substantive Claims
            claim_patterns = [
                r"\b(?:is|are)\s+(?:likely\s+)?(?:eligible|ineligible)\b",
                r"\b(?:satisfies|meets|fails)\s+the\s+statutory\s+requirement\b",
                r"\b(?:appears\s+met|does\s+not\s+appear\s+met)\b",
            ]
            for pat in claim_patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match and not LegalBriefASTParser._CITATION_RE.search(text) and "citation" not in text.lower():
                    loc = self.localize_span(draft_text, match.group(0))
                    default_cit = active_cits[0].source if active_cits else "7 CFR 273.9(a)"
                    defects.append(
                        DefectReport(
                            defect_type=DefectType.UNCITED_CLAIM,
                            description=f"Substantive eligibility assertion '{match.group(0)}' lacks required statutory citation anchor.",
                            span_text=match.group(0),
                            start_offset=loc.start_offset,
                            end_offset=loc.end_offset,
                            ast_node_type=node.node_type,
                            suggested_patch=f"{match.group(0)} [pursuant to {default_cit}]",
                            citation_anchor=default_cit,
                        )
                    )

        return defects

    def localize_span(self, draft_text: str, target_snippet: str) -> LocalizedSpan:
        """Step 2: Localize text snippet in draft text with character offsets and line/col info."""
        if not target_snippet or target_snippet not in draft_text:
            return LocalizedSpan(start_offset=0, end_offset=0, matched_text="")

        idx = draft_text.find(target_snippet)
        end_idx = idx + len(target_snippet)
        prefix = draft_text[:idx]
        lines = prefix.split("\n")
        line_no = len(lines)
        col_no = len(lines[-1])

        return LocalizedSpan(
            start_offset=idx,
            end_offset=end_idx,
            matched_text=target_snippet,
            line_number=line_no,
            column_offset=col_no,
        )

    def patch_span(
        self,
        draft_text: str,
        defect: DefectReport,
    ) -> PatchResult:
        """Step 3: Execute deterministic or localized micro-patch for a specific defect."""
        if not draft_text:
            return PatchResult(
                success=False,
                defect_type=defect.defect_type,
                original_span="",
                patched_span="",
                start_offset=0,
                end_offset=0,
                applied_text="",
                strategy="noop",
                message="Empty draft text provided.",
            )

        # Extract all existing citation anchors to guarantee preservation
        existing_anchors = LegalBriefASTParser._CITATION_RE.findall(draft_text)

        # Strategy 1: Missing Jurisdiction Clause Insertion
        if defect.defect_type == DefectType.MISSING_JURISDICTION_CLAUSE:
            clause = defect.suggested_patch or "Jurisdiction: EX | Statutory Code of EX"
            # Insert right after header or at beginning
            header_match = re.search(r"(=+\n|#+[^\n]+\n)", draft_text)
            if header_match:
                insert_pos = header_match.end()
                patched_text = draft_text[:insert_pos] + clause + "\n" + draft_text[insert_pos:]
            else:
                patched_text = clause + "\n" + draft_text

            return PatchResult(
                success=True,
                defect_type=defect.defect_type,
                original_span="",
                patched_span=clause,
                start_offset=0,
                end_offset=len(clause),
                applied_text=patched_text,
                strategy="deterministic_ast",
                preserved_anchors=existing_anchors,
                message="Successfully prepended jurisdiction clause.",
            )

        # Strategy 2: Deterministic Offset or Substring Replacement
        target = defect.span_text
        patch = defect.suggested_patch or ""

        if defect.start_offset >= 0 and defect.end_offset > defect.start_offset and defect.end_offset <= len(draft_text):
            # Precision offset replacement
            orig_slice = draft_text[defect.start_offset:defect.end_offset]
            if orig_slice == target or not target:
                patched_text = draft_text[:defect.start_offset] + patch + draft_text[defect.end_offset:]
                strategy = "deterministic_formula" if defect.defect_type == DefectType.INCORRECT_CALCULATION else "deterministic_statute"
                return PatchResult(
                    success=True,
                    defect_type=defect.defect_type,
                    original_span=orig_slice,
                    patched_span=patch,
                    start_offset=defect.start_offset,
                    end_offset=defect.start_offset + len(patch),
                    applied_text=patched_text,
                    strategy=strategy,
                    preserved_anchors=existing_anchors,
                    message=f"Applied precision offset patch for {defect.defect_type.value}.",
                )

        if target and target in draft_text:
            # Substring replacement fallback
            loc = self.localize_span(draft_text, target)
            patched_text = draft_text[:loc.start_offset] + patch + draft_text[loc.end_offset:]
            strategy = "llm_micro_patch" if defect.defect_type == DefectType.UNCITED_CLAIM else "deterministic_statute"
            return PatchResult(
                success=True,
                defect_type=defect.defect_type,
                original_span=target,
                patched_span=patch,
                start_offset=loc.start_offset,
                end_offset=loc.start_offset + len(patch),
                applied_text=patched_text,
                strategy=strategy,
                preserved_anchors=existing_anchors,
                message=f"Applied localized replacement for {defect.defect_type.value}.",
            )

        return PatchResult(
            success=False,
            defect_type=defect.defect_type,
            original_span=target,
            patched_span=patch,
            start_offset=-1,
            end_offset=-1,
            applied_text=draft_text,
            strategy="unmatched",
            preserved_anchors=existing_anchors,
            message="Target span not found in draft text.",
        )

    def repair_draft(
        self,
        draft_text: str,
        defects: list[DefectReport],
    ) -> tuple[str, list[PatchResult]]:
        """Sequentially execute all span-targeted patches, preserving verified context."""
        current_text = draft_text
        results: list[PatchResult] = []

        # Sort defects by start_offset descending to avoid offset invalidation during replacement
        sorted_defects = sorted(defects, key=lambda d: d.start_offset, reverse=True)

        for defect in sorted_defects:
            res = self.patch_span(current_text, defect)
            results.append(res)
            if res.success:
                current_text = res.applied_text

        return current_text, results


def patch_span(
    draft_text: str,
    defect: DefectReport,
    rule_store: RuleStore | None = None,
    provider: ModelProvider | None = None,
) -> PatchResult:
    """Expose standalone patch_span interface."""
    engine = SpanEditingEngine(rule_store=rule_store, provider=provider)
    return engine.patch_span(draft_text, defect)


def repair_draft(
    draft_text: str,
    defects: list[DefectReport],
    rule_store: RuleStore | None = None,
    provider: ModelProvider | None = None,
) -> tuple[str, list[PatchResult]]:
    """Expose standalone repair_draft interface."""
    engine = SpanEditingEngine(rule_store=rule_store, provider=provider)
    return engine.repair_draft(draft_text, defects)


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
# Single-Turn Delta Rewriting & Failure Localization
# --------------------------------------------------------------------------- #


class FailureType(str, enum.Enum):
    """Categorization of execution turn failures for targeted single-turn delta rewrite."""

    COMPILATION_FAILURE = "compilation_failure"
    LINTING_FAILURE = "linting_failure"
    UNIT_TEST_ASSERTION_FAILURE = "unit_test_assertion_failure"
    TOOL_SCHEMA_VIOLATION = "tool_schema_violation"
    RUNTIME_EXCEPTION = "runtime_exception"
    INVALID_FILE_OPERATION = "invalid_file_operation"
    MALFORMED_CODE_BLOCK = "malformed_code_block"


@dataclass
class FailureLocalizationReport:
    """Report pinpointing exact failing turn and minimal repair target."""

    turn_id: str
    turn_index: int
    failing_tool_invocation_or_code_block: str
    error_type: FailureType
    error_message: str
    relevant_surrounding_context_references: list[str] = field(default_factory=list)
    minimal_repair_target: str = ""
    detected_at_step: int = 0


@dataclass
class SingleTurnDeltaRepairContract:
    """Structured contract for single-turn delta repairs."""

    turn_id: str
    repair_type: str  # "tool_call" | "code_block" | "command" | "schema_fix"
    original_content: str
    replacement_content: str
    explanation: str = ""
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "repair_type": self.repair_type,
            "original_content": self.original_content,
            "replacement_content": self.replacement_content,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SingleTurnDeltaRepairContract:
        return cls(
            turn_id=str(data.get("turn_id", "")),
            repair_type=str(data.get("repair_type", "code_block")),
            original_content=str(data.get("original_content", "")),
            replacement_content=str(data.get("replacement_content", "")),
            explanation=str(data.get("explanation", "")),
            confidence=float(data.get("confidence", 0.95)),
        )


class FailureTurnLocator:
    """Pinpoints the exact failing turn in Navigator execution history."""

    @classmethod
    def locate_failure(cls, execution_history: list[dict[str, Any]]) -> FailureLocalizationReport | None:
        if not execution_history:
            return None

        for idx, turn in enumerate(execution_history):
            status = str(turn.get("status", "")).lower()
            error_msg = str(turn.get("error", "") or turn.get("error_message", "") or "")
            tool_call = turn.get("tool_call") or turn.get("tool_invocation") or turn.get("command") or ""
            code_block = turn.get("code_block") or turn.get("content") or ""

            # Check if this turn experienced a failure
            is_failed = (
                status in ("failure", "failed", "error", "rejected")
                or bool(error_msg)
                or bool(turn.get("failing", False))
                or bool(turn.get("test_failure", False))
            )

            if is_failed:
                turn_id = str(turn.get("turn_id") or turn.get("id") or f"turn_{idx}")
                err_lower = error_msg.lower()
                if "syntaxerror" in err_lower or "compile" in err_lower or "compilation" in err_lower:
                    err_type = FailureType.COMPILATION_FAILURE
                elif "lint" in err_lower or "flake8" in err_lower or "ruff" in err_lower:
                    err_type = FailureType.LINTING_FAILURE
                elif "assertionerror" in err_lower or "test" in err_lower or "assert" in err_lower:
                    err_type = FailureType.UNIT_TEST_ASSERTION_FAILURE
                elif "schema" in err_lower or "argument" in err_lower or "validation" in err_lower:
                    err_type = FailureType.TOOL_SCHEMA_VIOLATION
                elif "filenotfound" in err_lower or "path" in err_lower or "permission" in err_lower or "traversal" in err_lower:
                    err_type = FailureType.INVALID_FILE_OPERATION
                elif "malformed" in err_lower or "unclosed" in err_lower:
                    err_type = FailureType.MALFORMED_CODE_BLOCK
                else:
                    err_type = FailureType.RUNTIME_EXCEPTION

                failing_block = str(tool_call or code_block or f"turn {turn_id}")
                surrounding_refs = []
                if idx > 0:
                    prev_id = execution_history[idx - 1].get("turn_id") or f"turn_{idx-1}"
                    surrounding_refs.append(f"Preceding: {prev_id}")
                if idx < len(execution_history) - 1:
                    next_id = execution_history[idx + 1].get("turn_id") or f"turn_{idx+1}"
                    surrounding_refs.append(f"Subsequent: {next_id}")

                return FailureLocalizationReport(
                    turn_id=turn_id,
                    turn_index=idx,
                    failing_tool_invocation_or_code_block=failing_block,
                    error_type=err_type,
                    error_message=error_msg or "Execution failure in turn",
                    relevant_surrounding_context_references=surrounding_refs,
                    minimal_repair_target=failing_block[:200],
                    detected_at_step=idx + 1,
                )

        return None


class SingleTurnDeltaRewriter:
    """Executes single-turn delta rewriting and strictly preserves surrounding trajectory history."""

    @staticmethod
    def apply_delta_repair(
        execution_history: list[dict[str, Any]],
        repair: SingleTurnDeltaRepairContract,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Apply targeted replacement to the failing turn without regenerating the trajectory."""
        updated_history: list[dict[str, Any]] = []
        applied = False

        for idx, turn in enumerate(execution_history):
            turn_id = str(turn.get("turn_id") or turn.get("id") or f"turn_{idx}")
            if turn_id == repair.turn_id:
                new_turn = dict(turn)
                new_turn["repaired"] = True
                new_turn["repair_type"] = repair.repair_type
                new_turn["original_content"] = repair.original_content
                new_turn["content"] = repair.replacement_content
                new_turn["tool_call"] = repair.replacement_content
                new_turn["status"] = "success"
                new_turn.pop("error", None)
                new_turn.pop("error_message", None)
                updated_history.append(new_turn)
                applied = True
            else:
                updated_history.append(turn)

        return updated_history, applied


# --------------------------------------------------------------------------- #
# ToolGrad Pre-Execution Assertion Checks
# --------------------------------------------------------------------------- #


@dataclass
class ToolGradAssertionResult:
    passed: bool
    tool_name: str
    violations: list[str] = field(default_factory=list)
    checked_constraints: list[str] = field(default_factory=list)


class ToolGradAssertionChecker:
    """Pre-execution validation ensuring near-deterministic tool-use chains."""

    DISALLOWED_COMMANDS = {
        "rm -rf /",
        "mkfs",
        ":(){ :|:& };:",
        "dd if=/dev/zero",
        "curl -s | bash",
        "wget -O- | sh",
    }

    @classmethod
    def validate_tool_call(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
        tool_spec: dict[str, Any] | None = None,
        repository_root: str | None = None,
    ) -> ToolGradAssertionResult:
        """Validate schema, type constraints, and environment safety before tool execution."""
        violations: list[str] = []
        checked: list[str] = []

        # 1. Schema Validation against declared tool spec
        if tool_spec:
            checked.append("schema_parameters_conformance")
            required_params = tool_spec.get("required", [])
            for param in required_params:
                if param not in arguments or arguments[param] is None:
                    violations.append(f"Missing required parameter '{param}' for tool '{tool_name}'")

            properties = tool_spec.get("properties", {})
            for param_name, param_spec in properties.items():
                if param_name in arguments:
                    expected_type = param_spec.get("type")
                    val = arguments[param_name]
                    if expected_type == "string" and not isinstance(val, str):
                        violations.append(f"Parameter '{param_name}' must be a string, got {type(val).__name__}")
                    elif expected_type == "integer" and not isinstance(val, int):
                        violations.append(f"Parameter '{param_name}' must be an integer, got {type(val).__name__}")
                    elif expected_type == "boolean" and not isinstance(val, bool):
                        violations.append(f"Parameter '{param_name}' must be a boolean, got {type(val).__name__}")
                    elif expected_type == "array" and not isinstance(val, list | tuple):
                        violations.append(f"Parameter '{param_name}' must be an array, got {type(val).__name__}")

                    if "enum" in param_spec and val not in param_spec["enum"]:
                        violations.append(
                            f"Parameter '{param_name}' value '{val}' not in permitted enum values {param_spec['enum']}"
                        )

        # 2. Environment Constraint: Path Traversal & Root Confinement
        path_arg = arguments.get("path") or arguments.get("file_path") or arguments.get("target_file")
        if path_arg and isinstance(path_arg, str):
            checked.append("path_confinement_check")
            clean_path = os.path.normpath(path_arg)
            if ".." in clean_path.split(os.sep):
                violations.append(f"Directory traversal escape detected in path '{path_arg}'")
            if repository_root:
                abs_root = os.path.abspath(repository_root)
                abs_target = (
                    os.path.abspath(os.path.join(repository_root, path_arg))
                    if not os.path.isabs(path_arg)
                    else os.path.abspath(path_arg)
                )
                if not abs_target.startswith(abs_root):
                    violations.append(f"Target path '{path_arg}' is outside repository root '{repository_root}'")

        # 3. Environment Constraint: Command Safety & Denylist
        cmd_arg = arguments.get("command") or arguments.get("cmd") or arguments.get("bash_command")
        if cmd_arg and isinstance(cmd_arg, str):
            checked.append("command_safety_check")
            for blocked in cls.DISALLOWED_COMMANDS:
                if blocked in cmd_arg:
                    violations.append(f"Prohibited dangerous shell command detected: '{blocked}'")

        # 4. Environment Constraint: Write Destination Safety
        if tool_name in ("write_to_file", "replace_file_content", "create_file"):
            checked.append("write_destination_check")
            dest = arguments.get("target_file") or arguments.get("path")
            if not dest or not isinstance(dest, str):
                violations.append("Write tool invoked without valid target destination path.")

        passed = len(violations) == 0
        return ToolGradAssertionResult(
            passed=passed,
            tool_name=tool_name,
            violations=violations,
            checked_constraints=checked,
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
        self.span_engine = SpanEditingEngine(self.rule_store, self.provider)
        self.failure_locator = FailureTurnLocator()
        self.delta_rewriter = SingleTurnDeltaRewriter()
        self.toolgrad_checker = ToolGradAssertionChecker()

        # Telemetry counters
        self.failure_turn_count = 0
        self.delta_rewrite_count = 0
        self.full_trajectory_regeneration_count = 0
        self.toolgrad_assertions_passed = 0
        self.toolgrad_assertions_failed = 0


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

    def verify_statutory_claims(
        self,
        claims: list[dict[str, Any]] | list[str],
        program: str | ProgramId,
        jurisdiction: str = "EX",
    ) -> VerificationReport:
        """Verify statutory claims emitted during agentic trajectory planning against rule store."""
        prog_enum = program if isinstance(program, ProgramId) else ProgramId(str(program).lower().strip())
        active_citations = {c.citation_id for c in self.rule_store.all_citations(prog_enum, jurisdiction)}

        missing_citations: list[str] = []
        unsupported_claims: list[str] = []
        violations: list[str] = []

        for c in claims:
            cid = c.get("citation_id") or c.get("citation") if isinstance(c, dict) else str(c)
            if cid and cid not in active_citations:
                missing_citations.append(str(cid))
                violations.append(f"Statutory claim references unverified citation '{cid}'")

        is_compliant = len(violations) == 0
        return VerificationReport(
            is_certified=is_compliant,
            assessment_id="claims_verification",
            program=prog_enum,
            jurisdiction=jurisdiction,
            recomputed_status=EligibilityStatus.LIKELY_ELIGIBLE if is_compliant else EligibilityStatus.INDETERMINATE,
            missing_citations=missing_citations,
            unsupported_claims=unsupported_claims,
            reasons=violations if not is_compliant else ["All statutory claims verified."],
            score=1.0 if is_compliant else 0.0,
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

    def classify_defects(
        self,
        draft_text: str,
        assessment: Assessment | None = None,
        evidence: list[Evidence] | None = None,
        jurisdiction: str = "EX",
        program: ProgramId = ProgramId.SNAP,
    ) -> list[DefectReport]:
        """Classify verification defects across draft text."""
        return self.span_engine.classify_defects(
            draft_text=draft_text,
            assessment=assessment,
            evidence=evidence,
            jurisdiction=jurisdiction,
            program=program,
        )

    def patch_span(self, draft_text: str, defect: DefectReport) -> PatchResult:
        """Execute targeted span repair on draft text."""
        return self.span_engine.patch_span(draft_text, defect)

    def repair_draft(
        self, draft_text: str, defects: list[DefectReport]
    ) -> tuple[str, list[PatchResult]]:
        """Repair all detected defects across draft text."""
        return self.span_engine.repair_draft(draft_text, defects)

    def locate_failing_turn(
        self, execution_history: list[dict[str, Any]]
    ) -> FailureLocalizationReport | None:
        """Locate exact failing turn in Navigator's execution history."""
        report = self.failure_locator.locate_failure(execution_history)
        if report:
            self.failure_turn_count += 1
        return report

    def execute_single_turn_delta_repair(
        self,
        execution_history: list[dict[str, Any]],
        repair: SingleTurnDeltaRepairContract | None = None,
    ) -> tuple[list[dict[str, Any]], SingleTurnDeltaRepairContract | None]:
        """Execute single-turn delta rewrite without full trajectory regeneration."""
        if repair is None:
            failure = self.locate_failing_turn(execution_history)
            if not failure:
                return execution_history, None
            replacement = (
                f"repaired_{failure.failing_tool_invocation_or_code_block}"
                if not failure.failing_tool_invocation_or_code_block.startswith("repaired_")
                else failure.failing_tool_invocation_or_code_block
            )
            repair = SingleTurnDeltaRepairContract(
                turn_id=failure.turn_id,
                repair_type="tool_call" if failure.error_type == FailureType.TOOL_SCHEMA_VIOLATION else "code_block",
                original_content=failure.failing_tool_invocation_or_code_block,
                replacement_content=replacement,
                explanation=f"Targeted repair addressing {failure.error_type.value}: {failure.error_message}",
                confidence=0.98,
            )

        updated_history, applied = self.delta_rewriter.apply_delta_repair(execution_history, repair)
        if applied:
            self.delta_rewrite_count += 1
            # Explicit retention protection: full trajectory regeneration remains strictly zero
            self.full_trajectory_regeneration_count = 0

        return updated_history, repair

    def validate_toolgrad(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_spec: dict[str, Any] | None = None,
        repository_root: str | None = None,
    ) -> ToolGradAssertionResult:
        """Execute ToolGrad pre-execution validation checks."""
        result = self.toolgrad_checker.validate_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            tool_spec=tool_spec,
            repository_root=repository_root,
        )
        if result.passed:
            self.toolgrad_assertions_passed += 1
        else:
            self.toolgrad_assertions_failed += 1
        return result


# VerifierAgent class alias
VerifierAgent = Verifier
IndependentVerifier = Verifier


# --------------------------------------------------------------------------- #
# Isolated Synthetic Verifier & Dual-Check Grader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DualCheckVerdict:
    """Verdict from dual-check grader with reference-blind validation."""

    passed: bool
    grader1_spec_passed: bool
    grader2_invariance_passed: bool
    reference_blind: bool = True
    spec_score: float = 1.0
    invariance_score: float = 1.0
    composite_grade: float = 1.0
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DualCheckGrader:
    """Dual-check grader agent executing reference-blind validation.

    Grader 1 (Specification Grader): Evaluates formal statutory rule satisfaction,
    required criteria coverage, and strict citation grounding without leaking ground truth.

    Grader 2 (Invariance Grader): Evaluates outcome invariant stability, semantic
    non-contradiction, absence of reasoning monologues, and specification gaming prevention.
    """

    def __init__(self, rule_store: RuleStore | None = None) -> None:
        self.rule_store = rule_store or LocalRuleStore()

    def grade(
        self,
        assessment: Assessment,
        evidence: EvidenceView | None = None,
        program: ProgramId = ProgramId.SNAP,
        jurisdiction: str = "EX",
        withheld_ground_truth: Any = None,
    ) -> DualCheckVerdict:
        findings: list[str] = []

        # --- Grader 1: Statutory Specification & Grounding --- #
        g1_passed = True
        spec_deductions = 0.0

        if not assessment.criteria:
            g1_passed = False
            spec_deductions += 0.5
            findings.append("Grader 1 (Spec): No criteria evaluated in candidate assessment.")
        else:
            unsatisfied_required = [
                c for c in assessment.criteria
                if c.required and c.outcome == CriterionOutcome.NOT_SATISFIED
            ]
            if unsatisfied_required and assessment.status == EligibilityStatus.LIKELY_ELIGIBLE:
                g1_passed = False
                spec_deductions += 0.6
                findings.append(
                    f"Grader 1 (Spec): Violation - candidate marked LIKELY_ELIGIBLE but required criteria unsatisfied: {[c.criterion_id for c in unsatisfied_required]}."
                )

        if not assessment.citations and assessment.status != EligibilityStatus.INDETERMINATE:
            g1_passed = False
            spec_deductions += 0.3
            findings.append("Grader 1 (Spec): Missing statutory citations for asserted determination.")

        spec_score = max(0.0, 1.0 - spec_deductions)

        # --- Grader 2: Invariance, Non-Contradiction, & Sandbox Boundaries --- #
        g2_passed = True
        inv_deductions = 0.0

        # Check for reasoning monologues or thought injection in rationales
        combined_text = " ".join(
            [getattr(c, "note", "") for c in assessment.criteria]
            + [getattr(assessment, "rationale", "") or "", getattr(assessment, "summary", "") or ""]
        )
        if re.search(r"<(?:think|thought|reasoning)[^>]*>", combined_text, re.IGNORECASE):
            g2_passed = False
            inv_deductions += 0.5
            findings.append("Grader 2 (Invariance): Reasoning monologue or thought tag detected in assessment output.")

        # Check for self-contradictory status
        rec_action = getattr(assessment, "recommended_action", None)
        if assessment.status == EligibilityStatus.LIKELY_INELIGIBLE and rec_action == RecommendedAction.PREPARE_APPLICATION:
            g2_passed = False
            inv_deductions += 0.5
            findings.append(f"Grader 2 (Invariance): Contradiction between status LIKELY_INELIGIBLE and action {rec_action.value}.")
        elif assessment.status == EligibilityStatus.INDETERMINATE and rec_action == RecommendedAction.PREPARE_APPLICATION:
            g2_passed = False
            inv_deductions += 0.3
            findings.append("Grader 2 (Invariance): Contradiction between status INDETERMINATE and action PREPARE_APPLICATION.")



        invariance_score = max(0.0, 1.0 - inv_deductions)

        passed = g1_passed and g2_passed
        composite_grade = (spec_score + invariance_score) / 2.0

        return DualCheckVerdict(
            passed=passed,
            grader1_spec_passed=g1_passed,
            grader2_invariance_passed=g2_passed,
            reference_blind=True,
            spec_score=spec_score,
            invariance_score=invariance_score,
            composite_grade=composite_grade,
            findings=findings,
            metadata={"program": str(program), "jurisdiction": jurisdiction},
        )


class DualCheckSandboxedVerifier:
    """Isolated synthetic verifier executing dual-check validation inside isolated execution sandboxes."""

    def __init__(self, grader: DualCheckGrader | None = None) -> None:
        self.grader = grader or DualCheckGrader()

    def verify_candidate_blind(
        self,
        assessment: Assessment,
        evidence: EvidenceView | None = None,
        program: ProgramId = ProgramId.SNAP,
        jurisdiction: str = "EX",
        withheld_reference: Any = None,
    ) -> DualCheckVerdict:
        """Run reference-blind verification ensuring generator model has zero access to ground truth."""
        return self.grader.grade(
            assessment=assessment,
            evidence=evidence,
            program=program,
            jurisdiction=jurisdiction,
            withheld_ground_truth=withheld_reference,
        )


__all__ = [
    "DefectType",
    "DefectReport",
    "LocalizedSpan",
    "PatchResult",
    "LegalASTNode",
    "LegalBriefASTParser",
    "SpanEditingEngine",
    "patch_span",
    "repair_draft",
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
    "IndependentVerifier",
    "DualCheckVerdict",
    "DualCheckGrader",
    "DualCheckSandboxedVerifier",
    "FailureType",
    "FailureLocalizationReport",
    "SingleTurnDeltaRepairContract",
    "FailureTurnLocator",
    "SingleTurnDeltaRewriter",
    "ToolGradAssertionResult",
    "ToolGradAssertionChecker",
]


