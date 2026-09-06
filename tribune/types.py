"""Shared, strict, typed models for TRIBUNE.

This module is the single source of truth for every value that crosses a module
boundary: assessments, evidence, citations, provenance, abstention scores, audit
records, and the synthetic-case structures. Safety invariants that *must* hold no
matter which code path produced a value are enforced here as pydantic validators
so they cannot be bypassed (e.g. an eligibility claim without a citation simply
cannot be instantiated).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictStr,
    model_validator,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base for all TRIBUNE domain models.

    ``extra="forbid"`` rejects unknown fields (typo-proofing the typed contracts),
    ``validate_assignment`` keeps invariants enforced after construction, and
    ``use_enum_values=False`` keeps enums as enums so downstream code can pattern
    match on them.
    """

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, frozen=True, protected_namespaces=()
    )


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ProgramId(str, enum.Enum):
    SNAP = "snap"
    UNEMPLOYMENT = "unemployment"
    MEDICAID = "medicaid"
    HOUSING = "housing"
    APPEALS = "appeals"


class EvidenceType(str, enum.Enum):
    HOUSEHOLD_SIZE = "household_size"
    MONTHLY_INCOME = "monthly_income"
    ANNUAL_INCOME = "annual_income"
    LIQUID_ASSETS = "liquid_assets"
    RESIDENCY_STATE = "residency_state"
    RESIDENT = "resident"
    CITIZENSHIP_STATUS = "citizenship_status"  # citizen | qualified_immigrant | other
    AGE = "age"
    DISABLED = "disabled"
    PREGNANT = "pregnant"
    HAS_DEPENDENT_CHILD = "has_dependent_child"
    EMPLOYMENT_STATUS = "employment_status"  # employed | unemployed
    SEPARATION_REASON = "separation_reason"  # laid_off | fired_misconduct | quit_no_cause | quit_good_cause
    BASE_PERIOD_EARNINGS = "base_period_earnings"
    WEEKS_WORKED = "weeks_worked"
    ABLE_AND_AVAILABLE = "able_and_available"
    MONTHLY_RENT = "monthly_rent"
    WAITLIST_STATUS = "waitlist_status"  # open | closed | unknown
    DENIAL_DATE = "denial_date"  # ISO date string, for appeals
    DAYS_SINCE_DENIAL = "days_since_denial"
    APPEAL_GROUNDS = "appeal_grounds"


class CriterionOutcome(str, enum.Enum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNKNOWN = "unknown"  # insufficient evidence to evaluate this criterion


class EligibilityStatus(str, enum.Enum):
    LIKELY_ELIGIBLE = "likely_eligible"
    LIKELY_INELIGIBLE = "likely_ineligible"
    INDETERMINATE = "indeterminate"


class RecommendedAction(str, enum.Enum):
    PREPARE_APPLICATION = "prepare_application"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    ABSTAIN_AND_ESCALATE = "abstain_and_escalate"


class GroundTruthLabel(str, enum.Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class WaitlistStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class SMState(str, enum.Enum):
    """States of the per-case state machine."""

    PLAN = "plan"
    GATHER = "gather"
    ASSESS = "assess"
    VERIFY = "verify"
    PREPARE = "prepare"
    ABSTAIN = "abstain"
    REPLAN = "replan"
    DONE = "done"


class IngestMethod(str, enum.Enum):
    STRUCTURED = "structured"
    OCR = "ocr"
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


# --------------------------------------------------------------------------- #
# Provenance, citations, evidence
# --------------------------------------------------------------------------- #


class Provenance(StrictModel):
    """Where a piece of evidence came from and whether it has been anonymized.

    Provenance is attached at ingest time and travels with the evidence forever so
    a person or caseworker can always trace a claim back to its source document.
    """

    source_doc_id: str
    ingest_method: IngestMethod
    anonymized: bool
    content_hash: str
    created_at: datetime = Field(default_factory=_utcnow)
    notes: str = ""


class Citation(StrictModel):
    """A verifiable pointer to a governing rule, statute, or policy.

    Every substantive eligibility claim must carry at least one of these. The
    ``text`` is the searchable rule text used by late-interaction retrieval; the
    ``source`` and ``locator`` let a human find and read the original.
    """

    citation_id: str
    program: ProgramId
    jurisdiction: str
    source: str  # e.g. "7 CFR 273.9(a)(1)"
    title: str
    text: str
    locator: str = ""  # URL or section pointer
    effective_date: str | None = None  # ISO date the cited rule took effect, if known


class Evidence(StrictModel):
    """A single, typed, provenance-tagged fact about a person's situation."""

    evidence_id: str
    type: EvidenceType
    # Strict members so a round-trip through memory/JSON never silently coerces a
    # bool into a float (1.0) or vice versa. Construction sites always pass the
    # exact Python type (see casegen/ingestion).
    value: StrictBool | StrictFloat | StrictStr
    provenance: Provenance
    confidence: float = 1.0
    note: str = ""

    def as_float(self) -> float:
        if isinstance(self.value, bool):  # bool is an int subclass; treat distinctly
            raise TypeError(f"evidence {self.evidence_id} is boolean, not numeric")
        if isinstance(self.value, str):
            raise TypeError(f"evidence {self.evidence_id} is text, not numeric")
        return float(self.value)

    def as_bool(self) -> bool:
        if not isinstance(self.value, bool):
            raise TypeError(f"evidence {self.evidence_id} is not boolean")
        return self.value

    def as_str(self) -> str:
        if not isinstance(self.value, str):
            raise TypeError(f"evidence {self.evidence_id} is not text")
        return self.value


class EvidenceView:
    """Read-only, typed lookup over a list of :class:`Evidence`.

    Predicates ask this view for facts. A *missing* fact is the whole point: it is
    how a criterion becomes :class:`CriterionOutcome.UNKNOWN` instead of being
    silently assumed. This is not a pydantic model because it is a transient query
    helper, not a persisted value.
    """

    def __init__(self, evidence: list[Evidence]):
        self._by_type: dict[EvidenceType, Evidence] = {}
        for ev in evidence:
            self._by_type[ev.type] = ev  # last write wins

    def has(self, t: EvidenceType) -> bool:
        return t in self._by_type

    def get(self, t: EvidenceType) -> Evidence | None:
        return self._by_type.get(t)

    def num(self, t: EvidenceType) -> float | None:
        ev = self._by_type.get(t)
        return ev.as_float() if ev is not None else None

    def flag(self, t: EvidenceType) -> bool | None:
        ev = self._by_type.get(t)
        return ev.as_bool() if ev is not None else None

    def text(self, t: EvidenceType) -> str | None:
        ev = self._by_type.get(t)
        return ev.as_str() if ev is not None else None

    def ids(self, *types: EvidenceType) -> list[str]:
        out: list[str] = []
        for t in types:
            ev = self._by_type.get(t)
            if ev is not None:
                out.append(ev.evidence_id)
        return out


# --------------------------------------------------------------------------- #
# Assessment and verification
# --------------------------------------------------------------------------- #


class CriterionResult(StrictModel):
    """The evaluation of one program criterion against the available evidence."""

    criterion_id: str
    description: str
    outcome: CriterionOutcome
    required: bool
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _resolved_criteria_must_be_cited(self) -> CriterionResult:
        # A criterion that takes a position (satisfied / not satisfied) is a
        # substantive claim and therefore must point at the rule it relied on.
        if self.outcome is not CriterionOutcome.UNKNOWN and not self.citation_ids:
            raise ValueError(
                f"criterion '{self.criterion_id}' resolved to {self.outcome.value} "
                "without any citation; uncited eligibility logic is forbidden"
            )
        return self


class Assessment(StrictModel):
    """A fully-cited eligibility *assessment* — never a binding determination.

    Construction enforces TRIBUNE's cardinal rule: a non-indeterminate status is an
    eligibility claim, and an eligibility claim without citations cannot exist.
    """

    assessment_id: str
    case_id: str
    program: ProgramId
    jurisdiction: str
    status: EligibilityStatus
    criteria: list[CriterionResult]
    citations: list[Citation]
    evidence_ids: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
    self_confidence: float
    rationale: str
    waitlist_status: WaitlistStatus | None = None  # housing: eligibility != access
    attempt: int = 1
    notes: str = ""

    @model_validator(mode="after")
    def _enforce_citation_and_reference_integrity(self) -> Assessment:
        known = {c.citation_id for c in self.citations}

        # Every citation a criterion references must actually be present.
        for crit in self.criteria:
            for cid in crit.citation_ids:
                if cid not in known:
                    raise ValueError(
                        f"criterion '{crit.criterion_id}' references unknown citation '{cid}'"
                    )

        # An eligibility claim (eligible / ineligible) must be cited.
        if self.status is not EligibilityStatus.INDETERMINATE:
            if not self.citations:
                raise ValueError(
                    f"assessment {self.assessment_id} asserts {self.status.value} "
                    "with no citations; uncited eligibility claims are forbidden"
                )
            for crit in self.criteria:
                if crit.outcome is not CriterionOutcome.UNKNOWN and not crit.citation_ids:
                    raise ValueError(
                        f"assessment {self.assessment_id} asserts {self.status.value} but "
                        f"criterion '{crit.criterion_id}' is uncited"
                    )
        return self

    @property
    def is_assertion(self) -> bool:
        """True when the assessment takes an eligibility position (not indeterminate)."""
        return self.status is not EligibilityStatus.INDETERMINATE


class VerifierVerdict(StrictModel):
    """The verifier's independent re-derivation result."""

    approved: bool
    recomputed_status: EligibilityStatus
    missing_citations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    incomplete_coverage: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    self_testing_score: float = 1.0
    trajectory_steps: list[dict] = Field(default_factory=list)


class AbstentionScore(StrictModel):
    """Calibrated confidence and the resulting assert-or-abstain decision."""

    calibrated_confidence: float
    threshold: float
    abstain: bool
    reason: str
    features: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Usage metering (tokens, turns, cost)
# --------------------------------------------------------------------------- #


class ModelCallUsage(StrictModel):
    """Token accounting for a single model round-trip.

    ``estimated`` marks token counts produced by a deterministic local estimator
    (the offline provider has no real tokenizer) rather than reported by a serving
    backend. Costing treats both identically; reports surface the flag.
    """

    role: str  # "proposer" | "verifier"
    model: str
    tokenizer_id: str
    tokens_input: int
    tokens_output: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False


class TaskUsage(BaseModel):
    """Accumulated usage for one (case, program) task: tokens, turns, and cost.

    A *task* here is one program outcome — the unit the cost-per-completed-
    verification metric is defined over. Abstaining tasks are completed tasks and
    accumulate (and report) their actual cost like any other outcome.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    program: str = ""
    language: str = "en"
    turns: int = 0
    proposer_turns: int = 0
    verifier_turns: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: list[ModelCallUsage] = Field(default_factory=list)
    tokenizer_ids: list[str] = Field(default_factory=list)
    estimated: bool = False
    cost_usd: float | None = None
    cost_backend_id: str | None = None


# --------------------------------------------------------------------------- #
# Governance: prepared materials and audit
# --------------------------------------------------------------------------- #


class PreparedMaterials(StrictModel):
    """Application / appeal materials assembled by the preparer.

    ``submitted`` is structurally pinned to ``False``: TRIBUNE never auto-submits
    anything binding. Submission requires a human action routed through the
    action-gate (see :mod:`tribune.governance.action_gate`).
    """

    program: ProgramId
    jurisdiction: str
    case_id: str
    application_fields: dict[str, str] = Field(default_factory=dict)
    document_checklist: list[str] = Field(default_factory=list)
    appeal_packet: dict[str, str] | None = None
    submitted: bool = False
    action_required_note: str = (
        "TRIBUNE has prepared these materials. Nothing has been submitted. "
        "You or your navigator must review and submit them yourself."
    )

    @model_validator(mode="after")
    def _never_pre_submitted(self) -> PreparedMaterials:
        if self.submitted:
            raise ValueError("prepared materials may not be constructed as already submitted")
        return self


class SubmissionReceipt(StrictModel):
    """Proof that a human explicitly authorized a binding submission."""

    program: ProgramId
    case_id: str
    authorized_by: str
    signoff_token: str
    authorized_at: datetime = Field(default_factory=_utcnow)


class AuditRecord(StrictModel):
    """One append-only, hash-chained entry in the inspectable audit log."""

    record_id: str
    case_id: str
    sequence: int
    state: SMState
    agent: str
    action: str
    model_name: str = "n/a"
    model_version: str = "n/a"
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    payload: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    prev_hash: str = ""
    record_hash: str = ""


# --------------------------------------------------------------------------- #
# Applicant situations and synthetic cases
# --------------------------------------------------------------------------- #


class ApplicantSituation(StrictModel):
    """The complete, ground-truth facts of a (synthetic) person's situation.

    This is the *truth*. What TRIBUNE actually sees is a possibly-incomplete set of
    :class:`Evidence` derived from it — that gap is what makes criteria UNKNOWN and
    drives gather-more-evidence and abstention.
    """

    case_id: str
    jurisdiction: str
    household_size: int
    monthly_income: float
    liquid_assets: float
    resident: bool
    citizenship_status: str  # citizen | qualified_immigrant | undocumented
    age: int
    disabled: bool
    pregnant: bool
    has_dependent_child: bool
    employment_status: str  # employed | unemployed
    separation_reason: str | None = None
    base_period_earnings: float = 0.0
    weeks_worked: int = 0
    able_and_available: bool = True
    monthly_rent: float = 0.0
    waitlist_status: WaitlistStatus = WaitlistStatus.UNKNOWN
    days_since_denial: int | None = None
    appeal_grounds: str | None = None

    @property
    def annual_income(self) -> float:
        return self.monthly_income * 12.0


class ProgramGroundTruth(StrictModel):
    program: ProgramId
    label: GroundTruthLabel
    ambiguous: bool
    rationale: str
    decisive_criteria: list[str] = Field(default_factory=list)


class RawDocument(StrictModel):
    """A document as it arrives for ingestion (synthetic, structured, or a scan)."""

    doc_id: str
    doc_type: str  # pay_stub | benefit_letter | id | lease | denial_notice ...
    text: str = ""
    fields: dict[str, str] = Field(default_factory=dict)
    image_path: str | None = None


class SyntheticCase(StrictModel):
    case_id: str
    jurisdiction: str
    situation: ApplicantSituation
    evidence: list[Evidence]  # the (possibly incomplete) subset TRIBUNE sees
    documents: list[RawDocument]
    ground_truth: dict[ProgramId, ProgramGroundTruth]
    target_programs: list[ProgramId]
    language: str = "en"  # BCP-47-ish tag of the case documents ("en", "es")


# --------------------------------------------------------------------------- #
# Per-case run result (mutable accumulator)
# --------------------------------------------------------------------------- #


class ProgramOutcome(BaseModel):
    """The final outcome for a single program within a case run."""

    model_config = ConfigDict(extra="forbid")

    program: ProgramId
    assessment: Assessment | None = None
    verdict: VerifierVerdict | None = None
    abstention: AbstentionScore | None = None
    materials: PreparedMaterials | None = None
    final_state: SMState = SMState.DONE
    abstained: bool = False
    replans: int = 0
    usage: TaskUsage | None = None
    ocr_latency_ms: float = 0.0
    citation_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    human_path: str = (
        "A navigator or caseworker can review this with you. "
        "You may also contact the administering agency directly."
    )


class CaseRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    jurisdiction: str
    outcomes: list[ProgramOutcome] = Field(default_factory=list)
    audit: list[AuditRecord] = Field(default_factory=list)
    ocr_latency_ms: float = 0.0
    citation_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    workspace_version: int = 0
    token_reduction: TokenReductionMetric | None = None
    failure_traces: list[dict[str, Any]] = Field(default_factory=list)
    speculative_dispatched: int = 0
    speculative_hits: int = 0
    speculative_misses: int = 0
    speculative_cancelled: int = 0
    speculative_latency_saved_ms: float = 0.0

    def outcome_for(self, program: ProgramId) -> ProgramOutcome | None:
        for o in self.outcomes:
            if o.program is program:
                return o
        return None


# --------------------------------------------------------------------------- #
# Workspace & Delta Patch Architecture
# --------------------------------------------------------------------------- #


class PatchOperationType(str, enum.Enum):
    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"
    TEST = "test"
    APPEND_UNIQUE = "append_unique"
    MERGE_DICT = "merge_dict"


class PatchProvenance(StrictModel):
    """Provenance tracking origin, citations, and confidence for a delta patch."""

    agent_id: str
    timestamp: datetime = Field(default_factory=_utcnow)
    task_id: str = ""
    citation_keys: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    rationale_reference: str = ""
    notes: str = ""


class DeltaPatch(StrictModel):
    """Structured JSON patch operation modifying central shared workspace state.

    Supports JSON-pointer paths, version pinning, and conflict constraints.
    """

    schema_version: int = 1
    run_id: str
    agent_id: str
    operation: PatchOperationType
    path: str  # e.g. "/evidence", "/assessments/snap", "/metadata/status"
    value: Any = None
    expected_version: int | None = None  # Optimistic concurrency check
    conflict_strategy: str = "error"  # "error" | "overwrite" | "merge"
    provenance: PatchProvenance = Field(
        default_factory=lambda: PatchProvenance(agent_id="unknown")
    )


class WorkspaceSnapshot(StrictModel):
    """Immutable snapshot of the shared workspace state."""

    version: int
    case_id: str
    jurisdiction: str
    state_data: dict[str, Any]
    patch_count: int
    timestamp: datetime = Field(default_factory=_utcnow)


class TokenReductionMetric(BaseModel):
    """Metrics measuring token volume reductions from shared workspace context."""

    model_config = ConfigDict(extra="forbid")

    baseline_tokens: int
    workspace_tokens: int
    reduction_percentage: float
    agent_count: int
    patch_volume_bytes: int
    state_size_bytes: int


# --------------------------------------------------------------------------- #
# Guarded Harness Evolution & Failure Telemetry Models
# --------------------------------------------------------------------------- #


class FailureCategory(str, enum.Enum):
    GOVERNANCE_VIOLATION = "governance_violation"
    CITATION_MISMATCH = "citation_mismatch"
    PREDICATE_ERROR = "predicate_error"
    SCHEMA_DEVIATION = "schema_deviation"
    BOUNDARY_CONFLICT = "boundary_conflict"
    HALLUCINATION_MONOLOGUE = "hallucination_monologue"
    COVERAGE_GAP = "coverage_gap"
    GENERAL_FAILURE = "general_failure"


class FailureTrace(StrictModel):
    """Structured failure execution trace emitted during agent/state machine pipeline runs."""

    trace_id: str
    case_id: str
    program: ProgramId | None = None
    agent_id: str = "unknown"
    category: FailureCategory = FailureCategory.GENERAL_FAILURE
    error_message: str = ""
    context_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)


class PatchType(str, enum.Enum):
    PROMPT_REFINEMENT = "prompt_refinement"
    ROUTING_ADJUSTMENT = "routing_adjustment"
    GUARDRAIL_TUNING = "guardrail_tuning"
    CRITERIA_CLARIFICATION = "criteria_clarification"


class PatchStatus(str, enum.Enum):
    PROPOSED = "proposed"
    CANARY_TESTED = "canary_tested"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class AgentHarnessPatch(StrictModel):
    """Candidate prompt modification or routing adjustment patch for target agent harnesses."""

    patch_id: str
    target_agent: str
    target_program: ProgramId | None = None
    patch_type: PatchType = PatchType.PROMPT_REFINEMENT
    description: str = ""
    original_prompt_template: str = ""
    patched_prompt_template: str = ""
    routing_overrides: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    version: int = 1
    status: PatchStatus = PatchStatus.PROPOSED
    created_at: datetime = Field(default_factory=_utcnow)


class PromotionMetrics(StrictModel):
    """Evaluation metrics determining whether a candidate harness patch cleared promotion gating."""

    baseline_parity_ratio: float = 1.0
    canary_passed: bool = True
    confidently_wrong_count: int = 0
    citation_accuracy: float = 1.0
    abstention_recall: float = 1.0
    governance_regressions: int = 0
    approved: bool = False
    reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Phase 1: Proactive Context Engineering & Relational Graph Memory Types
# --------------------------------------------------------------------------- #


class CompactionUrgency(str, enum.Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class ContextAnalysis(StrictModel):
    """Calculated exact token count, information density, and Shannon entropy for working memory."""

    token_count: int
    information_density: float
    shannon_entropy: float
    working_memory_bytes: int = 0
    analyzed_at: datetime = Field(default_factory=_utcnow)


class BudgetStatus(StrictModel):
    """Remaining context window quota, consumption velocity, and compaction urgency status."""

    total_budget: int
    used_tokens: int
    remaining_quota: int
    consumption_velocity: float
    compaction_urgency: CompactionUrgency = CompactionUrgency.NORMAL
    is_urgent: bool = False
    active_spans_count: int = 0


class FoldResult(StrictModel):
    """Result of folding resolved interaction spans with semantic indexing headers."""

    span_id: str
    discarded_tokens: int
    indexing_header: str
    summary: str = ""
    preserved_keys: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)


class EntityNode(StrictModel):
    """A node in the relational entity-temporal graph memory store."""

    node_id: str
    name: str
    entity_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class EntityRelation(StrictModel):
    """A directed edge in the relational entity-temporal graph memory store."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalChain(StrictModel):
    """A verified causal deduction chain connecting premises, predicates, and consequences."""

    chain_id: str
    premise: str
    predicate: str
    consequence: str
    confidence: float = 1.0
    evidence_citations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)


class EntityGraph(StrictModel):
    """Relational entity graph snapshot containing nodes and relations."""

    nodes: list[EntityNode] = Field(default_factory=list)
    edges: list[EntityRelation] = Field(default_factory=list)


class GraphQuery(StrictModel):
    """Query specification for traversing the relational entity-temporal graph."""

    root_node_ids: list[str] = Field(default_factory=list)
    relation_filters: list[str] = Field(default_factory=list)
    entity_type_filters: list[str] = Field(default_factory=list)
    max_hops: int = 2
    start_time: datetime | None = None
    end_time: datetime | None = None


class SubGraphResult(StrictModel):
    """Subgraph extracted from episodic memory satisfying a GraphQuery."""

    nodes: list[EntityNode] = Field(default_factory=list)
    edges: list[EntityRelation] = Field(default_factory=list)
    causal_chains: list[CausalChain] = Field(default_factory=list)
    total_hop_depth: int = 0


# --------------------------------------------------------------------------- #
# Phase 2: Out-of-Band Governance & Hardened Sandbox Types
# --------------------------------------------------------------------------- #


class ActionProposal(StrictModel):
    """High-stakes action proposal emitted by agent for out-of-band evaluation."""

    proposal_id: str
    case_id: str
    agent_id: str
    action_type: str  # e.g. "formal_filing", "database_mutation", "external_api"
    target_resource: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class HMACAuthorizationToken(StrictModel):
    """Cryptographically signed, short-lived HMAC authorization token minted by supervisor."""

    token_id: str
    proposal_id: str
    action_type: str
    nonce: str
    signature: str
    issued_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime
    supervisor_id: str = "gatekeeper_supervisor_v1"



