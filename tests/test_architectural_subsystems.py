"""Comprehensive test suite for the 5 Architectural Subsystem Enhancements across Tribune.

Subsystems:
1. Decomposed Span-Level Editing (verifier.py)
2. Turn-Efficient Trajectory Orchestration (costmodel.py, router.py)
3. Visual Self-Verification Loop (ocr.py, workspace.py, graph_builder.py, navigator.py)
4. Watermark Analytics and Provenance Tracking (provenance.py, disclosure.py)
5. Local Hybrid MoE Execution (registry.yaml, local_rules.py)
"""

from __future__ import annotations

import pytest

# 1. Span-Level Editing Imports
from tribune.agents.verifier import (
    DefectReport,
    DefectType,
    LegalASTNode,
    LegalBriefASTParser,
    PatchResult,
    SpanEditingEngine,
    Verifier,
    patch_span,
    repair_draft,
)
from tribune.corpus.rule_store import LocalRuleStore
from tribune.providers.base import SynthesisRequest
from tribune.types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    Evidence,
    EvidenceType,
    ProgramId,
    RecommendedAction,
    ProgramOutcome,
)


# 2. Trajectory Cost & Routing Imports
from tribune.eval.costmodel import (
    CostModel,
    TrajectoryCostModel,
    TrajectoryParetoPoint,
)
from tribune.providers.router import (
    DataSovereigntyLevel,
    ModelRouter,
    TrajectoryRoutingDecision,
)

# 3. Visual Layout & Verification Imports
from tribune.ingestion.ocr import (
    BoundingBox,
    OcrIngest,
    VisualDocumentLayout,
    VisualLayoutDAG,
    VisualLayoutToken,
)
from tribune.context.workspace import WorkspaceContext, WorkspaceState, DeltaPatch, PatchOperationType
from tribune.context.graph_builder import (
    AgentGraphBuilder,
    VisualLayoutGraphNode,
    build_visual_layout_subgraph,
)
from tribune.agents.navigator import (
    AgencyLayoutRule,
    LayoutVerificationReport,
    Navigator,
    VisualLayoutCrossReferencer,
)

# 4. Watermark Analytics & Provenance Imports
from tribune.corpus.provenance import (
    C2PAAssertion,
    C2PAManifest,
    SynthIDTextWatermarkDetector,
    SynthIDWatermarkScore,
    sign_c2pa_manifest,
    verify_c2pa_manifest,
)
from tribune.governance.disclosure import (
    generate_determination_notice,
    generate_statutory_disclosure_block,
)

# 5. Local MoE Imports
from tribune.providers.local_rules import (
    LocalMoEProvider,
    MoEConfig,
    MoEExpertRouter,
    LocalRulesProvider,
)
from tribune.config import get_settings


# =========================================================================== #
# Subsystem 1: Decomposed Span-Level Editing
# =========================================================================== #


def test_legal_brief_ast_parser_structure() -> None:
    """Test parsing a determination brief into structural AST nodes with accurate offsets."""
    text = (
        "OFFICIAL DETERMINATION NOTICE\n"
        "Jurisdiction: WA\n"
        "Pursuant to 7 CFR 273.9(a), monthly income is $1,200.00.\n"
        "Outcome: Likely Eligible"
    )
    nodes = LegalBriefASTParser.parse(text)
    assert len(nodes) == 4

    types = [n.node_type for n in nodes]
    assert types == ["HEADER", "JURISDICTION_BLOCK", "CITATION_BLOCK", "CONCLUSION"]
    assert nodes[1].metadata.get("jurisdiction") == "WA"
    assert nodes[0].start_offset == 0
    assert nodes[0].end_offset == len("OFFICIAL DETERMINATION NOTICE")


def test_span_editing_classify_defects() -> None:
    """Test 3-step defect classification identifying missing jurisdiction, invalid citations, and calculation errors."""
    rule_store = LocalRuleStore()
    engine = SpanEditingEngine(rule_store=rule_store)

    # Draft missing jurisdiction and containing an uncited claim and invalid statute
    draft = (
        "OFFICIAL NOTICE\n"
        "The applicant satisfies the statutory requirement for food support.\n"
        "Governed by 99 CFR 999.123.\n"
        "Total gross income limit stated as $500.00."
    )

    defects = engine.classify_defects(
        draft_text=draft,
        jurisdiction="EX",
        program=ProgramId.SNAP,
    )

    defect_types = {d.defect_type for d in defects}
    assert DefectType.MISSING_JURISDICTION_CLAUSE in defect_types
    assert DefectType.OUTDATED_STATUTE_REFERENCE in defect_types
    assert DefectType.UNCITED_CLAIM in defect_types



def test_span_editing_patch_span_and_repair_draft() -> None:
    """Test targeted span patching preserving surrounding verified text and citation anchors."""
    rule_store = LocalRuleStore()
    engine = SpanEditingEngine(rule_store=rule_store)

    draft = (
        "OFFICIAL NOTICE\n"
        "Jurisdiction: EX\n"
        "The claimant appears met under statutory threshold.\n"
        "Governed by 7 CFR 273.9(a)."
    )

    # Localize span
    loc = engine.localize_span(draft, "The claimant appears met under statutory threshold.")
    assert loc.start_offset > 0
    assert loc.matched_text == "The claimant appears met under statutory threshold."

    # Create defect
    defect = DefectReport(
        defect_type=DefectType.UNCITED_CLAIM,
        description="Missing citation anchor",
        span_text=loc.matched_text,
        start_offset=loc.start_offset,
        end_offset=loc.end_offset,
        suggested_patch="The claimant meets the statutory gross income test [pursuant to 7 CFR 273.9(a)].",
    )

    patch_res = engine.patch_span(draft, defect)
    assert patch_res.success
    assert "7 CFR 273.9(a)" in patch_res.applied_text
    assert "Jurisdiction: EX" in patch_res.applied_text
    assert "7 CFR 273.9(a)" in patch_res.preserved_anchors

    # Test full repair_draft
    repaired, results = engine.repair_draft(draft, [defect])
    assert len(results) == 1
    assert "The claimant meets the statutory gross income test" in repaired


def test_verifier_span_editing_integration() -> None:
    """Test Verifier agent exposing classify_defects, patch_span, and repair_draft."""
    rule_store = LocalRuleStore()
    provider = LocalRulesProvider()
    verifier = Verifier(provider=provider, rule_store=rule_store)

    draft = "DETERMINATION\nApplicant is eligible for assistance."
    defects = verifier.classify_defects(draft_text=draft, jurisdiction="EX", program=ProgramId.SNAP)
    assert len(defects) >= 1

    repaired_text, patches = verifier.repair_draft(draft, defects)
    assert len(patches) >= 1
    assert repaired_text != draft


# =========================================================================== #
# Subsystem 2: Turn-Efficient Trajectory Orchestration
# =========================================================================== #


def test_trajectory_cost_model_formula() -> None:
    """Verify TrajectoryCostModel implements Effective Cost = Base Cost * (Turns)^gamma + State Overhead."""
    gamma = 1.25
    state_overhead = 0.00005
    model = TrajectoryCostModel(gamma=gamma, state_transition_overhead_usd=state_overhead)

    base_cost = 0.001
    turns = 4
    state_transitions = 2

    # Expected: 0.001 * (4^1.25) + (2 * 0.00005)
    # 4^1.25 = 4^(5/4) = 2^(5/2) = sqrt(32) ≈ 5.6568542
    # 0.001 * 5.6568542 = 0.00565685
    # + 0.00010 = 0.00575685
    eff_cost = model.compute_effective_cost(
        base_cost=base_cost,
        turns_per_task=turns,
        state_transitions=state_transitions,
    )

    expected = round((base_cost * (turns ** gamma)) + (state_transitions * state_overhead), 8)
    assert eff_cost == expected


def test_trajectory_cost_model_pareto_frontier() -> None:
    """Verify Pareto frontier computation under trajectory scaling."""
    model = TrajectoryCostModel(gamma=1.20)

    points = [
        {"label": "frontier_tier2", "cost_per_1k": 0.010, "turns_per_task": 1.0, "accuracy": 0.99},
        {"label": "mid_tier1", "cost_per_1k": 0.003, "turns_per_task": 2.5, "accuracy": 0.94},
        {"label": "low_tier0", "cost_per_1k": 0.001, "turns_per_task": 6.0, "accuracy": 0.88},
    ]

    frontier = model.compute_trajectory_pareto_frontier(points, reference_label="frontier_tier2")
    assert len(frontier) == 3
    for p in frontier:
        assert isinstance(p, TrajectoryParetoPoint)
        assert p.effective_cost_per_1k > 0.0


def test_model_router_trajectory_efficiency_routing() -> None:
    """Verify ModelRouter prioritizes high-horizon agent endpoints for multi-file workspace cases."""
    settings = get_settings()
    router = ModelRouter(settings=settings)

    # 1. Single file standard case
    dec_standard = router.route_by_trajectory_efficiency(
        intent="parsing",
        context_length=500,
        estimated_turns=1,
        multi_file=False,
    )
    assert dec_standard.tier in (0, 1)
    assert dec_standard.horizon_mode == "standard"

    # 2. Multi-file complex case requiring deep tool sequences
    dec_multi_file = router.route_by_trajectory_efficiency(
        intent="parsing",
        context_length=2000,
        estimated_turns=4,
        multi_file=True,
    )
    assert dec_multi_file.tier == 2
    assert dec_multi_file.horizon_mode == "high_horizon"
    assert dec_multi_file.multi_file is True

    # 3. Air-gapped sovereignty requirement
    dec_sovereign = router.route_by_trajectory_efficiency(
        intent="statutory_determination",
        sovereignty_level=DataSovereigntyLevel.AIR_GAPPED_LOCAL,
    )
    assert dec_sovereign.tier == 0
    assert dec_sovereign.horizon_mode == "sovereign_local"


# =========================================================================== #
# Subsystem 3: Visual Self-Verification Loop
# =========================================================================== #


def test_ocr_visual_layout_extraction_and_dag() -> None:
    """Test OCR visual layout extraction, bounding boxes, and reading-order DAG creation."""
    from tribune.types import RawDocument
    settings = get_settings()
    ocr = OcrIngest(settings=settings)

    doc_text = (
        "DEPARTMENT OF SOCIAL AND HEALTH SERVICES\n"
        "Notice Date: 2026-08-26\n"
        "Gross Income: $1,400.00\n"
        "Household Size: 3\n"
        "Authorized Caseworker Signature: Jane Doe"
    )
    raw_doc = RawDocument(doc_id="doc_snap_001", doc_type="benefit_notice", text=doc_text)

    evidence, layout = ocr.ingest_with_layout(raw_doc)
    assert len(evidence) >= 1
    assert isinstance(layout, VisualDocumentLayout)
    assert layout.doc_id == "doc_snap_001"
    assert len(layout.tokens) == 5

    # Check bounding boxes & token types
    header_tok = layout.tokens[0]
    assert header_tok.token_type == "HEADER"
    assert header_tok.bbox.y0 <= 0.25

    sig_tok = layout.tokens[-1]
    assert sig_tok.token_type == "SIGNATURE_BLOCK"
    assert sig_tok.bbox.y1 >= 0.65

    # Check reading-order DAG
    reading_order = layout.layout_dag.topological_reading_order()
    assert len(reading_order) == 5
    assert reading_order[0].token_id == header_tok.token_id


def test_workspace_visual_layout_storage() -> None:
    """Test storing and reading visual layout objects in WorkspaceContext via JSON pointers."""
    ctx = WorkspaceContext(case_id="case_visual_01", jurisdiction="WA")

    bbox = BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2)
    tok = VisualLayoutToken(token_id="t1", text="STATE NOTICE", bbox=bbox, token_type="HEADER")
    layout = VisualDocumentLayout(doc_id="doc_v1", tokens=[tok], layout_dag=VisualLayoutDAG(nodes=[tok]))

    snap = ctx.store_visual_layout(layout)
    assert snap.version >= 1

    stored_layout = ctx.get_visual_layout("doc_v1")
    assert stored_layout is not None
    assert stored_layout["doc_id"] == "doc_v1"
    assert len(stored_layout["tokens"]) == 1


def test_visual_layout_cross_referencer_and_navigator() -> None:
    """Test VisualLayoutCrossReferencer validating agency rendering rules and column ordering."""
    cross_ref = VisualLayoutCrossReferencer()

    # 1. Compliant Layout
    good_tokens = [
        {"token_id": "tok_1", "text": "AGENCY HEADER", "token_type": "HEADER", "bbox": [0.1, 0.05, 0.9, 0.15]},
        {"token_id": "tok_2", "text": "Income: $1,000", "token_type": "KEY_VALUE", "bbox": [0.1, 0.30, 0.5, 0.40]},
        {"token_id": "tok_3", "text": "Signature: Officer", "token_type": "SIGNATURE_BLOCK", "bbox": [0.5, 0.70, 0.9, 0.85]},
    ]
    good_layout = {"tokens": good_tokens, "edges": [("tok_1", "tok_2"), ("tok_2", "tok_3")]}

    report_good = cross_ref.verify_layout(good_layout)
    assert report_good.is_compliant is True
    assert report_good.compliance_score == 1.0
    assert report_good.header_placement_valid is True
    assert report_good.signature_block_valid is True

    # 2. Non-Compliant Layout (Header displaced to bottom, inverted reading order)
    bad_tokens = [
        {"token_id": "tok_1", "text": "DISPLACED HEADER", "token_type": "HEADER", "bbox": [0.1, 0.60, 0.9, 0.75]},
        {"token_id": "tok_2", "text": "Signature: Officer", "token_type": "SIGNATURE_BLOCK", "bbox": [0.5, 0.20, 0.9, 0.35]},
    ]
    bad_layout = {"tokens": bad_tokens, "edges": [("tok_1", "tok_2")]}

    report_bad = cross_ref.verify_layout(bad_layout)
    assert report_bad.is_compliant is False
    assert report_bad.header_placement_valid is False
    assert report_bad.signature_block_valid is False
    assert len(report_bad.violations) >= 2

    # 3. Navigator Integration
    navigator = Navigator()
    nav_report = navigator.verify_layout_compliance(good_layout)
    assert nav_report.is_compliant is True


# =========================================================================== #
# Subsystem 4: Watermark Analytics and Provenance Tracking
# =========================================================================== #


def test_synthid_text_watermark_analytics() -> None:
    """Test SynthID-Text statistical watermark calculation (green/red counts, z-score, p_synthetic)."""
    detector = SynthIDTextWatermarkDetector(gamma=0.5, z_threshold=2.5, seed=42)

    sample_text = (
        "The State Department of Social Services hereby certifies statutory compliance "
        "for the household pursuant to 7 CFR 273.9. All eligibility criteria have been verified."
    )

    score = detector.compute_watermark_score(sample_text)
    assert isinstance(score, SynthIDWatermarkScore)
    assert score.total_tokens > 10
    assert score.green_token_count + score.red_token_count == score.total_tokens
    assert 0.0 <= score.green_token_ratio <= 1.0
    assert 0.0 <= score.p_synthetic <= 1.0
    assert score.confidence_level in ("HIGH", "MEDIUM", "LOW", "NONE")


def test_c2pa_manifest_signing_and_verification() -> None:
    """Test C2PA manifest creation, SHA-256 assertions, and cryptographic digital signature validation."""
    doc_text = "OFFICIAL DETERMINATION: Eligible under 7 CFR 273.9."
    case_id = "case_c2pa_test_99"
    citations = ["7 CFR 273.9(a)"]

    manifest = sign_c2pa_manifest(
        document_text=doc_text,
        case_id=case_id,
        statutory_citations=citations,
        private_key="TEST_ROOT_KEY_99",
    )

    assert isinstance(manifest, C2PAManifest)
    assert manifest.claim_generator == "Tribune-Sovereign-Provenance-Engine/1.0"
    assert manifest.get_data_hash() != ""
    assert len(manifest.assertions) == 3

    # Verification on unaltered text
    is_valid = verify_c2pa_manifest(manifest, doc_text, private_key="TEST_ROOT_KEY_99")
    assert is_valid is True

    # Verification on tampered text
    tampered_text = "OFFICIAL DETERMINATION: INELIGIBLE under 7 CFR 273.9."
    assert verify_c2pa_manifest(manifest, tampered_text, private_key="TEST_ROOT_KEY_99") is False

    # Verification with wrong key
    assert verify_c2pa_manifest(manifest, doc_text, private_key="WRONG_KEY") is False


def test_governance_statutory_disclosure_block() -> None:
    """Test EU AI Act Article 50 statutory disclaimers and C2PA cryptographic audit blocks."""
    score = SynthIDWatermarkScore(
        green_token_count=18,
        red_token_count=6,
        total_tokens=24,
        green_token_ratio=0.75,
        z_score=3.25,
        p_synthetic=0.985,
        is_watermarked=True,
        confidence_level="HIGH",
    )

    manifest = sign_c2pa_manifest("Sample Text", "case_test_01")

    blocks = generate_statutory_disclosure_block(
        verification_status=True,
        watermark_score=score,
        c2pa_manifest=manifest,
    )
    block_text = "\n".join(blocks)

    assert "EU AI ACT ART. 50" in block_text
    assert "SynthID-Text statistical signature DETECTED" in block_text
    assert "C2PA CRYPTOGRAPHIC PROVENANCE" in block_text
    assert manifest.instance_id in block_text


def test_generate_determination_notice_with_provenance() -> None:
    """Test generate_determination_notice seamlessly embedding SynthID and C2PA audit blocks."""
    assessment = Assessment(
        assessment_id="asmt_001",
        case_id="case_001",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        criteria=[
            CriterionResult(
                criterion_id="snap_gross_income",
                description="Gross income limit",
                required=True,
                outcome=CriterionOutcome.SATISFIED,
                citation_ids=["7_cfr_273_9_a"],
            )
        ],
        citations=[
            Citation(
                citation_id="7_cfr_273_9_a",
                program=ProgramId.SNAP,
                jurisdiction="EX",
                source="7 CFR 273.9(a)",
                title="Income Eligibility Limits",
                text="Income limit",
            )
        ],
        self_confidence=0.95,
        rationale="Applicant gross income is below statutory limit.",
    )

    outcome = ProgramOutcome(
        program=ProgramId.SNAP,
        assessment=assessment,
        abstained=False,
    )

    notice = generate_determination_notice(
        outcome=outcome,
        attach_c2pa_audit=True,
    )

    assert "OFFICIAL DETERMINATION & DISCLOSURE NOTICE" in notice
    assert "STATUTORY AI TRANSPARENCY DISCLOSURE" in notice
    assert "C2PA CRYPTOGRAPHIC PROVENANCE & AUDIT MANIFEST" in notice
    assert "7 CFR 273.9(a)" in notice


# =========================================================================== #
# Subsystem 5: Local Hybrid MoE Execution
# =========================================================================== #


def test_moe_expert_router_top_k_gating() -> None:
    """Test MoEExpertRouter top-k gating vector, entropy, and usage accounting."""
    config = MoEConfig(
        model_name="qwen2.5-moe-7b-int4",
        num_experts=8,
        active_experts_per_token=2,
        quantization="int4_awq",
    )
    router = MoEExpertRouter(config)

    routing = router.route_context("Evaluate gross income for SNAP applicant", program="snap")
    assert "active_expert_indices" in routing
    assert len(routing["active_expert_indices"]) == 2
    assert len(routing["expert_weights"]) == 2
    assert routing["quantization"] == "int4_awq"
    assert routing["gating_entropy"] > 0.0

    # Expert counts updated
    total_uses = sum(router.expert_usage_counts.values())
    assert total_uses == 2


def test_local_moe_provider_execution() -> None:
    """Test LocalMoEProvider executing local air-gapped synthesis with MoE telemetry."""
    cfg = MoEConfig(
        model_name="qwen2.5-moe-7b-int8",
        num_experts=8,
        active_experts_per_token=2,
        quantization="bitsandbytes_int8",
    )
    provider = LocalMoEProvider(moe_config=cfg, role="proposer")

    cit = Citation(
        citation_id="7_cfr_273_9_a",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        source="7 CFR 273.9(a)",
        title="Gross Income Limit",
        text="Income limit",
    )

    req = SynthesisRequest(
        program=ProgramId.SNAP,
        jurisdiction="EX",
        criteria=[
            CriterionResult(
                criterion_id="snap_gross",
                description="Gross income",
                required=True,
                outcome=CriterionOutcome.SATISFIED,
                citation_ids=["7_cfr_273_9_a"],
            )
        ],
        required_total=1,
        coverage_complete=True,
        evidence_summary="Monthly income verified.",
        citations=[cit],
    )

    res = provider.synthesize_assessment(req)
    assert res.status == EligibilityStatus.LIKELY_ELIGIBLE
    assert "[Local MoE Execution]" in res.rationale
    assert "bitsandbytes_int8" in res.rationale



