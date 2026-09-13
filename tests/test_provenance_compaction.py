"""Tests for Phase 2: Trace-as-State Context Compaction & Cryptographic Citations."""

import pytest

from tribune.corpus.citations import (
    CitationSegment,
    CryptographicCitationMapper,
)
from tribune.corpus.provenance import (
    CompactedTraceState,
    ProvenanceNode,
    ProvenanceNodeType,
    TraceAsStateEngine,
)
from tribune.corpus.rule_store import LocalRuleStore


def test_trace_as_state_engine_node_creation_and_chaining():
    """Verify TraceAsStateEngine builds graph nodes and records causal parent links."""
    engine = TraceAsStateEngine()

    n1 = engine.add_node(
        turn=1,
        node_type=ProvenanceNodeType.REASONING,
        content="Applicant household size is 3 with monthly income $1800.",
        environmental_constraints={"max_gross_income_fpl": 1.30},
    )
    assert n1.node_id.startswith("node_1_")
    assert len(n1.content_hash) == 64

    n2 = engine.add_node(
        turn=1,
        node_type=ProvenanceNodeType.TOOL_DISPATCH,
        content="calculate_snap_gross_income(1800, 0, 3)",
        parent_ids=[n1.node_id],
        file_mutations=[{"path": "evidence/income.json", "mutation": "created"}],
        permission_changes=[{"scope": "filesystem:write", "granted": True}],
    )
    assert n2.parent_ids == [n1.node_id]
    assert n2.node_id in engine._children[n1.node_id]


def test_trace_as_state_compaction_preserves_invariants_and_token_budget():
    """Verify compaction distills historical turns while strictly preserving invariants and segment hashes."""
    engine = TraceAsStateEngine()

    # Turn 1
    n1 = engine.add_node(
        turn=1,
        node_type=ProvenanceNodeType.ACTION,
        content="Query SNAP statutory income limits for household size 3.",
        environmental_constraints={"jurisdiction": "EX", "fpl_basis": 2026},
        file_mutations=[{"path": "/tmp/case_snapshot.json", "mutation": "created"}],
        permission_changes=[{"role": "intake_agent", "elevation": "standard"}],
    )
    # Turn 2
    n2 = engine.add_node(
        turn=2,
        node_type=ProvenanceNodeType.ACTION,
        content="Calculate standard deduction $198 and apply gross income test.",
        environmental_constraints={"standard_deduction": 198.0},
        file_mutations=[{"path": "/tmp/calculation.log", "mutation": "created"}],
    )
    # Turn 3 (Active turn)
    n3 = engine.add_node(
        turn=3,
        node_type=ProvenanceNodeType.REASONING,
        content="Applicant meets gross income eligibility; checking liquid assets limit.",
        environmental_constraints={"liquid_asset_cap": 3000.0},
    )

    compacted: CompactedTraceState = engine.compact_trace_history([n1, n2, n3], max_token_budget=250)

    assert compacted.original_node_count == 3
    # Verify strict preservation of invariants
    assert compacted.preserved_constraints["jurisdiction"] == "EX"
    assert compacted.preserved_constraints["standard_deduction"] == 198.0
    assert compacted.preserved_constraints["liquid_asset_cap"] == 3000.0
    assert len(compacted.preserved_file_mutations) == 2
    assert len(compacted.preserved_permission_changes) == 1

    # Verify all content hashes are preserved in the state delta
    assert n1.content_hash in compacted.preserved_segment_hashes
    assert n2.content_hash in compacted.preserved_segment_hashes
    assert n3.content_hash in compacted.preserved_segment_hashes

    # Prepending context
    prepended = engine.prepend_to_context("Evaluate Medicaid criteria next.", compacted)
    assert "=== COMPACTED TRACE-AS-STATE" in prepended
    assert "Evaluate Medicaid criteria next." in prepended


def test_cryptographic_citation_mapper_and_deterministic_verification():
    """Verify CryptographicCitationMapper indexes segments and deterministically verifies claims."""
    rule_store = LocalRuleStore()
    mapper = CryptographicCitationMapper(rule_store=rule_store)

    doc_text = (
        "Statutory eligibility under 7 CFR 273.9 stipulates that monthly gross income "
        "must not exceed 130 percent of the federal poverty line for household size 3."
    )
    segments = mapper.segment_document(doc_text, source_id="7_CFR_273.9", segment_size_words=10)
    assert len(segments) >= 1
    seg_target = segments[0]

    # Construct compacted state preserving the segment hash
    engine = TraceAsStateEngine()
    node = engine.add_node(
        turn=1,
        node_type=ProvenanceNodeType.OBSERVATION,
        content="Ingested statutory segment.",
        metadata={"segment_hash": seg_target.segment_hash},
    )
    compacted = engine.compact_trace_history([node])

    # 1. Assertion explicitly citing the segment hash
    assertion_with_cite = f"Applicant meets criteria under segment [cite:{seg_target.segment_hash}]."
    report = mapper.verify_assertion(assertion_with_cite, compacted)
    assert report.is_verified is True
    assert seg_target.segment_hash in report.matched_hashes

    # 2. Stripped padding / compacted text still verifies deterministically
    stripped_assertion = f"Gross income test verified [cite:{seg_target.segment_hash[:16]}]."
    report_stripped = mapper.verify_assertion(stripped_assertion, compacted)
    assert report_stripped.is_verified is True

    # 3. Fabricated or forged citation hash fails verification
    forged_assertion = "Applicant is eligible under [cite:0000000000000000deadbeef12345678]."
    report_forged = mapper.verify_assertion(forged_assertion, compacted)
    assert report_forged.is_verified is False
    assert len(report_forged.unmatched_hashes) >= 1


def test_recursive_trace_consolidator_multi_level_compaction():
    """Verify RecursiveTraceConsolidator generates multi-level traces with deterministic hashes."""
    import hashlib
    from tribune.memory.consolidation import (
        ConsolidatedMemoryTrace,
        ProvenanceNode,
        RecursiveTraceConsolidator,
    )

    consolidator = RecursiveTraceConsolidator(
        max_chunk_tokens=100,
        compaction_threshold_tokens=50,  # low threshold to trigger multi-level compaction
        branching_factor=2,
    )

    parent_hash = hashlib.sha256(b"root_case_session_context").hexdigest()

    # Raw traces from subagents
    raw_subagent_traces = [
        {
            "source_id": "subagent_income_doc",
            "chunk_index": 1,
            "content": "Gross monthly income is $1,450 from W2 employer paystub.",
            "confidence": 0.98,
            "metadata": {"doc_type": "paystub"},
        },
        {
            "source_id": "subagent_deductions",
            "chunk_index": 2,
            "content": "Standard deduction of $198 applied for household size 3.",
            "confidence": 0.95,
            "metadata": {"statute": "7 CFR 273.9"},
        },
        {
            "source_id": "subagent_assets",
            "chunk_index": 3,
            "content": "Liquid assets total $850 in checking account, below $3,000 cap.",
            "confidence": 0.99,
            "metadata": {"source": "bank_statement"},
        },
        {
            "source_id": "subagent_residency",
            "chunk_index": 4,
            "content": "Applicant is permanent state resident under EX statutory jurisdiction.",
            "confidence": 0.97,
            "metadata": {"statute": "7 CFR 273.2"},
        },
    ]

    # Compact traces
    trace: ConsolidatedMemoryTrace = consolidator.compact_subagent_traces(
        raw_traces=raw_subagent_traces,
        parent_context_hash=parent_hash,
    )

    # 1. Verify multi-level compaction occurred (4 traces with branching factor 2 -> Level 1)
    assert trace.compaction_level >= 1
    assert len(trace.sub_traces) >= 2
    assert trace.confidence_score > 0.90

    # 2. Verify deterministic trace ID incorporates parent_context_hash
    expected_id = hashlib.sha256(f"{parent_hash}:{trace.root_assertion}".encode("utf-8")).hexdigest()
    assert trace.trace_id == expected_id

    # 3. Verify leaf citations are completely preserved across recursive compaction levels
    all_leaves = trace.resolve_all_leaf_citations()
    assert len(all_leaves) == 4
    leaf_sources = {leaf.source_id for leaf in all_leaves}
    assert leaf_sources == {
        "subagent_income_doc",
        "subagent_deductions",
        "subagent_assets",
        "subagent_residency",
    }

    # 4. Verify verified SHA-256 chunk hashes match content
    for item in raw_subagent_traces:
        expected_chunk_hash = hashlib.sha256(item["content"].encode("utf-8")).hexdigest()
        matching_leaf = next(l for l in all_leaves if l.source_id == item["source_id"])
        assert matching_leaf.content_hash == expected_chunk_hash


def test_hierarchical_trace_retriever_and_citation_locking():
    """Verify retrieval harness navigates root assertions down to leaf ProvenanceNode citations."""
    import hashlib
    from tribune.memory.consolidation import (
        ConsolidatedMemoryTrace,
        ProvenanceNode,
        RecursiveTraceConsolidator,
    )
    from tribune.memory.retrieval import CitationLockHarness, HierarchicalTraceRetriever

    consolidator = RecursiveTraceConsolidator(branching_factor=2)
    parent_hash = hashlib.sha256(b"parent_test_context").hexdigest()

    raw_traces = [
        {
            "source_id": "w2_paystub.pdf",
            "chunk_index": 1,
            "content": "Monthly gross earnings: $1500.00",
            "confidence": 0.99,
        },
        {
            "source_id": "7_cfr_273_9.txt",
            "chunk_index": 10,
            "content": "Gross income standard <= 130% FPL",
            "confidence": 1.0,
        },
    ]

    trace = consolidator.compact_subagent_traces(raw_traces, parent_context_hash=parent_hash)

    # 1. Test HierarchicalTraceRetriever navigation
    retriever = HierarchicalTraceRetriever()
    leaves = retriever.resolve_leaf_sources(trace)
    assert len(leaves) == 2
    assert leaves[0].source_id == "w2_paystub.pdf"
    assert leaves[1].source_id == "7_cfr_273_9.txt"

    hierarchy = retriever.navigate_trace_hierarchy(trace)
    assert hierarchy["trace_id"] == trace.trace_id
    assert len(hierarchy["leaf_citations"]) == 2

    # 2. Test CitationLockHarness integration with ConsolidatedMemoryTrace
    harness = CitationLockHarness()
    harness.register_evidence(trace)

    # Valid assertion citing leaf source_id
    valid_output = "Income is verified under w2_paystub.pdf and statutory standard 7_cfr_273_9.txt."
    is_valid, _ = harness.validate_assertion(valid_output)
    assert is_valid is True

    # Valid assertion citing leaf chunk hash
    leaf_hash = leaves[0].content_hash
    valid_cite_hash = f"Applicant meets wage eligibility [cite: {leaf_hash[:16]}]."
    is_valid_h, _ = harness.validate_assertion(valid_cite_hash)
    assert is_valid_h is True

    # Forged citation rejected
    forged_output = "Applicant is eligible under [cite: ungrounded_forged_document.pdf@deadbeef:1-5]."
    is_valid_f, signal = harness.validate_assertion(forged_output)
    assert is_valid_f is False
    assert signal.is_abstention is True
