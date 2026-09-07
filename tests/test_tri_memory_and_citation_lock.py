"""Tests for Track 2: Tri-Memory Architecture and Citation-Locked Retrieval."""


from tribune.context.graph import DependencyEdgeType, EntityEventGraph, EntityKind
from tribune.memory.hdm import HDMTier, HierarchicalDocumentaryMemory
from tribune.memory.retrieval import (
    AbstentionSignal,
    CitationLockHarness,
    ProvenancePointer,
)
from tribune.memory.timeline import MemoryEventsTimeline
from tribune.security.audit import SecurityEventType, get_security_audit_logger


def test_memory_events_timeline_temporal_immutability_and_reconciliation():
    """Verify MemoryEventsTimeline enforces append-only chronological ordering and deterministic state reconciliation."""
    timeline = MemoryEventsTimeline(timeline_id="case_101_timeline")

    # Append sequential events
    timeline.append(
        transaction_id="tx_01",
        state_change_delta={"status": "PLANNING", "eligibility_count": 0},
        timestamp=100.0,
    )
    timeline.append(
        transaction_id="tx_02",
        state_change_delta={"status": "GATHERING", "documents": ["tax_2025.pdf"]},
        executed_tool_invocation={"tool": "ocr_ingest", "doc": "tax_2025.pdf"},
        timestamp=101.0,
    )
    timeline.append(
        transaction_id="tx_03",
        state_change_delta={"status": "ASSESSING", "eligibility_count": 2},
        timestamp=102.0,
    )

    assert timeline.count() == 3
    assert timeline.verify_temporal_immutability() is True

    # State reconciliation at t=101.5 (should have tx_01 and tx_02 applied, but not tx_03)
    state_at_101 = timeline.get_state_at(timestamp=101.5)
    assert state_at_101["status"] == "GATHERING"
    assert state_at_101["eligibility_count"] == 0
    assert state_at_101["documents"] == ["tax_2025.pdf"]

    # State reconciliation at t=105.0 (all applied)
    state_at_105 = timeline.get_state_at(timestamp=105.0)
    assert state_at_105["status"] == "ASSESSING"
    assert state_at_105["eligibility_count"] == 2

    # Query range
    events_range = timeline.query_sequence_range(from_seq=1, to_seq=2)
    assert len(events_range) == 2
    assert events_range[0].transaction_id == "tx_02"
    assert events_range[1].transaction_id == "tx_03"


def test_entity_event_graph_k_hop_traversal():
    """Verify EntityEventGraph multi-hop neighborhood queries and directional edge tracking."""
    graph = EntityEventGraph(graph_id="tribune_repo")

    # Register entities: Module -> Class -> Method -> Database
    graph.add_entity("mod_pipeline", EntityKind.MODULE, "tribune.pipeline", "tribune/pipeline.py", line_start=1, line_end=50)
    graph.add_entity("cls_runner", EntityKind.CLASS, "PipelineRunner", "tribune/pipeline.py", line_start=51, line_end=200)
    graph.add_entity("mth_execute", EntityKind.METHOD, "execute_step", "tribune/pipeline.py", line_start=120, line_end=150)
    graph.add_entity("db_audit", EntityKind.DATABASE_SCHEMA, "AuditLogSchema", "schema/audit.sql", line_start=1, line_end=30)

    # Add directional dependencies
    graph.add_dependency("mod_pipeline", "cls_runner", DependencyEdgeType.IMPORTS)
    graph.add_dependency("cls_runner", "mth_execute", DependencyEdgeType.INHERITS)
    graph.add_dependency("mth_execute", "db_audit", DependencyEdgeType.MUTATES)

    # 1-hop outgoing from mod_pipeline
    hop1 = graph.get_k_hop_neighborhood("mod_pipeline", k=1, direction="out")
    node_ids_1 = {n["entity_id"] for n in hop1["nodes"]}
    assert "mod_pipeline" in node_ids_1
    assert "cls_runner" in node_ids_1
    assert "mth_execute" not in node_ids_1

    # 2-hop outgoing from mod_pipeline
    hop2 = graph.get_k_hop_neighborhood("mod_pipeline", k=2, direction="out")
    node_ids_2 = {n["entity_id"] for n in hop2["nodes"]}
    assert "cls_runner" in node_ids_2
    assert "mth_execute" in node_ids_2
    assert "db_audit" not in node_ids_2

    # 3-hop outgoing reaches database schema
    hop3 = graph.get_k_hop_neighborhood("mod_pipeline", k=3, direction="out")
    node_ids_3 = {n["entity_id"] for n in hop3["nodes"]}
    assert "db_audit" in node_ids_3

    # Shortest path
    path = graph.find_path("mod_pipeline", "db_audit")
    assert path is not None
    assert len(path) == 3
    assert path[0].edge_type == DependencyEdgeType.IMPORTS
    assert path[1].edge_type == DependencyEdgeType.INHERITS
    assert path[2].edge_type == DependencyEdgeType.MUTATES


def test_hierarchical_documentary_memory_tiers_and_drilldown():
    """Verify HDM stores L0, L1, L2 tiers and executes drill-down and rollup."""
    hdm = HierarchicalDocumentaryMemory()

    # L0: Global invariant
    hdm.store_document(
        doc_id="INV-001",
        tier=HDMTier.L0_GLOBAL,
        title="Zero Uncited Claim Invariant",
        content="No eligibility determination may be made without statutory citations.",
        source_uri="docs/architecture.md",
        line_start=10,
        line_end=20,
    )

    # L1: Module contract child of L0
    hdm.store_document(
        doc_id="CTR-SNAP",
        tier=HDMTier.L1_MODULE,
        title="SNAP Assessment Contract",
        content="Evaluates income against 130% FPL standard.",
        source_uri="tribune/corpus/programs/snap.py",
        parent_id="INV-001",
        line_start=15,
        line_end=45,
    )

    # L2: Concrete schema child of L1
    hdm.store_document(
        doc_id="SCH-SNAP-INC",
        tier=HDMTier.L2_CONCRETE,
        title="Gross Income Table 2026",
        content="Household size 1: $1,632 / month gross limit.",
        source_uri="tribune/corpus/programs/tables/snap_2026.json",
        parent_id="CTR-SNAP",
        line_start=1,
        line_end=10,
    )

    # Query tiers
    assert len(hdm.query_tier(HDMTier.L0_GLOBAL)) == 1
    assert len(hdm.query_tier(HDMTier.L1_MODULE)) == 1
    assert len(hdm.query_tier(HDMTier.L2_CONCRETE)) == 1

    # Drill down: L0 -> [L1]
    children_l0 = hdm.drill_down("INV-001")
    assert len(children_l0) == 1
    assert children_l0[0].doc_id == "CTR-SNAP"

    # Rollup: L2 -> L1 -> L0
    parent_l2 = hdm.rollup("SCH-SNAP-INC")
    assert parent_l2 is not None
    assert parent_l2.doc_id == "CTR-SNAP"

    parent_l1 = hdm.rollup(parent_l2.doc_id)
    assert parent_l1 is not None
    assert parent_l1.doc_id == "INV-001"


def test_citation_lock_harness_inclusion_enforcement():
    """Verify CitationLockHarness enforces C ⊆ O and returns a = ⊥ on ungrounded claims."""
    audit_logger = get_security_audit_logger()
    audit_logger.clear()

    harness = CitationLockHarness(case_id="case_citation_test")

    # Load open validated evidence pointers O into context
    ptr1 = ProvenancePointer.create("tribune/corpus/snap.py", commit_hash="a1b2c3d4", line_start=10, line_end=25)
    ptr2 = ProvenancePointer.create("tribune/corpus/medicaid.py", commit_hash="e5f6a7b8", line_start=50, line_end=60)
    harness.register_evidence([ptr1, ptr2])

    assert len(harness.get_open_pointers()) == 2

    # 1. Valid grounded assertion: C = {ptr1} ⊆ O
    valid_output = {
        "summary": "Eligible under SNAP guidelines.",
        "citations": [ptr1.pointer_id],
    }
    is_valid, out = harness.validate_assertion(valid_output, is_actionable=True)
    assert is_valid is True
    assert out == valid_output

    # 2. Ungrounded assertion: C contains unauthorized pointer (C ⊈ O)
    unauthorized_output = {
        "summary": "Modified housing rule arbitrarily.",
        "citations": ["unauthorized/external/rule.py@deadbeef:1-50"],
    }
    is_valid, res = harness.validate_assertion(unauthorized_output, is_actionable=True)
    assert is_valid is False
    assert isinstance(res, AbstentionSignal)
    assert res.is_abstention is True
    assert res.diagnostic_error == "UNGROUNDED_ASSERTION_VIOLATION"
    assert "unauthorized/external/rule.py" in res.unauthorized_citations[0]

    # Check security audit event
    events = audit_logger.get_events(event_type=SecurityEventType.UNGROUNDED_ASSERTION_VIOLATION)
    assert len(events) >= 1

    # 3. Actionable assertion with empty citations (C = ∅) must also be rejected
    empty_citation_output = {"summary": "Actionable decision without evidence."}
    is_valid_empty, res_empty = harness.validate_assertion(empty_citation_output, is_actionable=True)
    assert is_valid_empty is False
    assert isinstance(res_empty, AbstentionSignal)
