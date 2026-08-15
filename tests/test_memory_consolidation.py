"""Tests for Phase 2: State-Aware Trajectory Buffers & Statutory Constraint Extraction."""

from tribune.memory.consolidation import (
    MemoryConsolidator,
    StatutoryConstraintBlock,
    extract_statutory_constraints,
)
from tribune.memory.partitions import PartitionManager
from tribune.orchestration.pipeline import TrajectoryBuffer
from tribune.types import Evidence, EvidenceType, IngestMethod, Provenance, SMState


def _make_evidence(etype: EvidenceType, val, eid="e1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        type=etype,
        value=val,
        provenance=Provenance(
            source_doc_id="doc1",
            ingest_method=IngestMethod.STRUCTURED,
            anonymized=True,
            content_hash="h123",
        ),
    )


def test_statutory_constraint_extraction_basic():
    evidence_list = [
        _make_evidence(EvidenceType.HOUSEHOLD_SIZE, 4, "e1"),
        _make_evidence(EvidenceType.MONTHLY_INCOME, 3250.0, "e2"),
        _make_evidence(EvidenceType.DAYS_SINCE_DENIAL, 14, "e3"),
        _make_evidence(EvidenceType.LIQUID_ASSETS, 1500.0, "e4"),
    ]
    constraints = extract_statutory_constraints(evidence_list)
    assert constraints.household_size == 4
    assert constraints.income_thresholds.get("monthly_income") == 3250.0
    assert constraints.countable_income == 3250.0
    assert constraints.appeal_deadlines.get("days_since_denial") == 14
    assert constraints.statutory_bounds.get("liquid_assets") == 1500.0

    header = constraints.to_system_header()
    assert "=== STATUTORY CONSTRAINT HEADER (IMMUTABLE BOUNDS) ===" in header
    assert "Household Size: 4" in header
    assert "Countable Income: $3250.00" in header


def test_deterministic_memory_retention_100_turns():
    """Verify that statutory constraints persist unmutated after 100+ turns and context compaction."""
    # Construct 100+ turn mock dialogue / log history
    history = []
    # Seed turns with initial statutory factual bounds
    history.append("User Intake: household_size=3, monthly_income=$2694.00, gross_income=$2694.00")
    history.append("Navigator: checking income_threshold=$2694.00 against statutory SNAP limits")
    history.append("Verification Officer: documented medical_deduction=$350.00 standard offset")
    history.append("Appeals Officer: 90-day statutory window applicable; appeal_deadline=90 days")

    # Add 120 intermediate dialogue turns without repeating bounds
    for i in range(120):
        history.append(f"Turn {i}: Intermediate case discussion, document verification step {i}")

    # Extract constraints prior to context compaction
    pre_constraints = extract_statutory_constraints(history)
    assert pre_constraints.household_size == 3
    assert pre_constraints.income_thresholds["gross_income"] == 2694.0
    assert pre_constraints.medical_deductions == 350.0
    assert pre_constraints.appeal_deadlines["appeal_deadline"] == "90 days"
    assert pre_constraints.jurisdictional_timebars["statutory_appeal_window_days"] == 90

    # Execute memory consolidation & context compaction
    pm = PartitionManager()
    partition = pm.open("case-100-turn-test")
    consolidator = MemoryConsolidator(partition)

    post_constraints, compacted_text = consolidator.compact_context(history)

    # Invariant: Statutory constraints must remain strictly equal and unmutated
    assert post_constraints == pre_constraints
    assert "Household Size: 3" in compacted_text
    assert "Gross Income Limit: $2694.00" in compacted_text
    assert "Medical Deductions: $350.00" in compacted_text
    assert "90 days" in compacted_text


def test_trajectory_buffer_preserves_static_prefix_and_immutability():
    static_prefix = "STATIC_PREFIX_SYSTEM_PROMPT_V1"
    buffer0 = TrajectoryBuffer(static_prefix=static_prefix)
    assert buffer0.static_prefix == static_prefix
    assert len(buffer0.frames) == 0

    ev = [_make_evidence(EvidenceType.HOUSEHOLD_SIZE, 2, "ev_hh")]
    buffer1 = buffer0.append(SMState.GATHER, "navigator", "ingested evidence", {"evidence": ev})
    assert len(buffer0.frames) == 0  # Immutability of buffer0
    assert len(buffer1.frames) == 1
    assert buffer1.static_prefix == static_prefix
    assert buffer1.latest_evidence == ev

    buffer2 = buffer1.append(SMState.ASSESS, "proposer", "assessed eligibility", {"status": "likely_eligible"})
    assert len(buffer1.frames) == 1  # Immutability of buffer1
    assert len(buffer2.frames) == 2
    assert buffer2.static_prefix == static_prefix
    assert buffer2.latest_evidence == ev
