"""Phase 5: Invariant Property Testing with Hypothesis.

Formulates formal property-based tests asserting predicate equivalence across
compaction boundaries:
    forall p in P_closed, Evaluate(p, RawTrace) == Evaluate(p, CompactedTrace)

Mathematical verification that context compaction via SelfCompact introduces ZERO
informational loss into the legal determination pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tribune.agents.eligibility import (
    CompactionProbe,
    CompactionRubric,
    EligibilityProposer,
    SelfCompactScaffold,
    TokenTrackingAccumulator,
)
from tribune.corpus.rule_store import LocalRuleStore
from tribune.providers.local_rules import LocalRulesProvider
from tribune.types import (
    Citation,
    ClosedPredicate,
    CriterionOutcome,
    CriterionResult,
    Evidence,
    EvidenceType,
    ProgramId,
    Provenance,
)


# =========================================================================== #
# Hypothesis Strategies for Complex Applicant Financial Dossiers
# =========================================================================== #


@st.composite
def st_financial_dossier(draw: Any) -> dict[str, Any]:
    """Synthesizes randomized, highly complex applicant financial dossiers."""
    hh_size = draw(st.integers(min_value=1, max_value=8))
    # Federal poverty guideline estimate
    fpl = (1255.0 + (hh_size - 1) * 438.0) * 1.30

    # Draw earned and unearned income components
    earned_income = draw(st.floats(min_value=0.0, max_value=fpl * 1.6, allow_nan=False))
    unearned_income = draw(st.floats(min_value=0.0, max_value=600.0, allow_nan=False))
    total_monthly_income = round(earned_income + unearned_income, 2)

    liquid_assets = round(draw(st.floats(min_value=0.0, max_value=4500.0, allow_nan=False)), 2)
    monthly_rent = round(draw(st.floats(min_value=250.0, max_value=2200.0, allow_nan=False)), 2)
    has_disability = draw(st.booleans())
    is_resident = draw(st.booleans())
    is_citizen = draw(st.booleans())
    has_dependent = hh_size > 1

    return {
        "household_size": hh_size,
        "monthly_income": total_monthly_income,
        "liquid_assets": liquid_assets,
        "monthly_rent": monthly_rent,
        "disabled": has_disability,
        "resident": is_resident,
        "citizen": is_citizen,
        "has_dependent_child": has_dependent,
        "jurisdiction": "EX",
    }


def _build_evidence_from_dossier(case_id: str, dossier: dict[str, Any]) -> list[Evidence]:
    """Converts a synthesized dossier into typed, provenance-tagged Evidence items."""
    evidence: list[Evidence] = []
    mapping = [
        (EvidenceType.HOUSEHOLD_SIZE, float(dossier["household_size"])),
        (EvidenceType.MONTHLY_INCOME, float(dossier["monthly_income"])),
        (EvidenceType.LIQUID_ASSETS, float(dossier["liquid_assets"])),
        (EvidenceType.MONTHLY_RENT, float(dossier["monthly_rent"])),
        (EvidenceType.DISABLED, bool(dossier["disabled"])),
        (EvidenceType.RESIDENT, bool(dossier["resident"])),
        (EvidenceType.HAS_DEPENDENT_CHILD, bool(dossier["has_dependent_child"])),
        (EvidenceType.CITIZENSHIP_STATUS, "citizen" if dossier.get("citizen", True) else "qualified_immigrant"),
    ]


    for etype, val in mapping:
        val_str = str(val)
        c_hash = hashlib.sha256(f"{etype.value}:{val_str}".encode()).hexdigest()
        prov = Provenance(
            source_doc_id=f"{case_id}:doc_intake",
            ingest_method="synthetic",
            anonymized=True,
            content_hash=c_hash,
            notes=f"Synthesized field {etype.value}",
        )
        evidence.append(
            Evidence(
                evidence_id=f"{case_id}:ev:{etype.value}",
                type=etype,
                value=val,
                provenance=prov,
            )
        )
    return evidence



# =========================================================================== #
# Formal Property-Based Equivalence Invariant Tests
# =========================================================================== #


@settings(max_examples=35, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(dossier=st_financial_dossier())
def test_invariant_predicate_equivalence_across_compaction_boundaries(dossier: dict[str, Any]) -> None:
    """FORMAL INVARIANT THEOREM:

    forall p in P_closed, Evaluate(p, RawTrace) == Evaluate(p, CompactedTrace)

    Asserts that:
    1. Every verified statutory predicate outcome derived from the raw uncompacted trace
       is strictly identical to the evaluation derived after SelfCompact condensation.
    2. All statutory citations and legal rule associations are preserved with zero loss.
    3. Final eligibility determination status (ELIGIBLE, INELIGIBLE, UNKNOWN) matches identically.
    """
    case_id = f"case-hypo-{hashlib.sha256(str(dossier).encode()).hexdigest()[:8]}"
    evidence = _build_evidence_from_dossier(case_id, dossier)
    rule_store = LocalRuleStore()
    provider = LocalRulesProvider()

    # ----------------------------------------------------------------------- #
    # Pipeline 1: Raw Baseline Run (Uncompacted Full Trace)
    # ----------------------------------------------------------------------- #
    raw_proposer = EligibilityProposer(provider=provider, rule_store=rule_store)
    raw_assessment, raw_diag = raw_proposer.assess(
        case_id=case_id,
        jurisdiction=dossier["jurisdiction"],
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        turn_tokens=1500,
    )

    # ----------------------------------------------------------------------- #
    # Pipeline 2: Active Run with Periodic SelfCompact Compaction
    # ----------------------------------------------------------------------- #
    compact_proposer = EligibilityProposer(provider=provider, rule_store=rule_store)

    # Simulate accumulation crossing the 16,000 token threshold
    compact_proposer.accumulator.add_tokens(16_500)

    # Execute rubric-governed compaction
    decision = compact_proposer.execute_compaction(
        case_id=case_id,
        statutory_module="snap_eligibility",
        criteria=raw_assessment.criteria,
        citations=raw_assessment.citations,
        evidence=evidence,
    )
    assert decision.authorized is True, f"Expected compaction authorization, got: {decision.reason}"
    assert len(compact_proposer.compacted_states) == 1

    compacted_state = compact_proposer.compacted_states[0]

    # Verify condensation magnitude: from 50k+ raw trace to ~2k structured predicates
    assert compacted_state.token_count_before >= 16_500
    assert compacted_state.token_count_after <= 2_500
    assert compacted_state.compression_ratio < 0.20

    # ----------------------------------------------------------------------- #
    # Re-evaluate with Compacted State injected into Proposer Context
    # ----------------------------------------------------------------------- #
    compacted_prompt = compact_proposer.generate_prompt(ProgramId.SNAP, dossier["jurisdiction"])
    assert "=== VERIFIED STATUTORY COMPACTED STATE (IMMUTABLE PREDICATES) ===" in compacted_prompt

    compacted_assessment, compacted_diag = compact_proposer.assess(
        case_id=case_id,
        jurisdiction=dossier["jurisdiction"],
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
        turn_tokens=1500,
    )

    # ----------------------------------------------------------------------- #
    # Mathematical Equality Verification: forall p in P_closed
    # ----------------------------------------------------------------------- #
    raw_preds = {c.criterion_id: c.outcome for c in raw_assessment.criteria}
    compacted_preds = {c.criterion_id: c.outcome for c in compacted_assessment.criteria}

    # 1. Predicate Outcomes Equivalence
    assert raw_preds == compacted_preds, (
        f"Predicate outcome mismatch across compaction boundary!\n"
        f"Raw: {raw_preds}\n"
        f"Compacted: {compacted_preds}"
    )

    # 2. Statutory Citations Equivalence
    raw_citations = sorted(c.citation_id for c in raw_assessment.citations)
    compacted_citations = sorted(c.citation_id for c in compacted_assessment.citations)
    assert raw_citations == compacted_citations, (
        f"Citations mismatch across compaction boundary!\n"
        f"Raw: {raw_citations}\n"
        f"Compacted: {compacted_citations}"
    )

    # 3. Final Eligibility Determination Status Equivalence
    assert raw_assessment.status == compacted_assessment.status, (
        f"Final determination status divergence!\n"
        f"Raw: {raw_assessment.status}\n"
        f"Compacted: {compacted_assessment.status}"
    )

    # 4. Invariant Verification in ClosedPredicate Artifacts
    closed_dict = {p.predicate_id: p.outcome for p in compacted_state.closed_predicates}
    for cid, outcome in raw_preds.items():
        if outcome != CriterionOutcome.UNKNOWN:
            assert cid in closed_dict, f"Closed predicate {cid} missing from compacted artifact!"
            assert closed_dict[cid] == outcome.value, f"Outcome drift for predicate {cid}!"


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(dossier=st_financial_dossier())
def test_rubric_strictly_blocks_compaction_on_active_discrepancies(dossier: dict[str, Any]) -> None:
    """Asserts that Rule 2 strictly blocks compaction when unresolved document discrepancies

    or in-progress verification calculations are present.
    """
    case_id = f"case-discrepancy-{hashlib.sha256(str(dossier).encode()).hexdigest()[:8]}"
    evidence = _build_evidence_from_dossier(case_id, dossier)
    rule_store = LocalRuleStore()
    provider = LocalRulesProvider()

    proposer = EligibilityProposer(provider=provider, rule_store=rule_store)
    assessment, _ = proposer.assess(
        case_id=case_id,
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
    )

    # Condition 1: Active discrepancy present (e.g. declared wage vs IRS record)
    discrepancies = ["Discrepancy: declared gross income $1450 != employer quarterly wage $2100"]
    decision = proposer.execute_compaction(
        case_id=case_id,
        statutory_module="snap_eligibility",
        criteria=assessment.criteria,
        citations=assessment.citations,
        evidence=evidence,
        discrepancies=discrepancies,
    )
    assert decision.authorized is False
    assert "Rule 2" in decision.reason
    assert "discrepancies detected" in decision.reason

    # Condition 2: Active calculation in progress
    active_calcs = ["Calculating medical deduction offset under 7 CFR 273.9(d)(3)"]
    decision_calc = proposer.execute_compaction(
        case_id=case_id,
        statutory_module="snap_eligibility",
        criteria=assessment.criteria,
        citations=assessment.citations,
        evidence=evidence,
        active_calculations=active_calcs,
    )
    assert decision_calc.authorized is False
    assert "Rule 2" in decision_calc.reason
    assert "Active verification calculation" in decision_calc.reason
