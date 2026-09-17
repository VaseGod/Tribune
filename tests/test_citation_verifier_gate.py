"""Tests for the deterministic Citation AST Verifier Gate."""

import pytest
from tribune.corpus.citation_ast import (
    CitationASTParser,
    CitationType,
    CitationVerifierGate,
    ValidationStatus,
)
from tribune.corpus.citations import verify_citation_gate
from tribune.harness.loop import HarnessLoop
from tribune.harness.state import RunStatus, StepType
from tribune.inference.registry import MockOfflineProvider


def test_valid_statutory_citations_accepted():
    gate = CitationVerifierGate()

    # 1. CFR Citation
    res_cfr = gate.verify_citation("7 C.F.R. § 273.9")
    assert res_cfr.is_valid is True
    assert res_cfr.validation_status == ValidationStatus.VALID
    assert res_cfr.citation_type == CitationType.REGULATION_CFR.value

    # 2. USC Citation
    res_usc = gate.verify_citation("42 U.S.C. § 1396a")
    assert res_usc.is_valid is True
    assert res_usc.validation_status == ValidationStatus.VALID
    assert res_usc.citation_type == CitationType.STATUTE_USC.value

    # 3. Supreme Court Reporter Citation
    res_scotus = gate.verify_citation("397 U.S. 254 (1970)")
    assert res_scotus.is_valid is True
    assert res_scotus.validation_status == ValidationStatus.VALID
    assert res_scotus.citation_type == CitationType.CASE_LAW_REPORTER.value

    # 4. Federal Reporter Citation
    res_f3d = gate.verify_citation("123 F.3d 456")
    assert res_f3d.is_valid is True
    assert res_f3d.validation_status == ValidationStatus.VALID

    # 5. State Rule Citation
    res_state = gate.verify_citation("EX-SNAP-001")
    assert res_state.is_valid is True
    assert res_state.jurisdiction == "EX"


def test_malformed_citation_rejected():
    gate = CitationVerifierGate()
    res = gate.verify_citation("totally invalid non-citation text")

    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.MALFORMED
    assert "does not match" in str(res.failure_reason)
    assert res.remediation_hint is not None


def test_unknown_reporter_rejected():
    gate = CitationVerifierGate()
    # Fake reporter "Fake. Rep."
    res = gate.verify_citation("100 Fake. Rep. 200")

    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.UNKNOWN_REPORTER
    assert "not an official federal or regional reporter" in str(res.failure_reason)


def test_invalid_jurisdiction_rejected():
    # Only allow EX and US
    gate = CitationVerifierGate(allowed_jurisdictions={"EX", "US"})
    res = gate.verify_citation("ZZ-SNAP-001")

    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.INVALID_JURISDICTION
    assert "outside allowed jurisdictions" in str(res.failure_reason)


def test_mandatory_citation_failure_stops_loop():
    provider = MockOfflineProvider(canned_response="Drafting with invalid citation: 999 FAKE 111")

    def citation_verifier_callback(state, response):
        # Extract citations or test directly
        gate_res = verify_citation_gate("999 FAKE 111")
        return {
            "is_valid": gate_res.is_valid,
            "status": gate_res.validation_status.value,
            "failure_reason": gate_res.failure_reason,
            "remediation_hint": gate_res.remediation_hint,
            "fail_fast": True,  # Mandatory gate
        }

    loop = HarnessLoop(
        provider=provider,
        verifier_callback=citation_verifier_callback,
        max_steps=5,
    )

    state = loop.run(
        task_id="task_citation_gate_fail",
        initial_messages=[{"role": "user", "content": "Draft appeal with citation"}],
    )

    # Mandatory verifier failure must halt the loop and fail the task
    assert state.status == RunStatus.FAILED
    assert any(phrase in str(state.failure_reason) for phrase in ("Verifier gate failed", "not an official", "does not match"))
    assert any(s.step_type == StepType.VERIFIER and s.outcome == "FAILED" for s in state.steps)
