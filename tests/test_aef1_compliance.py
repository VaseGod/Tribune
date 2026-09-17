"""Tests for AEF-1 compliance reporting and checklist generation."""

import json
import os
import tempfile
import pytest
from tribune.redteam.aef_compliance import AEFChecklist, AEFComplianceEngine


def test_aef1_checklist_fields_completeness():
    engine = AEFComplianceEngine()
    checklist = engine.build_checklist(
        sandbox_mode="container",
        network_isolated=True,
    )

    assert checklist.compliance_standard == "AEF-1 (Autonomous Evaluation Framework - Tier 1)"
    assert checklist.overall_status == "CONFORMANT"

    # Verify 10 mandatory dimensions
    assert checklist.system_access_depth is not None
    assert checklist.computational_budgets is not None
    assert checklist.testing_autonomy is not None
    assert checklist.safe_harbor_execution_parameters is not None
    assert checklist.sandbox_isolation_status == "CONTAINER_ACTIVE"
    assert checklist.verifier_gate_enforcement is not None
    assert checklist.policy_gate_triggers is not None
    assert checklist.redaction_status is not None
    assert checklist.trace_integrity is not None
    assert checklist.model_routing_configuration is not None


def test_aef1_non_compliance_and_deviations_flagged():
    engine = AEFComplianceEngine()

    # Build with degraded fallback, network breach, and ledger failure
    checklist = engine.build_checklist(
        sandbox_mode="local_fallback",
        network_isolated=False,
        ledger_info={"chain_validated": False},
        budget_stats={"budget_breached": True},
    )

    assert checklist.overall_status == "NON_CONFORMANT"
    assert len(checklist.findings_and_deviations) >= 3
    assert any("Container isolation not enforced" in d for d in checklist.findings_and_deviations)
    assert any("Network egress was not disabled" in d for d in checklist.findings_and_deviations)
    assert any("failed cryptographic verification" in d for d in checklist.findings_and_deviations)


def test_aef1_report_exports_json_and_markdown():
    engine = AEFComplianceEngine()
    checklist = engine.build_checklist(sandbox_mode="container", network_isolated=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path, md_path = engine.export_reports(checklist, tmp_dir)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        # Validate JSON format
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["compliance_standard"].startswith("AEF-1")
            assert data["overall_status"] == "CONFORMANT"

        # Validate Markdown format
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
            assert "# AEF-1 Audit Conformance Report" in md_text
            assert "System Access Depth" in md_text
            assert "Trace Integrity" in md_text
