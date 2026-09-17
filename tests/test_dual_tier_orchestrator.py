"""Tests for dual-tier model orchestration and task classification."""

import pytest
from tribune.casegen.simulation import (
    DualTierSimulationOrchestrator,
    LeadOrchestrator,
    WorkerOrchestrator,
)
from tribune.casegen.task_classifier import (
    DeterministicTaskClassifier,
    TaskComplexityClass,
    TaskWorkflowCategory,
)
from tribune.inference.registry import MockOfflineProvider


def test_task_classifier_routes_administrative_to_worker():
    classifier = DeterministicTaskClassifier()
    decision = classifier.classify("Please verify the notice filing date and draft docket entry.")

    assert decision.task_class == TaskComplexityClass.STANDARD_ADMINISTRATIVE
    assert decision.selected_tier == "worker"
    assert decision.selected_model == "deepseek-v4.1-flash"
    assert decision.expected_token_budget == 2_000


def test_task_classifier_routes_complex_statutory_to_lead():
    classifier = DeterministicTaskClassifier()
    decision = classifier.classify(
        "Analyze constitutional standing under Article III and multi-party liability under Chevron statutory interpretation.",
        citations_count=6,
        party_count=3,
    )

    assert decision.task_class == TaskComplexityClass.COMPLEX_STATUTORY
    assert decision.selected_tier == "lead"
    assert decision.selected_model == "gpt-4o"
    assert decision.expected_token_budget == 8_000
    assert decision.complexity_score >= 0.40


def test_task_classifier_escalates_on_prior_verifier_failures():
    classifier = DeterministicTaskClassifier()
    # Simple administrative text that would normally be worker tier
    decision_normal = classifier.classify("Notice date verification", prior_verifier_failures=0)
    assert decision_normal.selected_tier == "worker"

    # With prior verifier failure -> escalates to lead tier!
    decision_escalated = classifier.classify("Notice date verification", prior_verifier_failures=2)
    assert decision_escalated.selected_tier == "lead"
    assert "Escalation from 2 prior verifier gate rejections" in decision_escalated.routing_reason


def test_dual_tier_orchestrator_dispatch_and_audit_logging():
    provider = MockOfflineProvider(canned_response="Draft document complete.")
    lead = LeadOrchestrator(provider=provider, model="gpt-4o")
    worker = WorkerOrchestrator(provider=provider, model="deepseek-v4.1-flash")
    orchestrator = DualTierSimulationOrchestrator(lead_orchestrator=lead, worker_orchestrator=worker)

    # 1. Dispatch administrative task
    res_admin = orchestrator.dispatch_task(
        task_id="admin_001",
        task_description="Complete standard intake form for SNAP application.",
    )
    assert res_admin["tier"] == "worker"
    assert res_admin["model"] == "deepseek-v4.1-flash"

    # 2. Dispatch complex task
    res_complex = orchestrator.dispatch_task(
        task_id="complex_001",
        task_description="Synthesize appellate brief on constitutional standing.",
        citations_count=5,
    )
    assert res_complex["tier"] == "lead"
    assert res_complex["model"] == "gpt-4o"

    # 3. Test explicit routing override
    res_override = orchestrator.dispatch_task(
        task_id="override_001",
        task_description="Complete simple form.",
        force_tier="lead",
    )
    assert res_override["tier"] == "lead"

    # 4. Verify audit logs
    assert len(orchestrator.audit_log) == 3
    assert orchestrator.audit_log[0]["task_id"] == "admin_001"
    assert orchestrator.audit_log[0]["selected_tier"] == "worker"
    assert orchestrator.audit_log[1]["task_id"] == "complex_001"
    assert orchestrator.audit_log[1]["selected_tier"] == "lead"
