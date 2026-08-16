"""Comprehensive integration and unit test suite for the four core architectural pillars:

1. Subagent Worktree Isolation (MemoryPartition, SubagentMemoryPartition, AsyncDAGRunner)
2. Two-Stage Evaluator-Generator Pattern (Pass 1 Milestone Verification, ActionGate gating, Pass 2 Notice Generation)
3. Atomic Transaction Logging & Crash Recovery (CheckpointManager, Router recovery plan, resume_from_checkpoint)
4. Quantization Parity & Cost Benchmarking (Decision parity metrics, Pareto frontier computation)
5. End-to-End Integration Scenario
"""

import asyncio
import os
import pytest

from tribune.agents.verifier import VerificationReport, Verifier
from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.config import TribuneSettings
from tribune.corpus.rule_store import LocalRuleStore
from tribune.eval.costmodel import CostModel, ParetoPoint
from tribune.eval.quant_sensitivity.backends import smoke_ladder
from tribune.eval.quant_sensitivity.ladder import run_ladder
from tribune.governance.action_gate import ActionBlocked, ActionGate, PreConditionError
from tribune.governance.audit import AuditLog, CheckpointManager
from tribune.governance.disclosure import generate_determination_notice
from tribune.memory.partitions import AccessDenied, CasePartition, PartitionManager, SubagentMemoryPartition
from tribune.orchestration.dag import AsyncDAGRunner, DAG, DAGRunner, Task, TaskStatus
from tribune.orchestration.pipeline import CasePipeline
from tribune.orchestration.router import RecoveryPlan, Router
from tribune.types import (
    Assessment,
    Citation,
    CriterionOutcome,
    CriterionResult,
    EligibilityStatus,
    Evidence,
    EvidenceType,
    IngestMethod,
    ProgramId,
    ProgramOutcome,
    Provenance,
    RecommendedAction,
    SMState,
    SyntheticCase,
)


# =========================================================================== #
# Pillar 1: Subagent Worktree Isolation Tests
# =========================================================================== #

def test_subagent_memory_partition_isolation():
    """Verify that distinct subagent partitions are completely isolated from each other."""
    mgr = PartitionManager()
    main_partition = mgr.open("case_pillar_1")
    
    # Open isolated subagent partitions for Medicaid and SNAP
    medicaid_subagent = mgr.open_subagent("case_pillar_1", "subagent_medicaid", program=ProgramId.MEDICAID)
    snap_subagent = mgr.open_subagent("case_pillar_1", "subagent_snap", program=ProgramId.SNAP)

    medicaid_subagent.write("evidence", "magi_income", "Note", {"monthly_income": 1200.0})
    snap_subagent.write("evidence", "gross_income", "Note", {"gross_income": 1400.0})

    # Subagents can read their own data
    assert medicaid_subagent.read("evidence", "magi_income") is not None
    assert snap_subagent.read("evidence", "gross_income") is not None

    # Subagents cannot see each other's keys
    assert medicaid_subagent.read("evidence", "gross_income") is None
    assert snap_subagent.read("evidence", "magi_income") is None

    # Direct cross-subagent read attempt raises AccessDenied
    with pytest.raises(AccessDenied):
        medicaid_subagent.try_read_other_subagent(snap_subagent, "evidence", "gross_income")


def test_subagent_partition_fork_and_merge():
    """Verify subagent partition branching (fork) and safe reconciliation (merge)."""
    mgr = PartitionManager()
    main_partition = mgr.open("case_fork_merge")
    main_partition.write("evidence", "base_doc", "Doc", {"text": "verified residency"})

    subagent = mgr.open_subagent(
        "case_fork_merge",
        "subagent_eval",
        program=ProgramId.SNAP,
        initial_records=main_partition.read_all("evidence"),
    )
    assert subagent.read("evidence", "base_doc") is not None

    # Subagent writes an assessment
    subagent.write("assessment", "snap_verdict", "Assessment", {"status": "likely_eligible"})

    # Main partition does not have it yet
    assert main_partition.read("assessment", "snap_verdict") is None

    # Merge subagent conclusions into main partition
    merged_count = mgr.merge_subagent(subagent, main_partition)
    assert merged_count >= 2
    assert main_partition.read("assessment", "snap_verdict") is not None


@pytest.mark.asyncio
async def test_async_dag_runner_concurrent_fanout():
    """Verify AsyncDAGRunner executes independent program tasks concurrently in topological waves."""
    dag = DAG()
    dag.add(Task(task_id="gather", kind="gather"))
    dag.add(Task(task_id="assess:medicaid", kind="assess", deps=["gather"], program=ProgramId.MEDICAID))
    dag.add(Task(task_id="assess:snap", kind="assess", deps=["gather"], program=ProgramId.SNAP))
    dag.add(Task(task_id="assess:housing", kind="assess", deps=["gather"], program=ProgramId.HOUSING))

    executed_order: list[str] = []
    concurrency_counter = 0
    max_concurrency = 0

    async def async_executor(task: Task):
        nonlocal concurrency_counter, max_concurrency
        concurrency_counter += 1
        max_concurrency = max(max_concurrency, concurrency_counter)
        await asyncio.sleep(0.05)  # Simulate non-blocking async work
        executed_order.append(task.task_id)
        concurrency_counter -= 1
        return f"result_{task.task_id}"

    runner = AsyncDAGRunner()
    results = await runner.run_async(dag, async_executor)

    assert len(results) == 4
    assert executed_order[0] == "gather"
    # Medicaid, SNAP, and Housing were executed in parallel wave 1 (max_concurrency > 1)
    assert max_concurrency >= 2
    assert set(executed_order[1:]) == {"assess:medicaid", "assess:snap", "assess:housing"}


# =========================================================================== #
# Pillar 2: Two-Stage Evaluator-Generator Pattern Tests
# =========================================================================== #

def test_two_stage_pass1_evaluation_and_pass2_notice_generation():
    """Verify Pass 1 milestone verification certifies findings before Pass 2 drafts notice."""
    pipeline = CasePipeline()
    prov = Provenance(source_doc_id="doc1", ingest_method=IngestMethod.MANUAL, anonymized=True, content_hash="hash123")
    evidence = [
        Evidence(evidence_id="ev1", type=EvidenceType.MONTHLY_INCOME, value=1000.0, provenance=prov),
        Evidence(evidence_id="ev2", type=EvidenceType.HOUSEHOLD_SIZE, value=2.0, provenance=prov),
        Evidence(evidence_id="ev3", type=EvidenceType.RESIDENT, value=True, provenance=prov),
        Evidence(evidence_id="ev4", type=EvidenceType.CITIZENSHIP_STATUS, value="citizen", provenance=prov),
        Evidence(evidence_id="ev5", type=EvidenceType.LIQUID_ASSETS, value=500.0, provenance=prov),
    ]

    # Propose candidate assessment with full statutory coverage
    assessment, _ = pipeline.proposer.assess(
        case_id="case_2stage",
        jurisdiction="EX",
        program=ProgramId.SNAP,
        evidence=evidence,
        k=8,
        attempt=1,
    )

    verifier = pipeline.verifier

    # Pass 1: Independent statutory evaluation & certification
    is_certified, report = verifier.evaluate_and_certify(assessment, evidence, "EX")
    assert isinstance(report, VerificationReport)
    assert is_certified is True
    assert len(report.missing_citations) == 0
    assert len(report.unsupported_claims) == 0

    outcome = ProgramOutcome(
        program=ProgramId.SNAP,
        assessment=assessment,
        final_state=SMState.DONE,
    )

    # Pass 2: ActionGate permits formal determination notice generation
    gate = ActionGate()
    notice = generate_determination_notice(outcome, verification_report=report, action_gate=gate)
    assert "OFFICIAL DETERMINATION & DISCLOSURE NOTICE — SNAP (FOOD ASSISTANCE)" in notice
    assert "7 CFR 273" in notice
    assert "Passed independent statutory re-derivation" in notice or "Passed Pass 1 statutory milestone verification" in notice
    assert "FAIR HEARING APPEAL RIGHTS" in notice


def test_action_gate_blocks_uncertified_notice_generation():
    """Verify ActionGate strictly blocks notice generation if Pass 1 verification failed."""
    invalid_citation = Citation(
        citation_id="invalid_fake_citation_999",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        title="Fake Law",
        source="Fake Code § 999",
        text="Fake rule",
    )
    crit = CriterionResult(
        criterion_id="snap_gross_income",
        description="Gross income test",
        outcome=CriterionOutcome.SATISFIED,
        required=True,
        citation_ids=["invalid_fake_citation_999"],
    )
    assessment = Assessment(
        assessment_id="case_invalid:snap:a1",
        case_id="case_invalid",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        status=EligibilityStatus.LIKELY_ELIGIBLE,
        criteria=[crit],
        citations=[invalid_citation],
        recommended_action=RecommendedAction.PREPARE_APPLICATION,
        self_confidence=0.90,
        rationale="Unverified claim with invalid citation",
    )

    prov = Provenance(source_doc_id="doc1", ingest_method=IngestMethod.MANUAL, anonymized=True, content_hash="hash123")
    evidence = [Evidence(evidence_id="ev1", type=EvidenceType.MONTHLY_INCOME, value=1000.0, provenance=prov)]

    pipeline = CasePipeline()
    verifier = pipeline.verifier

    # Pass 1 evaluates candidate
    is_certified, report = verifier.evaluate_and_certify(assessment, evidence, "EX")
    assert is_certified is False
    assert len(report.missing_citations) > 0

    outcome = ProgramOutcome(
        program=ProgramId.SNAP,
        assessment=assessment,
        final_state=SMState.DONE,
    )

    gate = ActionGate()
    # Pass 2 must be BLOCKED
    with pytest.raises(PreConditionError, match="Evaluation failed Pass 1 milestone verification"):
        generate_determination_notice(outcome, verification_report=report, action_gate=gate)


# =========================================================================== #
# Pillar 3: Atomic Transaction Logging & Crash Recovery Tests
# =========================================================================== #

def test_atomic_checkpoint_save_and_restore(tmp_path):
    """Verify atomic state-write and crash recovery loading."""
    checkpoint_file = str(tmp_path / "last_run.json")

    dag = DAG()
    dag.add(Task(task_id="gather", kind="gather", status=TaskStatus.COMPLETED))
    dag.add(Task(task_id="assess:medicaid", kind="assess", deps=["gather"], program=ProgramId.MEDICAID))
    dag.add(Task(task_id="assess:snap", kind="assess", deps=["gather"], program=ProgramId.SNAP))

    audit = AuditLog()
    audit.append("case_crash_1", SMState.PLAN, "navigator", "plan dag")
    audit.append("case_crash_1", SMState.GATHER, "navigator", "gathered evidence")

    saved_path = CheckpointManager.save_checkpoint(
        case_id="case_crash_1",
        jurisdiction="EX",
        dag_dict=dag.to_dict(),
        completed_task_ids=["gather"],
        outcomes_data=[],
        memory_snapshot={"evidence": [{"case_id": "case_crash_1", "kind": "evidence", "key": "k1", "record_type": "Doc", "payload": {"val": 1}}]},
        audit_records=audit.records("case_crash_1"),
        path=checkpoint_file,
    )
    assert os.path.exists(saved_path)

    router = Router()
    recovery_plan = router.inspect_checkpoint(checkpoint_file)
    assert recovery_plan is not None
    assert recovery_plan.case_id == "case_crash_1"
    assert recovery_plan.completed_task_ids == ["gather"]
    assert "assess:medicaid" in recovery_plan.pending_task_ids
    assert "assess:snap" in recovery_plan.pending_task_ids
    assert len(recovery_plan.restored_audit) == 2

    # Clear checkpoint test
    assert CheckpointManager.clear_checkpoint(checkpoint_file) is True
    assert not os.path.exists(checkpoint_file)


def test_pipeline_crash_and_resume_cycle(tmp_path):
    """Simulate a pipeline crash after gather and verify resumption executes remaining tasks."""
    checkpoint_file = str(tmp_path / "last_run.json")

    gen = SyntheticCaseGenerator(seed=42)
    case = gen.build_case(
        case_id="test_crash_resume_1",
        jurisdiction="EX",
        overrides={"monthly_income": 1100.0, "household_size": 2},
        target_programs=[ProgramId.MEDICAID, ProgramId.SNAP],
    )

    pipeline = CasePipeline(checkpoint_path=checkpoint_file)

    # 1. Simulate partial execution (gather completed, then crash occurs)
    dag = pipeline.navigator.plan(case)
    partition = pipeline.partitions.open(case.case_id)
    from tribune.memory.consolidation import MemoryConsolidator
    consolidator = MemoryConsolidator(partition)
    consolidator.store_evidence(case.evidence)

    pipeline._save_milestone_checkpoint(
        case_id=case.case_id,
        jurisdiction=case.jurisdiction,
        dag=dag,
        completed_task_ids=["gather"],
        outcomes=[],
        partition=partition,
    )
    assert os.path.exists(checkpoint_file)

    # 2. Resume execution from checkpoint using a new pipeline instance
    resumed_pipeline = CasePipeline(checkpoint_path=checkpoint_file)
    result = resumed_pipeline.resume_from_checkpoint(checkpoint_path=checkpoint_file, case=case)

    # Verify that all program outcomes completed successfully
    assert result.case_id == case.case_id
    assert len(result.outcomes) == 2
    programs = {o.program for o in result.outcomes}
    assert programs == {ProgramId.MEDICAID, ProgramId.SNAP}
    # Checkpoint should be cleared after successful resumption
    assert not os.path.exists(checkpoint_file)


# =========================================================================== #
# Pillar 4: Quantization Parity & Cost Benchmarking Tests
# =========================================================================== #

def test_cost_model_pareto_frontier():
    """Verify Pareto frontier computation across quant models."""
    cost_model = CostModel.load()

    points_data = [
        {"label": "fp16_cloud", "backend_id": "google:gemini-3.7-flash", "cost_per_1k": 5.00, "accuracy": 0.98, "parity_score": 0.98},
        {"label": "q8_local", "backend_id": "selfhost:vllm", "cost_per_1k": 1.20, "accuracy": 0.97, "parity_score": 0.97},
        {"label": "q4_local", "backend_id": "selfhost:llama.cpp", "cost_per_1k": 0.60, "accuracy": 0.95, "parity_score": 0.95},
        {"label": "q2_inferior", "backend_id": "selfhost:llama.cpp", "cost_per_1k": 0.70, "accuracy": 0.80, "parity_score": 0.75},  # Dominated by q4_local
        {"label": "iq1_degraded", "backend_id": "selfhost:llama.cpp", "cost_per_1k": 0.20, "accuracy": 0.60, "parity_score": 0.55},
    ]

    frontier = cost_model.compute_pareto_frontier(points_data, reference_label="fp16_cloud")
    assert len(frontier) == 5

    by_label = {p.label: p for p in frontier}
    assert by_label["fp16_cloud"].is_pareto_optimal is True
    assert by_label["q8_local"].is_pareto_optimal is True
    assert by_label["q4_local"].is_pareto_optimal is True
    assert by_label["iq1_degraded"].is_pareto_optimal is True
    # q2_inferior is dominated by q4_local (higher cost, lower accuracy)
    assert by_label["q2_inferior"].is_pareto_optimal is False
    assert "q4_local" in by_label["q2_inferior"].dominated_by
    assert by_label["q4_local"].cost_savings_pct > 80.0


def test_quant_ladder_evaluates_parity_metrics():
    """Verify quant ladder computes citation precision, recall, FPR, FNR, and Pareto points."""
    gen = SyntheticCaseGenerator(seed=101)
    cases = gen.generate_eval_set(n_per_program=2, ambiguous_ratio=0.2)

    ladder = smoke_ladder()
    result = run_ladder(ladder, cases=cases)

    assert len(result.rungs) == 2
    ref_rung = result.rungs[0]
    q2_rung = result.rungs[1]

    # Parity metrics populated
    assert 0.0 <= ref_rung.citation_precision <= 1.0
    assert 0.0 <= ref_rung.citation_recall <= 1.0
    assert 0.0 <= ref_rung.false_positive_rate <= 1.0
    assert 0.0 <= ref_rung.false_negative_rate <= 1.0
    assert not (q2_rung.kappa_vs_reference != q2_rung.kappa_vs_reference)  # Not NaN
    assert len(result.pareto_frontier) == 2


# =========================================================================== #
# Pillar 5: Comprehensive End-to-End Integration Test
# =========================================================================== #

@pytest.mark.asyncio
async def test_end_to_end_concurrent_multibenefit_crash_and_verified_notice(tmp_path):
    """Full End-to-End Test:
    1. Concurrent multi-benefit run (Medicaid + SNAP).
    2. Simulated crash & resumption cycle via .tribune/last_run.json.
    3. Two-stage verifier pass checking statutory rules before drafting formal determination notice.
    """
    checkpoint_file = str(tmp_path / "e2e_last_run.json")
    gen = SyntheticCaseGenerator(seed=2026)
    case = gen.build_case(
        case_id="e2e_multibenefit_case",
        jurisdiction="EX",
        overrides={"monthly_income": 1200.0, "household_size": 2, "resident": True, "citizenship_status": "citizen"},
        target_programs=[ProgramId.MEDICAID, ProgramId.SNAP],
    )

    # 1. Run full concurrent multi-benefit pipeline asynchronously
    pipeline = CasePipeline(checkpoint_path=checkpoint_file)
    async_result = await pipeline.run_case_async(case)
    assert async_result.case_id == case.case_id
    assert len(async_result.outcomes) == 2

    # 2. Simulate crash / recovery cycle
    # Manually checkpoint after gather
    dag = pipeline.navigator.plan(case)
    partition = pipeline.partitions.open(case.case_id)
    from tribune.memory.consolidation import MemoryConsolidator
    consolidator = MemoryConsolidator(partition)
    consolidator.store_evidence(case.evidence)

    pipeline._save_milestone_checkpoint(
        case_id=case.case_id,
        jurisdiction=case.jurisdiction,
        dag=dag,
        completed_task_ids=["gather"],
        outcomes=[],
        partition=partition,
    )
    assert os.path.exists(checkpoint_file)

    # Resume from checkpoint
    recovered_pipeline = CasePipeline(checkpoint_path=checkpoint_file)
    resumed_result = recovered_pipeline.resume_from_checkpoint(checkpoint_path=checkpoint_file, case=case)
    assert resumed_result.case_id == case.case_id
    assert len(resumed_result.outcomes) == 2

    # 3. Two-stage verifier milestone verification and determination notice generation
    snap_outcome = resumed_result.outcome_for(ProgramId.SNAP)
    assert snap_outcome is not None
    assert snap_outcome.assessment is not None

    is_certified, report = recovered_pipeline.verifier.evaluate_and_certify(
        snap_outcome.assessment,
        case.evidence,
        case.jurisdiction,
    )
    assert is_certified is True
    assert report.is_certified is True

    gate = ActionGate()
    notice = generate_determination_notice(snap_outcome, verification_report=report, action_gate=gate)
    assert "OFFICIAL DETERMINATION & DISCLOSURE NOTICE" in notice
    assert "FAIR HEARING APPEAL RIGHTS" in notice
