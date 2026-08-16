"""CasePipeline — wires the whole core loop together.

For a synthetic (or real-intake) case it: plans the sub-task DAG (navigator),
ingests documents into access-controlled case memory (gather), then runs the
ASSESS -> VERIFY -> {PREPARE | ABSTAIN | REPLAN} state machine for each target
program. Every step is written to the audit log, and any uncaught error fail-safes
into an abstention rather than crashing the case.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..abstention.calibration import Calibrator
from ..agents.eligibility import EligibilityProposer
from ..agents.navigator import Navigator
from ..agents.preparer import Preparer
from ..agents.verifier import Verifier
from ..config import TribuneSettings, get_settings
from ..corpus.rule_store import make_rule_store
from ..eval.costmodel import CostModel
from ..governance.action_gate import ActionGate
from ..governance.audit import AuditLog, CheckpointManager
from ..ingestion.base import make_doc_ingest
from ..instrumentation.tracing import init_tracing, span
from ..instrumentation.usage import UsageRecorder
from ..memory.consolidation import MemoryConsolidator, extract_statutory_constraints
from ..memory.partitions import CasePartition, PartitionManager, SubagentMemoryPartition
from ..providers.base import get_provider_for_role
from ..types import CaseRunResult, Evidence, ProgramOutcome, SMState, SyntheticCase
from .dag import AsyncDAGRunner, DAGRunner, Task
from .router import RecoveryPlan, Router
from .state_machine import CaseStateMachine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TrajectoryFrame:
    """An immutable record of one step within the case trajectory."""
    frame_id: str
    state: SMState
    agent: str
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class TrajectoryBuffer:
    """Immutable append-only trajectory buffer optimizing KV-cache prefix stability.

    Preserves static prefix tokens (system instructions, static definitions, statutory bounds)
    anchored at index 0 across multi-turn agent turns to maximize KV-cache hit rates.
    """
    static_prefix: str
    frames: tuple[TrajectoryFrame, ...] = ()

    def append(
        self, state: SMState, agent: str, action: str, data: dict[str, Any] | None = None
    ) -> TrajectoryBuffer:
        new_frame = TrajectoryFrame(
            frame_id=f"frame_{len(self.frames) + 1}",
            state=state,
            agent=agent,
            action=action,
            data=dict(data) if data else {},
        )
        return TrajectoryBuffer(
            static_prefix=self.static_prefix,
            frames=(*self.frames, new_frame),
        )

    @property
    def latest_evidence(self) -> list[Evidence]:
        for frame in reversed(self.frames):
            if "evidence" in frame.data and isinstance(frame.data["evidence"], list):
                return list(frame.data["evidence"])
        return []


def _make_recorder(settings: TribuneSettings) -> UsageRecorder:
    pricing_date = date.fromisoformat(settings.pricing_date) if settings.pricing_date else None
    cost_model = CostModel.load(settings.pricing_path or None)
    return UsageRecorder(cost_model=cost_model, pricing_date=pricing_date)


class CasePipeline:
    def __init__(
        self,
        settings: TribuneSettings | None = None,
        checkpoint_path: str = ".tribune/last_run.json",
    ) -> None:
        self.settings = settings or get_settings()
        self.checkpoint_path = checkpoint_path
        init_tracing()

        rule_store = make_rule_store(self.settings)
        self.recorder = _make_recorder(self.settings)
        proposer_provider = get_provider_for_role("proposer", self.settings, self.recorder)
        verifier_provider = get_provider_for_role("verifier", self.settings, self.recorder)

        self.proposer = EligibilityProposer(proposer_provider, rule_store)
        self.verifier = Verifier(verifier_provider, rule_store)
        self.calibrator = Calibrator(threshold=self.settings.abstention_threshold)
        self.preparer = Preparer(ActionGate())
        self.router = Router()
        self.navigator = Navigator(rule_store)
        self.partitions = PartitionManager()
        self.ingest = make_doc_ingest(self.settings)
        self.audit = AuditLog()
        self.runner = DAGRunner()
        self.async_runner = AsyncDAGRunner()
        self.sm = CaseStateMachine(
            proposer=self.proposer,
            verifier=self.verifier,
            calibrator=self.calibrator,
            preparer=self.preparer,
            router=self.router,
            audit=self.audit,
            recorder=self.recorder,
        )

    def _save_milestone_checkpoint(
        self,
        case_id: str,
        jurisdiction: str,
        dag: Any,
        completed_task_ids: list[str],
        outcomes: list[ProgramOutcome],
        partition: CasePartition,
    ) -> None:
        """Atomically persist pipeline checkpoint to disk for crash recovery."""
        outcomes_data = [o.model_dump(mode="json") for o in outcomes]
        CheckpointManager.save_checkpoint(
            case_id=case_id,
            jurisdiction=jurisdiction,
            dag_dict=dag.to_dict(),
            completed_task_ids=completed_task_ids,
            outcomes_data=outcomes_data,
            memory_snapshot=partition.snapshot(),
            audit_records=self.audit.records(case_id),
            path=self.checkpoint_path,
        )

    def run_case(self, case: SyntheticCase) -> CaseRunResult:
        with span("run_case", case_id=case.case_id, jurisdiction=case.jurisdiction):
            start_t = time.perf_counter()

            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            constraints = extract_statutory_constraints(case.evidence)
            static_prefix = (
                f"TRIBUNE System Instructions (Static Prefix v1.0)\n"
                f"Jurisdiction: {case.jurisdiction} | Case: {case.case_id}\n"
                f"{constraints.to_system_header()}"
            )
            trajectory = TrajectoryBuffer(static_prefix=static_prefix)

            self.audit.append(
                case.case_id, SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            trajectory = trajectory.append(
                SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            dag = self.navigator.plan(case)
            result = CaseRunResult(case_id=case.case_id, jurisdiction=case.jurisdiction)
            ocr_lat = 0.0
            completed_task_ids: list[str] = []

            def executor(task: Task):
                nonlocal trajectory, ocr_lat
                if task.kind == "gather":
                    self.audit.append(
                        case.case_id,
                        SMState.GATHER,
                        "navigator",
                        f"ingest {len(case.documents)} document(s) via '{self.ingest.name}' adapter",
                    )
                    g_start = time.perf_counter()
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                    ocr_lat = getattr(self.ingest, "last_latency_ms", (time.perf_counter() - g_start) * 1000.0)
                    result.ocr_latency_ms = ocr_lat
                    evidence = consolidator.consolidate_evidence()
                    trajectory = trajectory.append(
                        SMState.GATHER,
                        "navigator",
                        f"ingested {len(evidence)} evidence item(s)",
                        {"evidence": evidence},
                    )
                    completed_task_ids.append(task.task_id)
                    self._save_milestone_checkpoint(
                        case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                    )
                    return None

                program = task.program
                assert program is not None

                # Allocate dedicated, isolated SubagentMemoryPartition for worktree isolation
                subagent_id = task.subagent_id or f"subagent_{program.value}"
                evidence_recs = partition.read_all("evidence")
                subagent_partition = self.partitions.open_subagent(
                    case_id=case.case_id,
                    subagent_id=subagent_id,
                    program=program,
                    initial_records=evidence_recs,
                )
                subagent_consolidator = MemoryConsolidator(subagent_partition)

                self.recorder.start_task(program)
                current_evidence = trajectory.latest_evidence or case.evidence
                try:
                    outcome = self.sm.run_program(
                        case.case_id, case.jurisdiction, program, current_evidence, subagent_consolidator
                    )
                    outcome.ocr_latency_ms = ocr_lat
                    result.citation_latency_ms = max(result.citation_latency_ms, outcome.citation_latency_ms)
                    result.llm_latency_ms += outcome.llm_latency_ms
                    trajectory = trajectory.append(
                        outcome.final_state,
                        "state_machine",
                        f"program {program.value} completed with state {outcome.final_state.value}",
                        {"program": program.value, "abstained": outcome.abstained},
                    )
                    # Safe merge-back of subagent verified conclusions into primary partition
                    self.partitions.merge_subagent(subagent_partition, partition)
                except Exception as exc:  # fail-safe: never crash a case
                    self.audit.append(
                        case.case_id,
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        payload={"error": str(exc)[:200]},
                    )
                    trajectory = trajectory.append(
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        {"error": str(exc)[:200]},
                    )
                    outcome = ProgramOutcome(
                        program=program, abstained=True, final_state=SMState.ABSTAIN
                    )

                outcome.usage = self.recorder.finish_task()
                result.outcomes.append(outcome)
                completed_task_ids.append(task.task_id)

                self._save_milestone_checkpoint(
                    case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )
                return outcome

            self.runner.run(dag, executor)
            result.total_latency_ms = (time.perf_counter() - start_t) * 1000.0
            result.audit = self.audit.records(case.case_id)

            # Clear checkpoint upon successful complete run
            CheckpointManager.clear_checkpoint(self.checkpoint_path)
            return result

    async def run_case_async(self, case: SyntheticCase) -> CaseRunResult:
        """Asynchronous execution of case DAG with parallel subagent wave fan-out."""
        with span("run_case_async", case_id=case.case_id, jurisdiction=case.jurisdiction):
            start_t = time.perf_counter()

            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            constraints = extract_statutory_constraints(case.evidence)
            static_prefix = (
                f"TRIBUNE System Instructions (Static Prefix v1.0)\n"
                f"Jurisdiction: {case.jurisdiction} | Case: {case.case_id}\n"
                f"{constraints.to_system_header()}"
            )
            trajectory = TrajectoryBuffer(static_prefix=static_prefix)

            self.audit.append(
                case.case_id, SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            trajectory = trajectory.append(
                SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            dag = self.navigator.plan(case)
            result = CaseRunResult(case_id=case.case_id, jurisdiction=case.jurisdiction)
            ocr_lat = 0.0
            completed_task_ids: list[str] = []

            async def async_executor(task: Task):
                nonlocal trajectory, ocr_lat
                if task.kind == "gather":
                    self.audit.append(
                        case.case_id,
                        SMState.GATHER,
                        "navigator",
                        f"ingest {len(case.documents)} document(s) via '{self.ingest.name}' adapter",
                    )
                    g_start = time.perf_counter()
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                    ocr_lat = getattr(self.ingest, "last_latency_ms", (time.perf_counter() - g_start) * 1000.0)
                    result.ocr_latency_ms = ocr_lat
                    evidence = consolidator.consolidate_evidence()
                    trajectory = trajectory.append(
                        SMState.GATHER,
                        "navigator",
                        f"ingested {len(evidence)} evidence item(s)",
                        {"evidence": evidence},
                    )
                    completed_task_ids.append(task.task_id)
                    self._save_milestone_checkpoint(
                        case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                    )
                    return None

                program = task.program
                assert program is not None

                subagent_id = task.subagent_id or f"subagent_{program.value}"
                evidence_recs = partition.read_all("evidence")
                subagent_partition = self.partitions.open_subagent(
                    case_id=case.case_id,
                    subagent_id=subagent_id,
                    program=program,
                    initial_records=evidence_recs,
                )
                subagent_consolidator = MemoryConsolidator(subagent_partition)

                self.recorder.start_task(program)
                current_evidence = trajectory.latest_evidence or case.evidence
                try:
                    outcome = await asyncio.to_thread(
                        self.sm.run_program,
                        case.case_id,
                        case.jurisdiction,
                        program,
                        current_evidence,
                        subagent_consolidator,
                    )
                    outcome.ocr_latency_ms = ocr_lat
                    result.citation_latency_ms = max(result.citation_latency_ms, outcome.citation_latency_ms)
                    result.llm_latency_ms += outcome.llm_latency_ms
                    trajectory = trajectory.append(
                        outcome.final_state,
                        "state_machine",
                        f"program {program.value} completed with state {outcome.final_state.value}",
                        {"program": program.value, "abstained": outcome.abstained},
                    )
                    self.partitions.merge_subagent(subagent_partition, partition)
                except Exception as exc:
                    self.audit.append(
                        case.case_id,
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        payload={"error": str(exc)[:200]},
                    )
                    trajectory = trajectory.append(
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        {"error": str(exc)[:200]},
                    )
                    outcome = ProgramOutcome(
                        program=program, abstained=True, final_state=SMState.ABSTAIN
                    )

                outcome.usage = self.recorder.finish_task()
                result.outcomes.append(outcome)
                completed_task_ids.append(task.task_id)

                self._save_milestone_checkpoint(
                    case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )
                return outcome

            await self.async_runner.run_async(dag, async_executor)
            result.total_latency_ms = (time.perf_counter() - start_t) * 1000.0
            result.audit = self.audit.records(case.case_id)

            CheckpointManager.clear_checkpoint(self.checkpoint_path)
            return result

    def resume_from_checkpoint(
        self,
        checkpoint_path: str | None = None,
        case: SyntheticCase | None = None,
    ) -> CaseRunResult:
        """Resume pipeline execution from checkpoint after an unexpected interruption or crash."""
        path = checkpoint_path or self.checkpoint_path
        recovery_plan: RecoveryPlan | None = self.router.inspect_checkpoint(path)
        if not recovery_plan:
            if case is not None:
                return self.run_case(case)
            raise FileNotFoundError(f"No valid incomplete checkpoint found at '{path}' to resume from.")

        case_id = recovery_plan.case_id
        jurisdiction = recovery_plan.jurisdiction
        dag = recovery_plan.reconstructed_dag

        # Restore audit log
        self.audit.load_records(case_id, recovery_plan.restored_audit)

        # Restore memory partition
        partition = self.partitions.open(case_id)
        partition.restore(recovery_plan.memory_snapshot)
        consolidator = MemoryConsolidator(partition)

        result = CaseRunResult(case_id=case_id, jurisdiction=jurisdiction)
        result.outcomes.extend(recovery_plan.restored_outcomes)

        completed_task_ids = list(recovery_plan.completed_task_ids)

        def resume_executor(task: Task):
            if task.task_id in completed_task_ids:
                return None  # Skip already completed task

            if task.kind == "gather":
                if case is not None:
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                completed_task_ids.append(task.task_id)
                self._save_milestone_checkpoint(
                    case_id, jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )
                return None

            program = task.program
            assert program is not None

            subagent_id = task.subagent_id or f"subagent_{program.value}"
            evidence_recs = partition.read_all("evidence")
            subagent_partition = self.partitions.open_subagent(
                case_id=case_id,
                subagent_id=subagent_id,
                program=program,
                initial_records=evidence_recs,
            )
            subagent_consolidator = MemoryConsolidator(subagent_partition)

            self.recorder.start_task(program)
            evidence = consolidator.consolidate_evidence()
            try:
                outcome = self.sm.run_program(
                    case_id, jurisdiction, program, evidence, subagent_consolidator
                )
                self.partitions.merge_subagent(subagent_partition, partition)
            except Exception as exc:
                self.audit.append(
                    case_id,
                    SMState.ABSTAIN,
                    "navigator",
                    "fail-safe abstain on recovery due to internal error",
                    payload={"error": str(exc)[:200]},
                )
                outcome = ProgramOutcome(
                    program=program, abstained=True, final_state=SMState.ABSTAIN
                )

            outcome.usage = self.recorder.finish_task()
            result.outcomes.append(outcome)
            completed_task_ids.append(task.task_id)

            self._save_milestone_checkpoint(
                case_id, jurisdiction, dag, completed_task_ids, result.outcomes, partition
            )
            return outcome

        self.runner.run(dag, resume_executor)
        result.audit = self.audit.records(case_id)
        CheckpointManager.clear_checkpoint(path)
        return result

    def run_cases(self, cases: list[SyntheticCase]) -> list[CaseRunResult]:
        return [self.run_case(c) for c in cases]

