"""CasePipeline — wires the whole core loop together.

For a synthetic (or real-intake) case it: plans the sub-task DAG (navigator),
ingests documents into access-controlled case memory (gather), then runs the
ASSESS -> VERIFY -> {PREPARE | ABSTAIN | REPLAN} state machine for each target
program. Every step is written to the audit log, and any uncaught error fail-safes
into an abstention rather than crashing the case.
"""

from __future__ import annotations

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
from ..governance.audit import AuditLog
from ..ingestion.base import make_doc_ingest
from ..instrumentation.tracing import init_tracing, span
from ..instrumentation.usage import UsageRecorder
from ..memory.consolidation import MemoryConsolidator, extract_statutory_constraints
from ..memory.partitions import PartitionManager
from ..providers.base import get_provider_for_role
from ..types import CaseRunResult, Evidence, ProgramOutcome, SMState, SyntheticCase
from .dag import DAGRunner, Task
from .router import Router
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
    def __init__(self, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()
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
        self.sm = CaseStateMachine(
            proposer=self.proposer,
            verifier=self.verifier,
            calibrator=self.calibrator,
            preparer=self.preparer,
            router=self.router,
            audit=self.audit,
            recorder=self.recorder,
        )

    def run_case(self, case: SyntheticCase) -> CaseRunResult:
        with span("run_case", case_id=case.case_id, jurisdiction=case.jurisdiction):
            start_t = time.perf_counter()

            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            # Build static prefix containing instructions and statutory constraint anchors for KV cache hit rate
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
                    return None

                program = task.program
                assert program is not None
                self.recorder.start_task(program)
                current_evidence = trajectory.latest_evidence or case.evidence
                try:
                    outcome = self.sm.run_program(
                        case.case_id, case.jurisdiction, program, current_evidence, consolidator
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
                # Usage attaches to every completed task — abstentions included:
                # a correct abstention is a completed outcome at its actual cost.
                outcome.usage = self.recorder.finish_task()
                result.outcomes.append(outcome)
                return outcome

            self.runner.run(dag, executor)
            result.total_latency_ms = (time.perf_counter() - start_t) * 1000.0
            result.audit = self.audit.records(case.case_id)
            return result

    def run_cases(self, cases: list[SyntheticCase]) -> list[CaseRunResult]:
        return [self.run_case(c) for c in cases]

