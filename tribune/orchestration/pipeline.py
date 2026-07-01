"""CasePipeline — wires the whole core loop together.

For a synthetic (or real-intake) case it: plans the sub-task DAG (navigator),
ingests documents into access-controlled case memory (gather), then runs the
ASSESS -> VERIFY -> {PREPARE | ABSTAIN | REPLAN} state machine for each target
program. Every step is written to the audit log, and any uncaught error fail-safes
into an abstention rather than crashing the case.
"""

from __future__ import annotations

from datetime import date

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
from ..memory.consolidation import MemoryConsolidator
from ..memory.partitions import PartitionManager
from ..providers.base import get_provider_for_role
from ..types import CaseRunResult, ProgramOutcome, SMState, SyntheticCase
from .dag import DAGRunner, Task
from .router import Router
from .state_machine import CaseStateMachine


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
        self.navigator = Navigator()
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
            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            self.audit.append(
                case.case_id, SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            dag = self.navigator.plan(case)
            result = CaseRunResult(case_id=case.case_id, jurisdiction=case.jurisdiction)
            shared: dict[str, list] = {"evidence": []}

            def executor(task: Task):
                if task.kind == "gather":
                    self.audit.append(
                        case.case_id,
                        SMState.GATHER,
                        "navigator",
                        f"ingest {len(case.documents)} document(s) via '{self.ingest.name}' adapter",
                    )
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                    shared["evidence"] = consolidator.consolidate_evidence()
                    return None

                program = task.program
                assert program is not None
                self.recorder.start_task(program)
                try:
                    outcome = self.sm.run_program(
                        case.case_id, case.jurisdiction, program, shared["evidence"], consolidator
                    )
                except Exception as exc:  # fail-safe: never crash a case
                    self.audit.append(
                        case.case_id,
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        payload={"error": str(exc)[:200]},
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
            result.audit = self.audit.records(case.case_id)
            return result

    def run_cases(self, cases: list[SyntheticCase]) -> list[CaseRunResult]:
        return [self.run_case(c) for c in cases]
