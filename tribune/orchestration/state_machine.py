"""The per-program state machine.

    PLAN -> GATHER -> ASSESS -> VERIFY -> { PREPARE | ABSTAIN | REPLAN }

PLAN and GATHER happen once at the case level (the navigator plans the DAG and the
gather task ingests evidence). This machine runs the ASSESS/VERIFY core loop for a
single program, with REPLAN looping back to ASSESS on a recoverable verification
failure (incomplete rule coverage), and ABSTAIN as a first-class, correct-when-
uncertain terminal state. Every transition is written to the audit log.
"""

from __future__ import annotations

import time

from ..governance.audit import AuditLog
from ..instrumentation.usage import UsageRecorder
from ..memory.consolidation import MemoryConsolidator
from ..types import ProgramId, ProgramOutcome, RecommendedAction, SMState
from .router import Router


class CaseStateMachine:
    def __init__(
        self,
        proposer,
        verifier,
        calibrator,
        preparer,
        router: Router,
        audit: AuditLog,
        max_attempts: int = 3,
        recorder: UsageRecorder | None = None,
    ) -> None:
        self.proposer = proposer
        self.verifier = verifier
        self.calibrator = calibrator
        self.preparer = preparer
        self.router = router
        self.audit = audit
        self.max_attempts = max_attempts
        self.recorder = recorder

    def _turn(self, role: str) -> None:
        if self.recorder is not None:
            self.recorder.record_turn(role)

    def run_program(
        self,
        case_id: str,
        jurisdiction: str,
        program: ProgramId,
        evidence: list,
        consolidator: MemoryConsolidator,
    ) -> ProgramOutcome:
        prog_start = time.perf_counter()
        route = self.router.initial(program)
        attempt = 1
        replans = 0
        total_cit_lat = 0.0
        total_llm_lat = 0.0

        while True:
            self.audit.append(
                case_id,
                SMState.ASSESS,
                agent="eligibility_proposer",
                action=f"assess (attempt {attempt}, k={route.k}, tier={route.tier})",
                model_name=self.proposer.provider.name,
                model_version=self.proposer.provider.version,
            )
            self._turn("proposer")
            assess_start = time.perf_counter()
            assessment, diag = self.proposer.assess(
                case_id, jurisdiction, program, evidence, k=route.k, attempt=attempt
            )
            assess_lat = (time.perf_counter() - assess_start) * 1000.0
            total_llm_lat += assess_lat
            if hasattr(self.proposer.rule_store, "_retriever") and hasattr(self.proposer.rule_store._retriever, "last_latency_ms"):
                total_cit_lat += self.proposer.rule_store._retriever.last_latency_ms

            consolidator.store_assessment(assessment)
            cit_ids = [c.citation_id for c in assessment.citations]

            self.audit.append(
                case_id,
                SMState.VERIFY,
                agent="verifier",
                action=f"re-derive and verify ({assessment.status.value})",
                model_name=self.verifier.provider.name,
                model_version=self.verifier.provider.version,
                citation_ids=cit_ids,
            )
            self._turn("verifier")
            verify_start = time.perf_counter()
            verdict = self.verifier.verify(assessment, evidence, jurisdiction)
            verify_lat = (time.perf_counter() - verify_start) * 1000.0
            total_llm_lat += verify_lat

            if not verdict.approved:
                if verdict.incomplete_coverage and attempt < self.max_attempts:
                    route = self.router.escalate(program, route)
                    attempt += 1
                    replans += 1
                    self.audit.append(
                        case_id,
                        SMState.REPLAN,
                        agent="navigator",
                        action="replan: broaden retrieval to cover all required criteria",
                        payload={"missing": ", ".join(verdict.incomplete_coverage)[:200]},
                    )
                    continue
                # Verification cannot be satisfied -> abstain (fail safe).
                abst = self.calibrator.score(assessment, diag, verdict)
                self.audit.append(
                    case_id,
                    SMState.ABSTAIN,
                    agent="abstention",
                    action="abstain: assessment failed independent verification",
                    payload={"reason": abst.reason[:200]},
                )
                tot_lat = (time.perf_counter() - prog_start) * 1000.0
                return ProgramOutcome(
                    program=program,
                    assessment=assessment,
                    verdict=verdict,
                    abstention=abst,
                    final_state=SMState.ABSTAIN,
                    abstained=True,
                    replans=replans,
                    citation_latency_ms=total_cit_lat,
                    llm_latency_ms=total_llm_lat,
                    total_latency_ms=tot_lat,
                )

            # Verified. Now decide assert vs. abstain via calibrated confidence.
            abst = self.calibrator.score(assessment, diag, verdict)
            if abst.abstain:
                self.audit.append(
                    case_id,
                    SMState.ABSTAIN,
                    agent="abstention",
                    action="abstain and route to a human navigator",
                    payload={
                        "reason": abst.reason[:200],
                        "confidence": f"{abst.calibrated_confidence:.3f}",
                        "threshold": f"{abst.threshold:.2f}",
                    },
                )
                tot_lat = (time.perf_counter() - prog_start) * 1000.0
                return ProgramOutcome(
                    program=program,
                    assessment=assessment,
                    verdict=verdict,
                    abstention=abst,
                    final_state=SMState.ABSTAIN,
                    abstained=True,
                    replans=replans,
                    citation_latency_ms=total_cit_lat,
                    llm_latency_ms=total_llm_lat,
                    total_latency_ms=tot_lat,
                )

            if assessment.recommended_action is RecommendedAction.PREPARE_APPLICATION:
                materials = self.preparer.prepare(assessment, evidence)
                self.audit.append(
                    case_id,
                    SMState.PREPARE,
                    agent="preparer",
                    action="prepared application materials (NOT submitted)",
                    citation_ids=cit_ids,
                )
                tot_lat = (time.perf_counter() - prog_start) * 1000.0
                return ProgramOutcome(
                    program=program,
                    assessment=assessment,
                    verdict=verdict,
                    abstention=abst,
                    materials=materials,
                    final_state=SMState.PREPARE,
                    abstained=False,
                    replans=replans,
                    citation_latency_ms=total_cit_lat,
                    llm_latency_ms=total_llm_lat,
                    total_latency_ms=tot_lat,
                )

            # Asserted a result that does not lead to preparing an application
            # (e.g. a confident "likely ineligible" with a recommendation to seek
            # human review / consider an appeal).
            self.audit.append(
                case_id,
                SMState.DONE,
                agent="navigator",
                action=f"asserted {assessment.status.value}; recommend {assessment.recommended_action.value}",
                citation_ids=cit_ids,
            )
            tot_lat = (time.perf_counter() - prog_start) * 1000.0
            return ProgramOutcome(
                program=program,
                assessment=assessment,
                verdict=verdict,
                abstention=abst,
                final_state=SMState.DONE,
                abstained=False,
                replans=replans,
                citation_latency_ms=total_cit_lat,
                llm_latency_ms=total_llm_lat,
                total_latency_ms=tot_lat,
            )
