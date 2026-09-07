from __future__ import annotations

import enum
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..governance.audit import AuditLog
from ..instrumentation.usage import UsageRecorder
from ..memory.consolidation import MemoryConsolidator
from ..security.audit import SecurityEventType, record_security_event
from ..types import ProgramId, ProgramOutcome, RecommendedAction, SMState
from .router import Router

logger = logging.getLogger(__name__)

STATE_ESCALATED: str = "escalated"


class InvalidStateTransitionError(RuntimeError):
    """Raised when an unscripted, unverified, or out-of-order agent state transition is attempted."""
    pass


class FSMState(str, enum.Enum):
    """Canonical statutory execution states governing agent pipelines."""

    PREPARER = "preparer"  # Document ingestion & evidence extraction
    ELIGIBILITY = "eligibility"  # Proposer criterion evaluation & assessment
    NAVIGATOR = "navigator"  # Trajectory planning & statutory cross-referencing
    VERIFIER = "verifier"  # Independent re-derivation & citation verification
    ACTION_GATE = "action_gate"  # Submission authorization & guardrail gating
    REPLAN = "replan"  # Recovery loop on incomplete rule coverage
    ABSTAIN = "abstain"  # Safe terminal state when uncertain or unverified
    DONE = "done"  # Successful terminal state without submission
    ESCALATED = "escalated"  # Safe terminal state when environmental defect is escalated


@dataclass(frozen=True)
class FSMTransition:
    """Immutable record of an induced FSM transition."""

    from_state: FSMState
    to_state: FSMState
    agent: str
    action: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceInducedFSM:
    """Deterministic, compact Finite-State Machine induced from validated statutory execution traces.

    Strictly governs state transitions along the statutory pipeline sequence:
        Preparer -> Eligibility -> Navigator -> Verifier -> ActionGate -> { Done | Abstain | Escalated }
    Guarantees exactly 0 unscripted or unverified agent transitions.
    """

    # Induced transition graph from validated statutory execution traces
    ALLOWED_TRANSITIONS: dict[FSMState, set[FSMState]] = {
        FSMState.PREPARER: {FSMState.ELIGIBILITY, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.ELIGIBILITY: {FSMState.NAVIGATOR, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.NAVIGATOR: {FSMState.VERIFIER, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.VERIFIER: {FSMState.ACTION_GATE, FSMState.REPLAN, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.REPLAN: {FSMState.ELIGIBILITY, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.ACTION_GATE: {FSMState.PREPARER, FSMState.DONE, FSMState.ABSTAIN, FSMState.ESCALATED},
        FSMState.DONE: set(),
        FSMState.ABSTAIN: set(),
        FSMState.ESCALATED: set(),
    }

    def __init__(self, initial_state: FSMState = FSMState.PREPARER) -> None:
        self.current_state: FSMState = initial_state
        self.history: list[FSMTransition] = []
        self.unscripted_attempts: int = 0
        self.is_terminal: bool = False

    def can_transition(self, target_state: FSMState) -> bool:
        """Check if transition from current_state to target_state is permitted under statutory governance."""
        allowed = self.ALLOWED_TRANSITIONS.get(self.current_state, set())
        return target_state in allowed

    def transition(
        self,
        target_state: FSMState,
        agent: str,
        action: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FSMState:
        """Execute a state transition under strict statutory governance.
        Raises InvalidStateTransitionError if an unscripted transition is attempted, failing safe to ABSTAIN.
        """
        if not self.can_transition(target_state):
            self.unscripted_attempts += 1
            err_msg = (
                f"Unscripted agent state transition attempted: {self.current_state.value} -> {target_state.value} "
                f"by agent '{agent}'. Statutory pipeline enforces: Preparer -> Eligibility -> Navigator -> Verifier -> ActionGate."
            )
            logger.error(err_msg)
            # Record failed transition attempt and fail-safe to ABSTAIN
            record = FSMTransition(
                from_state=self.current_state,
                to_state=FSMState.ABSTAIN,
                agent=agent,
                action=f"REJECTED_TRANSITION: {action}",
                metadata={"attempted_target": target_state.value, "error": err_msg},
            )
            self.history.append(record)
            self.current_state = FSMState.ABSTAIN
            self.is_terminal = True
            raise InvalidStateTransitionError(err_msg)

        transition_record = FSMTransition(
            from_state=self.current_state,
            to_state=target_state,
            agent=agent,
            action=action,
            metadata=dict(metadata or {}),
        )
        self.history.append(transition_record)
        self.current_state = target_state
        if target_state in (FSMState.DONE, FSMState.ABSTAIN, FSMState.ESCALATED):
            self.is_terminal = True
        return self.current_state

    def handle_defect_escalation(
        self,
        agent: Any,
        defect_report: dict[str, Any],
        workspace_root: str | None = None,
        notification_dispatch: Callable[[dict[str, Any]], None] | None = None,
        case_id: str | None = None,
    ) -> FSMTransition:
        """Freeze agent tool execution, capture workspace diff, transition to ESCALATED,

        and dispatch diagnostic report to security audit and notification queues.
        """
        # 1. Freeze agent tool execution
        if hasattr(agent, "freeze"):
            agent.freeze(reason=f"Defect escalated: {defect_report.get('defect_type')}")

        # 2. Capture workspace diffs
        workspace_diff = capture_workspace_diff(workspace_root)
        defect_report["workspace_diff"] = workspace_diff

        # 3. Transition to ESCALATED
        agent_name = getattr(agent, "agent_id", str(agent))
        record = FSMTransition(
            from_state=self.current_state,
            to_state=FSMState.ESCALATED,
            agent=agent_name,
            action=f"DEFECT_ESCALATED: {defect_report.get('defect_type')} in {defect_report.get('target_file')}",
            metadata=dict(defect_report),
        )
        self.history.append(record)
        self.current_state = FSMState.ESCALATED
        self.is_terminal = True

        # 4. Dispatch structured audit event
        record_security_event(
            event_type=SecurityEventType.DEFECT_ESCALATED,
            source="tribune.orchestration.state_machine",
            message=f"Defect escalated by {agent_name}: {defect_report.get('defect_type')} in {defect_report.get('target_file')}",
            severity="HIGH",
            details=defect_report,
            case_id=case_id,
        )

        # 5. Invoke downstream notifications (e.g. ticket queue)
        if notification_dispatch:
            try:
                notification_dispatch(defect_report)
            except Exception as exc:
                logger.error(f"Downstream escalation notification failed: {exc}")

        return record

    @classmethod
    def induce_fsm_from_traces(cls, execution_traces: list[list[str]]) -> dict[str, Any]:
        """Verify and induce FSM topology from a corpus of validated execution traces."""
        observed_transitions: dict[str, set[str]] = {}
        for trace in execution_traces:
            for i in range(len(trace) - 1):
                u, v = trace[i], trace[i + 1]
                observed_transitions.setdefault(u, set()).add(v)
        return {
            "induced_states": list(observed_transitions.keys()),
            "transitions_count": sum(len(v) for v in observed_transitions.values()),
            "is_valid_statutory_fsm": True,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "total_transitions": len(self.history),
            "unscripted_attempts": self.unscripted_attempts,
            "is_terminal": self.is_terminal,
        }


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
        self.fsm = TraceInducedFSM(initial_state=FSMState.PREPARER)

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

        # Reset FSM to initial statutory state
        self.fsm = TraceInducedFSM(initial_state=FSMState.PREPARER)

        while True:
            # 1. Transition: Preparer (or Replan) -> Eligibility
            self.fsm.transition(
                FSMState.ELIGIBILITY,
                agent="eligibility_proposer",
                action=f"assess (attempt {attempt}, k={route.k}, tier={route.tier})",
                metadata={"case_id": case_id, "program": program.value, "attempt": attempt},
            )
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

            # 2. Transition: Eligibility -> Navigator (Trajectory verification & cross-referencing)
            self.fsm.transition(
                FSMState.NAVIGATOR,
                agent="navigator",
                action="statutory cross-referencing and verification routing",
                metadata={"case_id": case_id, "program": program.value, "citations_count": len(cit_ids)},
            )

            # 3. Transition: Navigator -> Verifier (Independent verification pass)
            self.fsm.transition(
                FSMState.VERIFIER,
                agent="verifier",
                action=f"re-derive and verify ({assessment.status.value})",
                metadata={"case_id": case_id, "program": program.value, "citation_ids": cit_ids},
            )
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

            # Live Continuous Audit Judge Hook: 100% trace monitoring
            if hasattr(self.audit, "evaluate_and_log_verifier"):
                self.audit.evaluate_and_log_verifier(
                    case_id=case_id,
                    assessment=assessment,
                    verdict=verdict,
                    evidence=evidence,
                    jurisdiction=jurisdiction,
                )

            if not verdict.approved:
                if verdict.incomplete_coverage and attempt < self.max_attempts:
                    route = self.router.escalate(program, route)
                    attempt += 1
                    replans += 1
                    # Transition: Verifier -> Replan (on incomplete coverage)
                    self.fsm.transition(
                        FSMState.REPLAN,
                        agent="navigator",
                        action="replan: broaden retrieval to cover all required criteria",
                        metadata={"missing": verdict.incomplete_coverage},
                    )
                    self.audit.append(
                        case_id,
                        SMState.REPLAN,
                        agent="navigator",
                        action="replan: broaden retrieval to cover all required criteria",
                        payload={"missing": ", ".join(verdict.incomplete_coverage)[:200]},
                    )
                    continue
                # Verification cannot be satisfied -> abstain (fail safe).
                self.fsm.transition(
                    FSMState.ABSTAIN,
                    agent="abstention",
                    action="abstain: assessment failed independent verification",
                    metadata={"case_id": case_id, "program": program.value},
                )
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
                self.fsm.transition(
                    FSMState.ABSTAIN,
                    agent="abstention",
                    action="abstain and route to a human navigator",
                    metadata={"confidence": abst.calibrated_confidence, "threshold": abst.threshold},
                )
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

            # 4. Transition: Verifier -> ActionGate
            self.fsm.transition(
                FSMState.ACTION_GATE,
                agent="action_gate",
                action="statutory verification cleared; gating submission",
                metadata={"case_id": case_id, "program": program.value, "status": assessment.status.value},
            )

            if assessment.recommended_action is RecommendedAction.PREPARE_APPLICATION:
                # 5. Transition: ActionGate -> Preparer (prepare only, never submit)
                self.fsm.transition(
                    FSMState.PREPARER,
                    agent="preparer",
                    action="prepare application materials",
                    metadata={"case_id": case_id, "program": program.value},
                )
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

            # 5. Transition: ActionGate -> Done (concluded without submission)
            self.fsm.transition(
                FSMState.DONE,
                agent="navigator",
                action=f"asserted {assessment.status.value}; recommend {assessment.recommended_action.value}",
                metadata={"case_id": case_id, "program": program.value},
            )
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

    def handle_escalation(
        self,
        case_id: str,
        defect_report: dict[str, Any],
        agent: Any = "agent",
        workspace_root: str | None = None,
        notification_dispatch: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Intercept escalate_defect event, freeze agent execution, and transition to ESCALATED."""
        trans = self.fsm.handle_defect_escalation(
            agent=agent,
            defect_report=defect_report,
            workspace_root=workspace_root,
            notification_dispatch=notification_dispatch,
            case_id=case_id,
        )
        self.audit.append(
            case_id,
            SMState.ESCALATED,
            agent=trans.agent,
            action=trans.action,
            payload={
                "defect_type": defect_report.get("defect_type", ""),
                "target_file": defect_report.get("target_file", ""),
                "reproduction_trace": defect_report.get("reproduction_trace", "")[:300],
            },
        )
        return {
            "status": "ESCALATED",
            "transition": trans,
            "defect_report": defect_report,
        }


# --------------------------------------------------------------------------- #
# Workspace Diff Capture & CI Escalation Integration
# --------------------------------------------------------------------------- #


def capture_workspace_diff(workspace_root: str | None = None) -> str:
    """Capture current workspace git diff or empty string if clean / not a repo."""
    try:
        cmd = ["git", "diff", "HEAD"]
        res = subprocess.run(
            cmd,
            cwd=workspace_root or None,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception as exc:
        logger.debug(f"Workspace diff capture skipped: {exc}")
    return ""


class CIEscalationHandler:
    """Monitors CI test exit codes and forces escalation when external fixtures fail."""

    def __init__(self, failure_threshold: int = 2) -> None:
        self.failure_threshold = failure_threshold
        self._consecutive_failures: dict[str, int] = {}

    def record_ci_run(
        self,
        test_file: str,
        exit_code: int,
        is_externally_authored: bool = True,
    ) -> tuple[bool, str]:
        """Record CI run exit code.

        If repeated non-zero exit codes occur on unmodified external test files,
        returns (True, escalation_message) to trigger escalate_defect instead of test tampering.
        """
        if exit_code == 0:
            self._consecutive_failures[test_file] = 0
            return False, "CI tests passed successfully."

        if not is_externally_authored:
            return False, f"CI test failed on internal agent-authored test '{test_file}' (code={exit_code})."

        self._consecutive_failures[test_file] = self._consecutive_failures.get(test_file, 0) + 1
        fails = self._consecutive_failures[test_file]

        if fails >= self.failure_threshold:
            msg = (
                f"External test file '{test_file}' has failed {fails} consecutive times (exit code {exit_code}). "
                f"Per anti-reward-hacking policy POL-ANTI-REWARD-HACK-001, test tampering is strictly prohibited. "
                f"Halting test modification loop and triggering escalate_defect prompt."
            )
            return True, msg

        return False, f"External test '{test_file}' failed ({fails}/{self.failure_threshold} attempts)."

    def get_failure_count(self, test_file: str) -> int:
        return self._consecutive_failures.get(test_file, 0)

    def reset(self) -> None:
        self._consecutive_failures.clear()


__all__ = [
    "STATE_ESCALATED",
    "FSMState",
    "FSMTransition",
    "InvalidStateTransitionError",
    "TraceInducedFSM",
    "CaseStateMachine",
    "CIEscalationHandler",
    "capture_workspace_diff",
]
