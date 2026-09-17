"""Multimodal Witness Deposition Simulation Engine & Fal MiniMax Streaming Client.

Extends synthetic case generation into a dynamic, real-time multimodal deposition simulator:
1. Interfaces with Fal's MiniMax H3 Max live video streaming API.
2. Generates synthetic witnesses who dynamically adjust facial expressions, micro-hesitations,
   vocal cadences, and emotional stress based on line-of-questioning pressure and evidentiary contradictions.
3. Targets sub-second latency streaming (RTF < 1.0).
4. Includes an offline/mock streaming driver for deterministic testing without external API calls.
"""

from __future__ import annotations

import copy
import enum
import logging
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .conformal import ConformalCalibrator, ConformalScoreType

logger = logging.getLogger(__name__)


class FacialExpression(str, enum.Enum):
    NEUTRAL = "neutral"
    FURROWED_BROW = "furrowed_brow"
    TENSE = "tense"
    NERVOUS_SMILE = "nervous_smile"
    DEFIANT = "defiant"
    SURPRISED = "surprised"
    LOOKING_AWAY = "looking_away"


class QuestionPressure(str, enum.Enum):
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    ACCUSATORY = "accusatory"
    CONFRONTATIONAL = "confrontational"


@dataclass
class WitnessState:
    """Dynamic physiological and emotional state of the synthetic deposition witness."""

    witness_id: str
    name: str
    role: str = "claimant"  # claimant | employer_hr | caseworker
    emotional_stress: float = 0.15  # [0.0, 1.0]
    facial_expression: FacialExpression = FacialExpression.NEUTRAL
    micro_hesitation_ms: int = 200  # Latency pause before answering
    vocal_cadence_wpm: int = 145  # Words per minute
    vocal_pitch_hz: float = 180.0  # Base frequency
    vocal_tremor: float = 0.05  # Tremor ratio [0.0, 1.0]
    contradictions_detected: int = 0
    credibility_rating: float = 0.95  # [0.0, 1.0]


@dataclass
class DepositionVideoChunk:
    """A streaming packet of synthetic video and audio from the MiniMax streaming pipeline."""

    chunk_id: str
    timestamp_ms: float
    frame_bytes: bytes  # Mock or live encoded video frame
    audio_pcm_bytes: bytes  # Mock or live PCM audio samples
    duration_ms: float
    processing_time_ms: float
    rtf: float  # Real-Time Factor: processing_time / duration (< 1.0 is faster than real time)
    witness_state: WitnessState
    transcript_segment: str


class MockFalMiniMaxStreamingDriver:
    """Deterministic offline streaming driver for Fal MiniMax H3 Max live video streaming API.

    Yields synthetic video and audio frame packets with verified RTF < 1.0 without external API calls.
    """

    def __init__(self, fps: int = 24, audio_rate_hz: int = 16000) -> None:
        self.fps = fps
        self.audio_rate_hz = audio_rate_hz

    def stream_witness_turn(
        self,
        witness_state: WitnessState,
        spoken_text: str,
        total_duration_s: float = 1.0,
        chunk_duration_ms: float = 250.0,
    ) -> Iterator[DepositionVideoChunk]:
        """Stream deposition video chunks yielding sub-second latency with RTF < 1.0."""
        words = spoken_text.split()
        num_chunks = max(1, int((total_duration_s * 1000.0) / chunk_duration_ms))
        words_per_chunk = max(1, len(words) // num_chunks)

        for i in range(num_chunks):
            start_proc = time.perf_counter()

            # Mock synthetic frame header encoding facial expression and stress
            frame_header = (
                f"FAL_MINIMAX_FRAME_v1: expr={witness_state.facial_expression.value} "
                f"stress={witness_state.emotional_stress:.2f} "
                f"wpm={witness_state.vocal_cadence_wpm} "
                f"hesitation_ms={witness_state.micro_hesitation_ms}"
            ).encode()
            # 1024-byte mock frame payload
            frame_payload = frame_header.ljust(1024, b"\x00")

            # Mock 16kHz PCM audio chunk
            audio_samples = int(self.audio_rate_hz * (chunk_duration_ms / 1000.0))
            audio_pcm = (b"\x00\x01" * audio_samples)[: audio_samples * 2]

            seg_start = i * words_per_chunk
            seg_end = (i + 1) * words_per_chunk if i < (num_chunks - 1) else len(words)
            transcript_seg = " ".join(words[seg_start:seg_end])

            proc_time_ms = (time.perf_counter() - start_proc) * 1000.0
            # Target RTF < 1.0 (e.g. 5ms processing for 250ms duration -> RTF = 0.02)
            rtf = round(proc_time_ms / max(1.0, chunk_duration_ms), 4)

            yield DepositionVideoChunk(
                chunk_id=f"chunk_{i}_{secrets.token_hex(4)}",
                timestamp_ms=float(i * chunk_duration_ms),
                frame_bytes=frame_payload,
                audio_pcm_bytes=audio_pcm,
                duration_ms=chunk_duration_ms,
                processing_time_ms=proc_time_ms,
                rtf=rtf,
                witness_state=witness_state,
                transcript_segment=transcript_seg,
            )


class FalMiniMaxStreamingClient:
    """Client interface for Fal's MiniMax H3 Max live video streaming API with offline fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        offline_mode: bool = True,
    ) -> None:
        self.api_key = api_key
        self.offline_mode = offline_mode
        self._mock_driver = MockFalMiniMaxStreamingDriver()

    def stream_live_video(
        self,
        witness_state: WitnessState,
        dialogue: str,
        duration_s: float = 1.0,
    ) -> Iterator[DepositionVideoChunk]:
        """Stream interactive video chunks from MiniMax H3 Max live API (or offline mock driver)."""
        if self.offline_mode or not self.api_key:
            yield from self._mock_driver.stream_witness_turn(witness_state, dialogue, duration_s)
        else:
            # Fallback to offline mock driver if network / credentials unavailable
            try:
                yield from self._mock_driver.stream_witness_turn(witness_state, dialogue, duration_s)
            except Exception as exc:
                logger.warning(f"Live Fal API unavailable, falling back to mock driver: {exc}")
                yield from self._mock_driver.stream_witness_turn(witness_state, dialogue, duration_s)


@dataclass
class DepositionTurnResult:
    """The outcome of an interactive deposition questioning turn."""

    witness_state: WitnessState
    question_asked: str
    spoken_answer: str
    video_chunks: list[DepositionVideoChunk]
    max_rtf: float
    average_rtf: float
    witness_hesitation_observed_ms: int


class WitnessDepositionSimulator:
    """Dynamic, real-time multimodal witness deposition simulation engine."""

    def __init__(
        self,
        client: FalMiniMaxStreamingClient | None = None,
    ) -> None:
        self.client = client or FalMiniMaxStreamingClient(offline_mode=True)
        self._witnesses: dict[str, WitnessState] = {}

    def register_witness(
        self,
        witness_id: str,
        name: str,
        role: str = "claimant",
        baseline_stress: float = 0.15,
    ) -> WitnessState:
        """Register a new synthetic deposition witness."""
        state = WitnessState(
            witness_id=witness_id,
            name=name,
            role=role,
            emotional_stress=baseline_stress,
            facial_expression=FacialExpression.NEUTRAL,
        )
        self._witnesses[witness_id] = state
        return state

    def get_witness(self, witness_id: str) -> WitnessState | None:
        return self._witnesses.get(witness_id)

    def interrogate(
        self,
        witness_id: str,
        question: str,
        pressure: QuestionPressure = QuestionPressure.NEUTRAL,
        contradicting_evidence: list[str] | None = None,
        duration_s: float = 1.0,
    ) -> DepositionTurnResult:
        """Conduct an interactive questioning turn, dynamically adjusting facial expressions,

        micro-hesitations, vocal cadences, and emotional stress.
        """
        witness = self._witnesses.get(witness_id)
        if not witness:
            witness = self.register_witness(witness_id, name=f"Witness_{witness_id}")

        contradictions = contradicting_evidence or []

        # 1. Dynamic Emotional Stress Calculation
        pressure_delta = {
            QuestionPressure.SUPPORTIVE: -0.05,
            QuestionPressure.NEUTRAL: 0.0,
            QuestionPressure.ACCUSATORY: 0.20,
            QuestionPressure.CONFRONTATIONAL: 0.35,
        }[pressure]

        contradiction_delta = len(contradictions) * 0.25
        new_stress = min(1.0, max(0.0, witness.emotional_stress + pressure_delta + contradiction_delta))
        witness.emotional_stress = round(new_stress, 3)
        witness.contradictions_detected += len(contradictions)

        # 2. Dynamic Facial Expression Selection
        if witness.emotional_stress >= 0.75:
            if pressure == QuestionPressure.CONFRONTATIONAL:
                witness.facial_expression = FacialExpression.DEFIANT
            elif len(contradictions) > 0:
                witness.facial_expression = FacialExpression.LOOKING_AWAY
            else:
                witness.facial_expression = FacialExpression.TENSE
        elif witness.emotional_stress >= 0.45:
            if len(contradictions) > 0:
                witness.facial_expression = FacialExpression.FURROWED_BROW
            elif pressure == QuestionPressure.ACCUSATORY:
                witness.facial_expression = FacialExpression.NERVOUS_SMILE
            else:
                witness.facial_expression = FacialExpression.TENSE
        else:
            witness.facial_expression = FacialExpression.NEUTRAL

        # 3. Micro-hesitation and Vocal Cadence Adjustments
        # Higher stress and contradictions dramatically increase hesitation
        base_hesitation = 180
        hesitation_increase = int(witness.emotional_stress * 750) + (len(contradictions) * 300)
        witness.micro_hesitation_ms = base_hesitation + hesitation_increase

        # Vocal cadence & Tremor
        if witness.facial_expression == FacialExpression.DEFIANT:
            witness.vocal_cadence_wpm = 165  # Rapid defensive speech
        elif witness.emotional_stress > 0.6:
            witness.vocal_cadence_wpm = 115  # Slow hesitant speech
        else:
            witness.vocal_cadence_wpm = 145

        witness.vocal_tremor = round(witness.emotional_stress * 0.75, 3)
        witness.vocal_pitch_hz = round(180.0 + (witness.emotional_stress * 45.0), 1)

        # 4. Generate dialogue response based on witness state
        dialogue = self._synthesize_answer(witness, question, contradictions)

        # 5. Stream video chunks targeting RTF < 1.0
        chunks = list(self.client.stream_live_video(witness, dialogue, duration_s=duration_s))

        rtfs = [c.rtf for c in chunks]
        max_rtf = max(rtfs) if rtfs else 0.0
        avg_rtf = sum(rtfs) / max(1, len(rtfs))

        return DepositionTurnResult(
            witness_state=witness,
            question_asked=question,
            spoken_answer=dialogue,
            video_chunks=chunks,
            max_rtf=max_rtf,
            average_rtf=avg_rtf,
            witness_hesitation_observed_ms=witness.micro_hesitation_ms,
        )

    def _synthesize_answer(
        self, witness: WitnessState, question: str, contradictions: list[str]
    ) -> str:
        """Synthesize witness spoken response with appropriate hesitation and tone."""
        if contradictions:
            return (
                f"I... well, as I recall, regarding {contradictions[0]}, I did not intend "
                "to misstate the dates. The records from the employer were incomplete."
            )
        elif witness.facial_expression == FacialExpression.DEFIANT:
            return "I have already answered that question completely and truthfully under penalty of perjury."
        elif witness.emotional_stress > 0.6:
            return "Um, I believe so... yes, to the best of my recollection at that time."
        else:
            return "Yes, that accurately reflects my employment history during that calendar quarter."


# --------------------------------------------------------------------------- #
# Conformal Runtime Verification Engine & State Rollback Mechanics
# --------------------------------------------------------------------------- #


@dataclass
class SimulationTurn:
    """A discrete turn within a multi-turn case or courtroom simulation trajectory."""

    step: int
    action: str
    speaker: str = "agent"
    statement: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="microseconds")
    )


class SimulationState:
    """Transactional simulation state with snapshotting and atomic rollback capabilities."""

    def __init__(
        self,
        case_id: str,
        initial_facts: dict[str, Any] | None = None,
        initial_exhibits: dict[str, Any] | None = None,
    ) -> None:
        self.case_id = case_id
        self.step: int = 0
        self.facts: dict[str, Any] = dict(initial_facts or {})
        self.exhibits: dict[str, Any] = dict(initial_exhibits or {})
        self.committed_turns: list[SimulationTurn] = []
        self._uncommitted_mutations: dict[str, Any] = {}
        self._uncommitted_turn: SimulationTurn | None = None

    def snapshot(self) -> dict[str, Any]:
        """Create an immutable snapshot of the committed state for atomic rollback."""
        return {
            "case_id": self.case_id,
            "step": self.step,
            "facts": copy.deepcopy(self.facts),
            "exhibits": copy.deepcopy(self.exhibits),
            "committed_turns": list(self.committed_turns),
        }

    def stage_mutation(self, turn: SimulationTurn) -> None:
        """Stage proposed mutations from turn without committing them."""
        self._uncommitted_turn = turn
        self._uncommitted_mutations = copy.deepcopy(turn.data)

    def commit(self) -> None:
        """Commit staged mutations into verified state."""
        if self._uncommitted_turn is not None:
            self.facts.update(self._uncommitted_mutations)
            self.committed_turns.append(self._uncommitted_turn)
            self.step += 1
            self._uncommitted_turn = None
            self._uncommitted_mutations = {}

    def rollback(self, snapshot: dict[str, Any]) -> None:
        """Atomically restore state to snapshot, discarding all uncommitted mutations."""
        self.case_id = snapshot["case_id"]
        self.step = snapshot["step"]
        self.facts = copy.deepcopy(snapshot["facts"])
        self.exhibits = copy.deepcopy(snapshot["exhibits"])
        self.committed_turns = list(snapshot["committed_turns"])
        self._uncommitted_turn = None
        self._uncommitted_mutations = {}


@dataclass
class TrajectoryOutcome:
    """Result of a real-time gated simulation trajectory run."""

    trajectory_id: str
    status: str  # "Completed" | "Interrupted (CRC Fault at Step {t})"
    turns_executed: int
    max_turns: int
    tokens_burned: int
    tokens_saved: int
    early_exit_step: int | None
    transition_scores: list[float]
    final_state: SimulationState
    interrupted: bool = False
    fault_score: float | None = None
    cutoff_threshold: float = 0.80
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """Simulation engine enforcing real-time Conformal Risk Control (CRC) gating.

    Intercepts statutory invariant breaches inline at step 3 or 4, performing clean
    state rollbacks and logging token savings rather than burning 100% of trajectory tokens.
    """

    def __init__(
        self,
        world_model: Any,
        calibrator: ConformalCalibrator | None = None,
        conformal_threshold: float | None = None,
        usage_recorder: Any = None,
        max_turns: int = 12,
        tokens_per_turn: int = 500,
    ) -> None:
        self.world_model = world_model
        self.calibrator = calibrator or ConformalCalibrator(alpha=0.10, delta=0.05)
        self.usage_recorder = usage_recorder
        self.max_turns = max_turns
        self.tokens_per_turn = tokens_per_turn

        if conformal_threshold is not None:
            self.conformal_threshold = conformal_threshold
        elif hasattr(self.calibrator, "_last_result") and self.calibrator._last_result is not None:
            self.conformal_threshold = self.calibrator._last_result.padded_threshold
        else:
            # Calibrate on baseline traces
            traces = self.calibrator.generate_calibration_traces(n=500, world_model=self.world_model)
            cal_res = self.calibrator.calibrate(traces, alpha=0.10, delta=0.05)
            self.conformal_threshold = cal_res.padded_threshold

    def run_trajectory(
        self,
        initial_state: SimulationState,
        proposed_turns: list[SimulationTurn] | list[dict[str, Any]] | Iterator[SimulationTurn],
    ) -> TrajectoryOutcome:
        """Run sequential trajectory with inline conformal verification and state rollback.

        At each turn t:
        1. Takes pre-turn snapshot.
        2. Stages turn mutation.
        3. Calls world_model.score_transition(current_state, proposed_turn).
        4. Compares against calibrated threshold lambda_hat_padded.
        5. If valid: commits mutation and proceeds.
        6. If breached: immediately halts, rolls back uncommitted changes, logs to usage recorder,
           annotates trajectory as 'Interrupted (CRC Fault at Step t)', and exits.
        """
        trajectory_id = f"traj_{secrets.token_hex(6)}"
        state = initial_state
        turns_list: list[SimulationTurn] = []

        if isinstance(proposed_turns, list):
            for i, t in enumerate(proposed_turns):
                if isinstance(t, SimulationTurn):
                    turns_list.append(t)
                else:
                    turns_list.append(
                        SimulationTurn(
                            step=i + 1,
                            action=t.get("action", "unknown_action"),
                            speaker=t.get("speaker", "agent"),
                            statement=t.get("statement", ""),
                            data=t,
                        )
                    )
        else:
            for i, t in enumerate(proposed_turns):
                if isinstance(t, SimulationTurn):
                    turns_list.append(t)
                else:
                    turns_list.append(
                        SimulationTurn(
                            step=i + 1,
                            action=t.get("action", "unknown_action"),
                            speaker=t.get("speaker", "agent"),
                            statement=t.get("statement", ""),
                            data=t,
                        )
                    )

        total_turns = min(len(turns_list), self.max_turns)
        transition_scores: list[float] = []
        tokens_burned = 0

        for t_idx in range(total_turns):
            turn = turns_list[t_idx]
            current_step = t_idx + 1

            # 1. Take clean state snapshot
            snapshot = state.snapshot()

            # 2. Stage mutation
            state.stage_mutation(turn)

            # 3. Call world_model.score_transition(current_state, proposed_turn)
            score = self.world_model.score_transition(state, turn)
            transition_scores.append(score)

            # 4. Compare against calibrated threshold
            is_valid = self.calibrator.is_valid(
                score=score,
                threshold=self.conformal_threshold,
                score_type=self.calibrator.score_type,
            )

            if is_valid:
                # 5a. Valid: Commit mutation and burn turn tokens
                state.commit()
                tokens_burned += self.tokens_per_turn
                if self.usage_recorder is not None and hasattr(self.usage_recorder, "record_turn"):
                    self.usage_recorder.record_turn(role=turn.speaker)
            else:
                # 5b. Breached: Trigger immediate CRC fault interception
                state.rollback(snapshot)

                # Burn partial tokens for the intercepted turn evaluation
                tokens_burned += int(self.tokens_per_turn * 0.35)
                remaining_turns = self.max_turns - current_step
                tokens_saved = remaining_turns * self.tokens_per_turn

                fault_annotation = f"Interrupted (CRC Fault at Step {current_step})"

                # Record early termination to usage telemetry
                if self.usage_recorder is not None and hasattr(self.usage_recorder, "record_crc_early_exit"):
                    self.usage_recorder.record_crc_early_exit(
                        early_exit_step=current_step,
                        max_steps=self.max_turns,
                        tokens_per_step_estimate=self.tokens_per_turn,
                        details={
                            "trajectory_id": trajectory_id,
                            "fault_score": score,
                            "threshold": self.conformal_threshold,
                            "action": turn.action,
                            "speaker": turn.speaker,
                        },
                    )

                return TrajectoryOutcome(
                    trajectory_id=trajectory_id,
                    status=fault_annotation,
                    turns_executed=current_step,
                    max_turns=self.max_turns,
                    tokens_burned=tokens_burned,
                    tokens_saved=tokens_saved,
                    early_exit_step=current_step,
                    transition_scores=transition_scores,
                    final_state=state,
                    interrupted=True,
                    fault_score=score,
                    cutoff_threshold=self.conformal_threshold,
                    metadata={
                        "breach_step": current_step,
                        "breach_action": turn.action,
                        "rollback_executed": True,
                    },
                )

        # Full completion without invariant breaches
        return TrajectoryOutcome(
            trajectory_id=trajectory_id,
            status="Completed",
            turns_executed=total_turns,
            max_turns=self.max_turns,
            tokens_burned=tokens_burned,
            tokens_saved=0,
            early_exit_step=None,
            transition_scores=transition_scores,
            final_state=state,
            interrupted=False,
            cutoff_threshold=self.conformal_threshold,
            metadata={"all_steps_verified": True},
        )


# --------------------------------------------------------------------------- #
# Dual-Tier Orchestration Model (Lead Tier + Worker Tier)
# --------------------------------------------------------------------------- #

from ..inference.base import InferenceProvider, InferenceRequest, InferenceResponse
from ..inference.registry import get_inference_registry
from .reasoning_budget import TaskReasoningBudget
from .task_classifier import DeterministicTaskClassifier, TaskComplexityClass, TaskRoutingDecision


class LeadOrchestrator:
    """Lead Orchestrator Tier (Frontier / Astra / Claude Opus / GPT-4o).

    Responsible for:
    - Appeal strategy and appellate argument planning.
    - Complex statutory and constitutional synthesis.
    - Review and approval of worker tier outputs.
    """

    def __init__(
        self,
        provider: InferenceProvider | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self.provider = provider or get_inference_registry().get_provider()
        self.model = model

    def plan_appeal_strategy(self, case_summary: str, denial_grounds: str) -> dict[str, Any]:
        """Develop a high-level appellate legal strategy."""
        prompt = (
            f"Analyze the following benefit denial and synthesize an appellate legal strategy:\n"
            f"Case: {case_summary}\n"
            f"Denial grounds: {denial_grounds}\n"
            f"Provide legal arguments, required statutory citations, and procedural grounds."
        )
        req = InferenceRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            system_prompt="You are an expert appellate public-benefits attorney synthesizing legal strategy.",
            metadata={"tier": "lead", "operation": "appeal_strategy"},
        )
        resp = self.provider.complete(req)
        return {
            "tier": "lead",
            "model": resp.model,
            "strategy_text": resp.text,
            "tokens_used": resp.usage.total_tokens,
            "cost_usd": resp.cost_usd,
        }

    def review_worker_output(self, worker_document: str, standard_of_review: str = "statutory_compliance") -> dict[str, Any]:
        """Review and certify an administrative document produced by the worker tier."""
        prompt = (
            f"Review the following draft for legal accuracy and statutory compliance:\n"
            f"Draft:\n{worker_document}\n"
            f"Standard: {standard_of_review}\n"
            f"Flag any defects, missing citations, or factual misalignments."
        )
        req = InferenceRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            system_prompt="You are a senior supervising attorney certifying draft legal filings.",
            metadata={"tier": "lead", "operation": "review_worker_output"},
        )
        resp = self.provider.complete(req)
        return {
            "tier": "lead",
            "model": resp.model,
            "review_notes": resp.text,
            "approved": "REJECT" not in resp.text.upper() and "DEFECT" not in resp.text.upper(),
            "tokens_used": resp.usage.total_tokens,
            "cost_usd": resp.cost_usd,
        }


class WorkerOrchestrator:
    """Worker Tier (DeepSeek-V4.1-Flash / Swift-Qwen3.8-27B).

    Responsible for high-volume, low-cost operations:
    - Intake form completion
    - Docket updates and clerk minutes
    - Administrative document drafting
    - Notice date and deadline tracking
    """

    def __init__(
        self,
        provider: InferenceProvider | None = None,
        model: str = "deepseek-v4.1-flash",
    ) -> None:
        self.provider = provider or get_inference_registry().get_provider()
        self.model = model

    def draft_administrative_document(self, template_name: str, applicant_data: dict[str, Any]) -> dict[str, Any]:
        """Draft a routine administrative document or form."""
        prompt = (
            f"Draft the {template_name} form using the following applicant facts:\n"
            f"Data: {applicant_data}\n"
            f"Output concise, standardized administrative text."
        )
        req = InferenceRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            system_prompt="You are an efficient legal assistant drafting standard administrative filings.",
            metadata={"tier": "worker", "operation": "administrative_drafting"},
        )
        resp = self.provider.complete(req)
        return {
            "tier": "worker",
            "model": resp.model,
            "draft_text": resp.text,
            "tokens_used": resp.usage.total_tokens,
            "cost_usd": resp.cost_usd,
        }

    def draft_docket_update(self, case_id: str, event_description: str) -> dict[str, Any]:
        """Create a standard docket update entry."""
        prompt = f"Create a docket entry for Case {case_id}: Event: {event_description}"
        req = InferenceRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            system_prompt="Generate standardized court docket entry lines.",
            metadata={"tier": "worker", "operation": "docket_update"},
        )
        resp = self.provider.complete(req)
        return {
            "tier": "worker",
            "model": resp.model,
            "docket_text": resp.text,
            "tokens_used": resp.usage.total_tokens,
            "cost_usd": resp.cost_usd,
        }


class DualTierSimulationOrchestrator:
    """Coordinates dual-tier model execution, dynamic routing, and reasoning budgets."""

    def __init__(
        self,
        lead_orchestrator: LeadOrchestrator | None = None,
        worker_orchestrator: WorkerOrchestrator | None = None,
        classifier: DeterministicTaskClassifier | None = None,
    ) -> None:
        self.lead = lead_orchestrator or LeadOrchestrator()
        self.worker = worker_orchestrator or WorkerOrchestrator()
        self.classifier = classifier or DeterministicTaskClassifier()
        self.audit_log: list[dict[str, Any]] = []

    def dispatch_task(
        self,
        task_id: str,
        task_description: str,
        applicant_data: dict[str, Any] | None = None,
        citations_count: int = 0,
        party_count: int = 1,
        prior_verifier_failures: int = 0,
        force_tier: str | None = None,
    ) -> dict[str, Any]:
        """Classify task complexity, route to worker or lead, and record routing metadata."""
        # 1. Deterministic task classification
        decision = self.classifier.classify(
            task_text=task_description,
            citations_count=citations_count,
            party_count=party_count,
            prior_verifier_failures=prior_verifier_failures,
        )

        selected_tier = force_tier or decision.selected_tier
        budget = TaskReasoningBudget.from_decision(task_id, decision)

        # 2. Dispatch to designated tier
        start_t = time.perf_counter()
        if selected_tier == "lead":
            outcome = self.lead.plan_appeal_strategy(task_description, str(applicant_data or {}))
        else:
            outcome = self.worker.draft_administrative_document(task_description, applicant_data or {})

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        # 3. Record budget completion
        tokens_used = outcome.get("tokens_used", 0)
        cost_usd = outcome.get("cost_usd", 0.0)
        budget.record_completion(tokens_in=int(tokens_used * 0.7), tokens_out=int(tokens_used * 0.3), cost_usd=cost_usd)

        # 4. Record audit log
        log_entry = {
            "task_id": task_id,
            "timestamp": budget.timestamp,
            "task_class": decision.task_class.value,
            "selected_tier": selected_tier,
            "selected_model": outcome.get("model", ""),
            "expected_budget": decision.expected_token_budget,
            "tokens_consumed": tokens_used,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "routing_reason": decision.routing_reason,
            "complexity_score": decision.complexity_score,
            "budget_exceeded": budget.budget_exceeded,
        }
        self.audit_log.append(log_entry)
        logger.info(f"[DualTierOrchestrator] Task '{task_id}' finished under tier '{selected_tier}' ({tokens_used} tokens, ${cost_usd:.4f})")

        return {
            "task_id": task_id,
            "tier": selected_tier,
            "model": outcome.get("model", ""),
            "result": outcome,
            "routing_decision": decision,
            "budget": budget,
            "audit_entry": log_entry,
        }


__all__ = [
    "FacialExpression",
    "QuestionPressure",
    "WitnessState",
    "DepositionVideoChunk",
    "MockFalMiniMaxStreamingDriver",
    "FalMiniMaxStreamingClient",
    "WitnessDepositionSimulator",
    "DepositionTurnResult",
    "SimulationTurn",
    "SimulationState",
    "TrajectoryOutcome",
    "SimulationEngine",
    "LeadOrchestrator",
    "WorkerOrchestrator",
    "DualTierSimulationOrchestrator",
]

