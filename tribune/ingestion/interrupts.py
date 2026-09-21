"""Interruption Handling, Barge-In Detection, and Paralinguistic Marker Extraction.

Provides rapid outgoing audio flush (<50ms target) upon detected user speech during playback,
preserves conversational state across interrupts, and computes paralinguistic markers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .audio_transport import AudioPacket, AudioTransport

logger = logging.getLogger(__name__)


@dataclass
class ParalinguisticMarkers:
    """Paralinguistic cues and markers extracted from the audio stream."""

    voice_activity_level: float = 0.0  # [0.0, 1.0]
    interruption_urgency: float = 0.0  # [0.0, 1.0]
    speech_rate_estimate: float = 0.0  # syllables or words per second
    silence_duration: float = 0.0  # seconds of silence preceding frame
    barge_in_event: bool = False
    acoustic_entropy: float = 0.0
    energy_rms: float = 0.0


@dataclass
class ConversationalState:
    """State preserved across interruption boundaries."""

    session_id: str
    active_topic: str = ""
    last_system_utterance: str = ""
    system_playback_interrupted: bool = False
    interrupted_playback_offset_ms: float = 0.0
    accumulated_transcript: list[str] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class InterruptionDetector:
    """Detects user barge-in speech during system playback, computes urgency, and triggers fast flush."""

    def __init__(
        self,
        vad_threshold: float = 0.50,
        consecutive_frames_required: int = 2,
        flush_target_ms: float = 50.0,
    ) -> None:
        self.vad_threshold = vad_threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.flush_target_ms = flush_target_ms

        self._consecutive_speech_frames = 0
        self._last_speech_time_s: float = time.time()
        self.total_interruptions = 0
        self.last_flush_latency_ms = 0.0

    def analyze_packet(
        self,
        packet: AudioPacket,
        is_system_playing: bool,
    ) -> ParalinguisticMarkers:
        """Analyze an incoming audio packet for voice activity and barge-in markers."""
        current_t = time.time()
        silence_duration = max(0.0, current_t - self._last_speech_time_s)

        # Estimate energy and VAD
        vol = packet.volume
        is_speech = packet.is_speech or (vol >= self.vad_threshold)

        if is_speech:
            self._consecutive_speech_frames += 1
            self._last_speech_time_s = current_t
        else:
            self._consecutive_speech_frames = 0

        barge_in = False
        urgency = 0.0

        if is_system_playing and self._consecutive_speech_frames >= self.consecutive_frames_required:
            barge_in = True
            # Higher volume / sudden interruption increases urgency
            urgency = min(1.0, 0.5 + vol * 0.5)

        # Approximate speech rate estimate (normalized words/sec proxy)
        speech_rate = round(3.5 + vol * 1.5, 2) if is_speech else 0.0

        return ParalinguisticMarkers(
            voice_activity_level=round(min(1.0, vol), 4),
            interruption_urgency=round(urgency, 4),
            speech_rate_estimate=speech_rate,
            silence_duration=round(silence_duration, 4),
            barge_in_event=barge_in,
            energy_rms=round(vol, 4),
        )

    async def handle_barge_in(
        self,
        transport: AudioTransport,
        state: ConversationalState,
        interrupted_offset_ms: float = 0.0,
    ) -> float:
        """Execute fast interruption flush on transport and update conversation state.

        Guarantees conversational state is preserved while cutting audio playback immediately.
        """
        start_t = time.perf_counter()

        # Flush outbound transport
        flush_ms = await transport.flush()

        # Update conversational state
        state.system_playback_interrupted = True
        state.interrupted_playback_offset_ms = interrupted_offset_ms
        self.total_interruptions += 1

        total_latency_ms = (time.perf_counter() - start_t) * 1000.0 + flush_ms
        self.last_flush_latency_ms = total_latency_ms

        logger.info(
            f"[InterruptionDetector] Barge-in executed: flushed in {total_latency_ms:.2f}ms "
            f"(target: <{self.flush_target_ms:.0f}ms). Preserved state for session {state.session_id}."
        )
        return total_latency_ms


__all__ = [
    "ParalinguisticMarkers",
    "ConversationalState",
    "InterruptionDetector",
]
