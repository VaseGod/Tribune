"""Audio Transport Abstraction & Adapters for Full-Duplex Audio Streaming.

Provides pluggable transport hooks for:
1. Local testing loopback transport (in-memory async queue for local tests & CI)
2. WebRTC media stream transport adapter
3. LiveKit room and audio track transport adapter
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class TransportState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class AudioPacket:
    """A discrete streaming audio frame or packet."""

    payload: bytes
    timestamp_ms: float
    seq: int = 0
    is_speech: bool = False
    volume: float = 0.0  # RMS or peak amplitude [0.0, 1.0]
    duration_ms: float = 20.0  # e.g. 20ms frame
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AudioTransport(Protocol):
    """Pluggable transport protocol for full-duplex bidirectional audio."""

    async def connect(self) -> None: ...

    async def send_packet(self, packet: AudioPacket) -> None: ...

    async def receive_packet(self) -> AudioPacket | None: ...

    async def flush(self) -> float:
        """Flush any pending outgoing packets. Returns flush duration in ms."""
        ...

    async def close(self) -> None: ...

    @property
    def state(self) -> TransportState: ...


class LocalLoopbackTransport:
    """In-memory async queue transport for unit testing and offline development."""

    def __init__(self, buffer_size: int = 100) -> None:
        self.inbound_queue: asyncio.Queue[AudioPacket] = asyncio.Queue(maxsize=buffer_size)
        self.outbound_queue: asyncio.Queue[AudioPacket] = asyncio.Queue(maxsize=buffer_size)
        self._state = TransportState.IDLE
        self.dropped_packets = 0

    async def connect(self) -> None:
        self._state = TransportState.CONNECTED

    async def send_packet(self, packet: AudioPacket) -> None:
        """Send outbound audio packet (system speech to client)."""
        try:
            self.outbound_queue.put_nowait(packet)
            self._state = TransportState.STREAMING
        except asyncio.QueueFull:
            self.dropped_packets += 1
            logger.warning("[LocalLoopbackTransport] Outbound buffer full, dropped packet")

    async def push_inbound_packet(self, packet: AudioPacket) -> None:
        """Simulate client microphone input packet for tests."""
        try:
            self.inbound_queue.put_nowait(packet)
        except asyncio.QueueFull:
            self.dropped_packets += 1

    async def receive_packet(self) -> AudioPacket | None:
        """Receive next inbound packet from client."""
        try:
            return await asyncio.wait_for(self.inbound_queue.get(), timeout=0.1)
        except (TimeoutError, asyncio.CancelledError):
            return None

    async def flush(self) -> float:
        """Rapidly drain/flush all queued outbound packets (<50ms target)."""
        start_t = asyncio.get_event_loop().time()
        drained = 0
        while not self.outbound_queue.empty():
            try:
                self.outbound_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        flush_duration_ms = (asyncio.get_event_loop().time() - start_t) * 1000.0
        logger.debug(f"[LocalLoopbackTransport] Flushed {drained} outbound packets in {flush_duration_ms:.2f}ms")
        return flush_duration_ms

    async def close(self) -> None:
        self._state = TransportState.DISCONNECTED

    @property
    def state(self) -> TransportState:
        return self._state


class WebRTCTransportAdapter:
    """Production WebRTC audio track transport adapter stub."""

    def __init__(self, rtc_peer_connection: Any = None) -> None:
        self.pc = rtc_peer_connection
        self._state = TransportState.IDLE
        self.outbound_buffer: list[AudioPacket] = []

    async def connect(self) -> None:
        self._state = TransportState.CONNECTED

    async def send_packet(self, packet: AudioPacket) -> None:
        self._state = TransportState.STREAMING
        self.outbound_buffer.append(packet)

    async def receive_packet(self) -> AudioPacket | None:
        await asyncio.sleep(0.01)
        return None

    async def flush(self) -> float:
        start_t = asyncio.get_event_loop().time()
        self.outbound_buffer.clear()
        flush_ms = (asyncio.get_event_loop().time() - start_t) * 1000.0
        return flush_ms

    async def close(self) -> None:
        self._state = TransportState.DISCONNECTED

    @property
    def state(self) -> TransportState:
        return self._state


class LiveKitTransportAdapter:
    """Production LiveKit audio room and track transport adapter stub."""

    def __init__(self, room: Any = None, participant_id: str = "") -> None:
        self.room = room
        self.participant_id = participant_id
        self._state = TransportState.IDLE
        self.outbound_buffer: list[AudioPacket] = []

    async def connect(self) -> None:
        self._state = TransportState.CONNECTED

    async def send_packet(self, packet: AudioPacket) -> None:
        self._state = TransportState.STREAMING
        self.outbound_buffer.append(packet)

    async def receive_packet(self) -> AudioPacket | None:
        await asyncio.sleep(0.01)
        return None

    async def flush(self) -> float:
        start_t = asyncio.get_event_loop().time()
        self.outbound_buffer.clear()
        flush_ms = (asyncio.get_event_loop().time() - start_t) * 1000.0
        return flush_ms

    async def close(self) -> None:
        self._state = TransportState.DISCONNECTED

    @property
    def state(self) -> TransportState:
        return self._state


__all__ = [
    "TransportState",
    "AudioPacket",
    "AudioTransport",
    "LocalLoopbackTransport",
    "WebRTCTransportAdapter",
    "LiveKitTransportAdapter",
]
