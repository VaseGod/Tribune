"""Unit and integration tests for FullDuplexAcousticPipeline, transports, and streaming ingestion."""

from __future__ import annotations

import pytest

from tribune.ingestion.acoustic import (
    AudioPacket,
    FullDuplexAcousticPipeline,
    LiveKitTransportAdapter,
    LocalLoopbackTransport,
    WebRTCTransportAdapter,
)


@pytest.mark.asyncio
async def test_full_duplex_pipeline_lifecycle() -> None:
    transport = LocalLoopbackTransport()
    pipeline = FullDuplexAcousticPipeline(transport=transport, session_id="test_sess_1")

    await pipeline.start()
    assert pipeline.telemetry.active_state == "connected"

    # Feed packet
    pkt = AudioPacket(
        payload=b"\x00\x01\x02\x03",
        timestamp_ms=100.0,
        seq=1,
        is_speech=True,
        volume=0.65,
        metadata={"partial_text": "hello caseworker"},
    )
    markers, partial_txt, is_final = await pipeline.process_inbound_packet(pkt)
    assert markers.voice_activity_level >= 0.60
    assert partial_txt == "hello caseworker"
    assert is_final is False

    await pipeline.stop()
    assert pipeline.telemetry.active_state == "disconnected"


@pytest.mark.asyncio
async def test_outbound_playback_and_sequential_fallback() -> None:
    transport = LocalLoopbackTransport()
    pipeline = FullDuplexAcousticPipeline(transport=transport)
    await pipeline.start()

    packets = [
        AudioPacket(payload=b"\x10\x20", timestamp_ms=0.0, seq=1),
        AudioPacket(payload=b"\x30\x40", timestamp_ms=20.0, seq=2),
    ]
    completed = await pipeline.play_outbound_audio(packets, system_text="Good morning.")
    assert completed is True
    assert pipeline.telemetry.turn_taking_latency_ms > 0.0

    # Sequential fallback mode
    raw_chunks = [b"chunk1", b"chunk2", b"chunk3"]
    seq_res = pipeline.process_sequential_chunks(raw_chunks)
    assert len(seq_res) == 3
    assert seq_res[0]["chunk_index"] == 0
    assert "delay_ms" in seq_res[0]

    await pipeline.stop()


@pytest.mark.asyncio
async def test_transport_adapters_interface() -> None:
    webrtc = WebRTCTransportAdapter()
    await webrtc.connect()
    pkt = AudioPacket(payload=b"\x01", timestamp_ms=0.0)
    await webrtc.send_packet(pkt)
    flush_ms = await webrtc.flush()
    assert flush_ms >= 0.0
    await webrtc.close()

    livekit = LiveKitTransportAdapter()
    await livekit.connect()
    await livekit.send_packet(pkt)
    flush_lk = await livekit.flush()
    assert flush_lk >= 0.0
    await livekit.close()
