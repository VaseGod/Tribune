"""Unit tests for interruption handling, fast flush latency (<50ms), and background tool execution."""

from __future__ import annotations

import asyncio
import time

import pytest

from tribune.ingestion.acoustic import (
    AudioPacket,
    FullDuplexAcousticPipeline,
    LocalLoopbackTransport,
)


@pytest.mark.asyncio
async def test_simulated_interruption_fast_flush() -> None:
    """Verify incoming speech during playback triggers flush <50ms and preserves state."""
    transport = LocalLoopbackTransport()
    pipeline = FullDuplexAcousticPipeline(transport=transport, session_id="sess_interrupt_test")
    await pipeline.start()

    # Pre-populate outbound queue with pending audio packets to simulate active playback
    for i in range(10):
        await transport.send_packet(AudioPacket(payload=b"\xaa\xbb", timestamp_ms=i * 20.0, seq=i))

    pipeline.is_system_playing = True
    pipeline.state.last_system_utterance = "Your SNAP certification will be renewed based on 7 CFR 273.9."
    pipeline.state.active_topic = "SNAP recertification"

    # User speaks loudly to interrupt
    barge_in_packet_1 = AudioPacket(
        payload=b"\xff\xee",
        timestamp_ms=150.0,
        seq=1,
        is_speech=True,
        volume=0.85,
        metadata={"partial_text": "Wait, my rent changed!"},
    )
    barge_in_packet_2 = AudioPacket(
        payload=b"\xff\xee",
        timestamp_ms=170.0,
        seq=2,
        is_speech=True,
        volume=0.90,
        metadata={"partial_text": "Wait, my rent changed!"},
    )

    await pipeline.process_inbound_packet(barge_in_packet_1)
    markers, text, _ = await pipeline.process_inbound_packet(barge_in_packet_2)

    # Verify barge-in was detected
    assert markers.barge_in_event is True
    assert markers.interruption_urgency >= 0.80

    # Verify flush target <50ms
    flush_latency = pipeline.telemetry.interruption_flush_latency_ms
    assert flush_latency < 50.0, f"Flush latency {flush_latency}ms exceeded 50ms target!"

    # Outbound queue should be completely cleared
    assert transport.outbound_queue.empty() is True
    assert pipeline.is_system_playing is False

    # Verify conversational state was preserved across interrupt
    assert pipeline.state.system_playback_interrupted is True
    assert pipeline.state.active_topic == "SNAP recertification"
    assert "SNAP certification" in pipeline.state.last_system_utterance

    await pipeline.stop()


@pytest.mark.asyncio
async def test_background_tool_call_non_blocking() -> None:
    """Verify background tool calls execute asynchronously without blocking the audio loop."""
    pipeline = FullDuplexAcousticPipeline()
    await pipeline.start()

    # Simulate slow database query (100ms)
    async def slow_eligibility_lookup(case_id: str) -> dict[str, str]:
        await asyncio.sleep(0.1)
        return {"case_id": case_id, "status": "eligible", "fpl_ratio": "1.15"}

    start_t = time.perf_counter()
    call_id = await pipeline.dispatch_spoken_tool(
        "slow_eligibility_lookup",
        slow_eligibility_lookup,
        case_id="case_999",
    )
    dispatch_duration_ms = (time.perf_counter() - start_t) * 1000.0

    # Dispatch must be instantaneous (<15ms) without waiting for 100ms db query
    assert dispatch_duration_ms < 15.0, f"Dispatch took {dispatch_duration_ms}ms, blocking audio loop!"

    # While background tool runs, audio processing continues uninterrupted
    pkt = AudioPacket(payload=b"\x12\x34", timestamp_ms=50.0, seq=1, is_speech=False, volume=0.1)
    markers, _, _ = await pipeline.process_inbound_packet(pkt)
    assert markers.voice_activity_level < 0.2

    # Now wait for tool completion
    result = await pipeline.tool_executor.wait_for_call(call_id, timeout=2.0)
    assert result is not None
    assert result.success is True
    assert result.result["status"] == "eligible"
    assert pipeline.telemetry.background_tool_call_count == 1

    await pipeline.stop()
