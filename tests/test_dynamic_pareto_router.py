"""Unit & integration tests for Dynamic Pareto Cost & Model Routing with SLA Tracking."""

import time

from tribune.providers.router import ModelRouter, SLATracker


def test_sla_tracker_circuit_breaking_and_recovery():
    """Verify SLATracker trips circuit when latency or error rate breaches SLA threshold."""
    tracker = SLATracker(tier=1, sla_target_p95_ms=100.0, recovery_timeout_sec=0.1)

    # 1. Healthy calls
    for _ in range(10):
        tracker.record_call(latency_ms=50.0, is_error=False)
    assert tracker.is_healthy() is True
    assert tracker.circuit_open is False

    # 2. Record severe latency breaches (250ms > 200ms threshold)
    for _ in range(10):
        tracker.record_call(latency_ms=300.0, is_error=True)
    assert tracker.circuit_open is True
    assert tracker.is_healthy() is False

    # 3. Half-open recovery after timeout
    time.sleep(0.15)
    assert tracker.is_healthy() is True
    assert tracker.circuit_open is False


def test_dynamic_pareto_router_sla_escalation():
    """Verify that ModelRouter dynamically escalates when Tier 1 SLA is breached."""
    router = ModelRouter()

    # Manually trip Tier 1 SLA tracker
    router.sla_trackers[1].circuit_open = True
    router.sla_trackers[1].circuit_opened_at = time.time()

    # Routing routine parsing task should escalate from Tier 1 to Tier 2
    assigned_tier = router.classify_task(intent="parsing")
    assert assigned_tier == 2
    assert router.stats["sla_escalations"] >= 1
