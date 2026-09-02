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


def test_nemo_switchyard_step_routing_operational_vs_cognitive():
    """Verify that StepRouter routes lightweight tasks to high-throughput endpoints and cognitive tasks to frontier engines."""
    from tribune.providers.router import ModelRouter, OperationalStepType, StepRouter

    step_router = StepRouter(
        high_throughput_model="gemini-3.7-flash",
        frontier_reasoning_model="deepseek-v4-pro",
    )

    # 1. Lightweight operational tasks -> High throughput endpoint (Tier 1)
    dec_fold = step_router.route_step(OperationalStepType.CONTEXT_FOLD_PLANNING)
    assert dec_fold.tier == 1
    assert dec_fold.target_model == "gemini-3.7-flash"
    assert dec_fold.endpoint_category == "high_throughput"
    assert dec_fold.is_frontier_reasoning is False

    dec_json = step_router.route_step(OperationalStepType.JSON_EXTRACTION)
    assert dec_json.tier == 1
    assert dec_json.endpoint_category == "high_throughput"

    dec_param = step_router.route_step(OperationalStepType.TOOL_PARAM_FORMATTING)
    assert dec_param.tier == 1
    assert dec_param.endpoint_category == "high_throughput"

    # 2. Deep cognitive tasks -> Frontier reasoning engine (Tier 2)
    dec_arch = step_router.route_step(OperationalStepType.ARCHITECTURAL_DESIGN)
    assert dec_arch.tier == 2
    assert dec_arch.target_model == "deepseek-v4-pro"
    assert dec_arch.endpoint_category == "frontier_reasoning"
    assert dec_arch.is_frontier_reasoning is True

    dec_synth = step_router.route_step(OperationalStepType.CODE_SYNTHESIS)
    assert dec_synth.tier == 2
    assert dec_synth.is_frontier_reasoning is True

    dec_sec = step_router.route_step(OperationalStepType.SECURITY_VERIFICATION)
    assert dec_sec.tier == 2
    assert dec_sec.is_frontier_reasoning is True

    # 3. ModelRouter wrapper execution
    router = ModelRouter()
    step_dec = router.route_operational_step(OperationalStepType.JSON_EXTRACTION, context_tokens=150)
    assert step_dec.tier == 1
    assert router.stats["step_routing_calls"] >= 1

