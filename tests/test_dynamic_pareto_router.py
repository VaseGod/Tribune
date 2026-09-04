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


def test_multi_objective_pareto_routing_sweeps():
    """Verify multi-objective dynamic Pareto-frontier optimization across TTFT, ITL, and unit cost sweeps."""
    router = ModelRouter()

    # 1. Cost-prioritized sweep (cost_weight=0.90, ttft=0.05, itl=0.05) -> should pick lowest unit cost (Tier 0 local dense)
    dec_cost = router.route_dynamic_pareto(
        intent="batch_processing",
        ttft_weight=0.05,
        itl_weight=0.05,
        cost_weight=0.90,
    )
    assert dec_cost.selected_endpoint in ("qwen3.8-27b", "qwen2.5-7b")
    assert dec_cost.unit_cost_usd_per_1k <= 0.0001

    # 2. ITL-prioritized sweep (streaming / interactive generation: itl_weight=0.85) -> should pick lowest ITL
    dec_itl = router.route_dynamic_pareto(
        intent="streaming_interactive",
        ttft_weight=0.10,
        itl_weight=0.85,
        cost_weight=0.05,
    )
    assert dec_itl.selected_endpoint == "qwen2.5-7b"
    assert dec_itl.itl_ms <= 7.0

    # 3. TTFT-prioritized sweep (fast time-to-first-token: ttft_weight=0.85) -> should pick lowest TTFT
    dec_ttft = router.route_dynamic_pareto(
        intent="urgent_triage",
        ttft_weight=0.85,
        itl_weight=0.10,
        cost_weight=0.05,
    )
    assert dec_ttft.selected_endpoint in ("qwen3.8-27b", "qwen2.5-7b")
    assert dec_ttft.ttft_ms <= 45.0


def test_live_telemetry_prefill_vs_decode_ingestion():
    """Verify live telemetry ingestion differentiating prefill overhead (TTFT) from decode overhead (ITL)."""
    router = ModelRouter()

    # Ingest telemetry for a new speculative endpoint
    ep = router.ingest_live_telemetry(
        endpoint_name="vllm-speculative-eagle3",
        ttft_ms=50.0,
        itl_ms=5.0,
        unit_cost_usd_per_1k=0.00015,
        speculative_supported=True,
        speculative_method="EAGLE_3",
        tier=1,
    )
    assert ep.endpoint_name == "vllm-speculative-eagle3"
    assert ep.speculative_supported is True
    assert ep.speculative_method == "EAGLE_3"

    # Verify Pareto frontier includes new high-performing endpoint
    frontier = router.compute_dynamic_pareto_frontier(require_speculative=True)
    names = [e.endpoint_name for e in frontier]
    assert "vllm-speculative-eagle3" in names

    # Routing with generation throughput focus should select this new endpoint
    dec = router.route_dynamic_pareto(
        intent="statutory_determination",
        ttft_weight=0.20,
        itl_weight=0.60,
        cost_weight=0.20,
        require_speculative=True,
    )
    assert dec.selected_endpoint == "vllm-speculative-eagle3"
    assert dec.speculative_method == "EAGLE_3"


def test_speculative_statutory_routing_mtp_eagle3():
    """Verify complex statutory queries route to speculative decoding endpoints (MTP k=3 / EAGLE-3)."""
    router = ModelRouter()

    # Complex statutory ambiguity query
    res = router.route_statutory_speculative(
        query="Applicant appeals denial due to conflicting custody arrangements under SNAP deduction rules",
        intent="statutory_ambiguity",
        context="7 CFR 273.9 deduction limits and household composition criteria under dispute...",
    )
    assert res["status"] == "routed"
    assert res["speculative_decoding"] is True
    assert res["speculative_method"] in ("MTP_3", "EAGLE_3", "draft_model")
    assert router.stats["speculative_draft_calls"] >= 1


def test_client_side_speculative_decoding_and_retry_recovery():
    """Verify client-side speculative decoding config, 429/503 retry recovery, and deterministic Idempotency-Key."""
    import urllib.error
    from tribune.providers.llm_client import (
        BaseHTTPProviderAdapter,
        LLMCompletionRequest,
        LocalRulesLLMAdapter,
        SpeculativeDecodingConfig,
    )

    # 1. Test client-side MTP draft depth k=3 and EAGLE-3 configuration
    cfg_mtp = SpeculativeDecodingConfig(method="MTP", draft_depth=3)
    assert cfg_mtp.draft_depth == 3
    assert cfg_mtp.method == "MTP"

    cfg_eagle = SpeculativeDecodingConfig(method="EAGLE-3", draft_depth=3, eagle_version="EAGLE-3")
    assert cfg_eagle.method == "EAGLE-3"

    # 2. LocalRulesLLMAdapter emits speculative metrics when configured
    adapter = LocalRulesLLMAdapter()
    req = LLMCompletionRequest(
        messages=[{"role": "user", "content": "Check eligibility"}],
        speculative_config=cfg_mtp,
    )
    resp = adapter.complete(req)
    assert resp.speculative_metrics["draft_depth"] == 3
    assert resp.speculative_metrics["verified"] is True
    assert resp.speculative_metrics["speedup_factor"] > 1.0

    # 3. Verify HTTP 429 Retry-After parsing and deterministic Idempotency-Key propagation
    class MockResponseHeaders(dict):
        def get(self, key, default=None):
            for k, v in self.items():
                if k.lower() == key.lower():
                    return v
            return default

    http_adapter = BaseHTTPProviderAdapter(
        provider_name="test_provider",
        base_url="http://mock-api.local",
        api_key="test_key",
        model="test-model",
        max_retries=2,
        base_backoff_sec=0.01,
        max_backoff_sec=0.05,
    )

    parsed_retry = http_adapter.parse_retry_after(MockResponseHeaders({"Retry-After": "0.02"}))
    assert parsed_retry == 0.02

    # Verify deterministic Idempotency-Key generation and propagation
    attempts_recorded = []
    idempotency_keys_seen = []

    def mock_urlopen(request, timeout):
        attempts_recorded.append(time.time())
        idempotency_keys_seen.append(request.get_header("Idempotency-key") or request.headers.get("Idempotency-key") or request.headers.get("Idempotency-Key"))
        if len(attempts_recorded) < 2:
            raise urllib.error.HTTPError(
                url="http://mock-api.local",
                code=429,
                msg="Rate Limited",
                hdrs=MockResponseHeaders({"Retry-After": "0.01"}),
                fp=None,
            )
        # Succeed on 2nd attempt
        import io
        return io.BytesIO(b'{"choices": [{"message": {"content": "Verified successfully"}}]}')

    import unittest.mock
    with unittest.mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        payload = {"test": "data", "query": "statutory"}
        body, latency = http_adapter._post_json(
            "http://mock-api.local/chat", payload, headers={"Content-Type": "application/json"}
        )
        assert body["choices"][0]["message"]["content"] == "Verified successfully"
        assert len(attempts_recorded) == 2
        # Deterministic Idempotency-Key UUID was preserved across retries
        assert len(idempotency_keys_seen) == 2
        assert idempotency_keys_seen[0] is not None
        assert idempotency_keys_seen[0] == idempotency_keys_seen[1]


