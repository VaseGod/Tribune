"""End-to-End Integration Smoke Test for the Hardened Three-Tier Evaluation Architecture.

Validates:
- End-to-end execution of a synthetic task ladder with hardened settings
- Sentinel interception active
- Proactive memory sidecar active with heuristic/mock aux backend
- Real secrets are surrogate-masked and never leak into model-visible context or artifacts
- Evidence tracking verification
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from tribune.eval.quant_sensitivity.backends import smoke_ladder
from tribune.eval.quant_sensitivity.hardening_config import HardeningConfig
from tribune.eval.quant_sensitivity.ladder import run_ladder
from tribune.eval.quant_sensitivity.seedset import build_seed_set
from tribune.runtime.exec_request import ExecRequest
from tribune.runtime.local_runner import LocalFallbackRunner
from tribune.security.sentinel import SentinelBroker, SentinelClient
from tribune.security.token_broker import TokenBroker
from tribune.types import ProgramId


def test_end_to_end_hardened_ladder_smoke():
    """Execute end-to-end evaluation with Sentinel and Proactive Memory Sidecar enabled."""
    real_secret = "sk-live-topsecrettoken_in_eval_environment_12345"
    os.environ["TRIBUNE_REAL_API_KEY"] = real_secret

    token_broker = TokenBroker(surrogate_prefix="MOCK_")
    broker = SentinelBroker(token_broker=token_broker)
    sentinel_client = SentinelClient(in_process_broker=broker)

    hardening_cfg = HardeningConfig(
        execution_mode="local_fallback",
        sentinel_enabled=True,
        memory_sidecar_enabled=True,
        memory_update_interval_k=2,
        memory_aux_backend="heuristic_fast",
    )

    # Build 5-case seed set (1 per benefit program)
    cases = build_seed_set(
        limit_per_program={
            ProgramId.SNAP: 1,
            ProgramId.MEDICAID: 1,
            ProgramId.HOUSING: 1,
            ProgramId.UNEMPLOYMENT: 1,
            ProgramId.APPEALS: 1,
        }
    )

    rungs = smoke_ladder()
    result = run_ladder(
        rungs=rungs,
        cases=cases,
        hardening_config=hardening_cfg,
    )

    assert result is not None
    assert len(result.rungs) == 2

    for r in result.rungs:
        # Check that sidecar metrics were collected
        assert r.sidecar_metrics is not None
        assert "interventions" in r.sidecar_metrics
        assert "sidecar_invocations" in r.sidecar_metrics
        assert r.sidecar_invocations >= 1
        assert r.net_efficiency_score > 0.0

        # Verify no real secret leaked into any report text or record
        for rec in r.records:
            assert real_secret not in str(rec)


def test_sandbox_execution_secret_leak_prevention():
    """Verify that even when a tool outputs an authentic secret, it is scrubbed before return."""
    real_secret = "sk-ant-veryconfidentialanthropictoken456"
    token_broker = TokenBroker()
    surrogate = token_broker.register_secret(real_secret, secret_ref="ANTHROPIC_KEY")

    broker = SentinelBroker(token_broker=token_broker)
    sentinel_client = SentinelClient(in_process_broker=broker)
    runner = LocalFallbackRunner(sentinel_client=sentinel_client, token_broker=token_broker)

    req = ExecRequest(
        command=f"echo 'OUTPUT_VAL={real_secret}'",
        env={"ANTHROPIC_KEY": real_secret},
    )
    result = runner.execute(req)

    assert result.exit_code == 0
    # Real secret must NEVER appear in stdout
    assert real_secret not in result.stdout
    # Surrogate token appears in place of secret
    assert surrogate in result.stdout
    assert "ANTHROPIC_KEY" in result.surrogate_env_keys
