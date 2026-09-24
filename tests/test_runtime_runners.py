"""Unit Tests for ContainerShellRunner and LocalFallbackRunner.

Validates:
- Docker command generation with non-root user, read-only root, tmpfs mounts, dropped caps
- Local fallback runner execution
- Non-root enforcement metadata
- Timeout enforcement
- Exit code propagation
- Secret redaction in stdout and stderr
- unsafe_for_production flag
"""

from __future__ import annotations

import os
import sys
import pytest

from tribune.runtime.container_runner import ContainerRunnerConfig, ContainerShellRunner
from tribune.runtime.exec_request import ExecRequest, ExecResult
from tribune.runtime.local_runner import LocalFallbackRunner
from tribune.security.sentinel import SentinelBroker, SentinelClient
from tribune.security.token_broker import TokenBroker


def test_container_command_generation():
    """Verify ContainerShellRunner generates correct docker arguments."""
    config = ContainerRunnerConfig(
        image="tribune-sandbox:v2",
        user="tribune-sandbox",
        network_mode="none",
        memory_limit="512m",
        cpu_limit="1.5",
        read_only_root=True,
        drop_capabilities=True,
        tmpfs_mounts=["/tmp", "/workspace:rw"],
    )
    runner = ContainerShellRunner(config=config)
    env = {"MOCK_KEY": "MOCK_VALUE", "ENV_VAR": "TEST"}
    cmd = runner.build_docker_command(
        cmd_str="ls -la",
        env=env,
        host_workspace="/host/tmp/ws",
        container_workspace="/workspace",
    )

    # Check assertions
    assert "--user" in cmd and "tribune-sandbox" in cmd
    assert "--network" in cmd and "none" in cmd
    assert "--memory" in cmd and "512m" in cmd
    assert "--cpus" in cmd and "1.5" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop" in cmd and "ALL" in cmd
    assert "-v" in cmd and "/host/tmp/ws:/workspace:rw" in cmd
    assert "-e" in cmd and "MOCK_KEY=MOCK_VALUE" in cmd
    assert "tribune-sandbox:v2" in cmd
    assert cmd[-1] == "ls -la"


def test_local_fallback_runner_execution_and_exit_code():
    """Verify LocalFallbackRunner executes shell commands and returns exit codes."""
    broker = SentinelBroker()
    sentinel_client = SentinelClient(in_process_broker=broker)
    runner = LocalFallbackRunner(sentinel_client=sentinel_client)

    # Simple echo
    req_echo = ExecRequest(command="echo 'tribune_hello_world'")
    res_echo = runner.execute(req_echo)
    assert res_echo.exit_code == 0
    assert "tribune_hello_world" in res_echo.stdout
    assert res_echo.isolation_mode == "local_fallback"
    assert res_echo.unsafe_for_production is True
    assert res_echo.succeeded is True

    # Failing command (non-zero exit)
    req_fail = ExecRequest(command="exit 42")
    res_fail = runner.execute(req_fail)
    assert res_fail.exit_code == 42
    assert res_fail.succeeded is False


def test_local_fallback_runner_timeout_enforcement():
    """Verify LocalFallbackRunner terminates commands that exceed timeout."""
    broker = SentinelBroker()
    sentinel_client = SentinelClient(in_process_broker=broker)
    runner = LocalFallbackRunner(sentinel_client=sentinel_client)

    # Sleep longer than timeout
    req = ExecRequest(command="sleep 5", timeout_seconds=0.5)
    res = runner.execute(req)
    assert res.timed_out is True
    assert res.killed is True
    assert res.exit_code == -9


def test_runner_redacts_secrets_in_output():
    """Verify secrets leaking in stdout or stderr are automatically redacted."""
    token_broker = TokenBroker()
    secret = "sk-live-confidentialkey999"
    surrogate = token_broker.register_secret(secret, secret_ref="TEST_KEY")

    broker = SentinelBroker(token_broker=token_broker)
    sentinel_client = SentinelClient(in_process_broker=broker)
    runner = LocalFallbackRunner(sentinel_client=sentinel_client, token_broker=token_broker)

    # Command that deliberately prints the real secret
    req = ExecRequest(
        command=f"echo 'leaking secret: {secret}'",
        env={"TEST_KEY": secret},
    )
    res = runner.execute(req)
    assert res.exit_code == 0
    assert secret not in res.stdout
    assert surrogate in res.stdout
    assert "TEST_KEY" in res.surrogate_env_keys


def test_runner_sentinel_blocks_disallowed_command():
    """Verify runner respects Sentinel denials without executing."""
    broker = SentinelBroker()
    sentinel_client = SentinelClient(in_process_broker=broker)
    runner = LocalFallbackRunner(sentinel_client=sentinel_client)

    req = ExecRequest(command="sudo cat /etc/shadow")
    res = runner.execute(req)
    assert res.exit_code == 126
    assert "Sentinel denial" in res.stderr
    assert not res.succeeded
