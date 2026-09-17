"""Tests for the sandbox runtime controller, timeouts, and artifact collection."""

import os
import tempfile
import pytest
from sandbox.sandbox_runtime import (
    DockerSandboxRuntime,
    LocalFallbackSandboxRuntime,
    SandboxConfig,
    get_sandbox_runtime,
)


def test_sandbox_local_fallback_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = SandboxConfig(
            timeout_s=5.0,
            host_artifacts_dir=tmp_dir,
            mode="local_fallback",
        )
        runtime = LocalFallbackSandboxRuntime(config)

        res = runtime.execute(["python", "-c", "print('SANDBOX_OK')"])

        assert res.exit_code == 0
        assert "SANDBOX_OK" in res.stdout
        assert res.timed_out is False
        assert res.isolation_mode == "local_fallback"


def test_sandbox_timeout_kill_switch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Strict 1-second timeout
        config = SandboxConfig(
            timeout_s=1.0,
            host_artifacts_dir=tmp_dir,
            mode="local_fallback",
        )
        runtime = LocalFallbackSandboxRuntime(config)

        # Run process that sleeps for 5 seconds
        res = runtime.execute(["python", "-c", "import time; time.sleep(5)"])

        assert res.timed_out is True
        assert res.killed is True
        assert res.exit_code == -9


def test_sandbox_artifact_collection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = SandboxConfig(
            timeout_s=5.0,
            host_artifacts_dir=tmp_dir,
            mode="local_fallback",
        )
        runtime = LocalFallbackSandboxRuntime(config)

        # Script that writes an artifact file
        script = (
            "import os\n"
            "path = os.path.join(os.environ['TRIBUNE_ARTIFACTS_DIR'], 'result.txt')\n"
            "with open(path, 'w') as f: f.write('ARTIFACT_EVIDENCE')\n"
        )
        res = runtime.execute(["python", "-c", script])

        assert res.exit_code == 0
        assert len(res.collected_artifacts) >= 1
        assert any("result.txt" in a for a in res.collected_artifacts)

        # Verify artifact contents
        artifact_path = os.path.join(tmp_dir, "result.txt")
        with open(artifact_path, "r") as f:
            assert f.read() == "ARTIFACT_EVIDENCE"


def test_sandbox_factory_mode():
    config_fallback = SandboxConfig(mode="local_fallback")
    runtime = get_sandbox_runtime(config_fallback)
    assert isinstance(runtime, LocalFallbackSandboxRuntime)
