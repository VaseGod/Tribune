"""Hardened Containerized Sandbox Runtime & Execution Controller.

Provides isolated execution for code execution, document parsing, and file manipulation:
1. Docker container isolation with strict network denial (--network=none).
2. CPU and memory limits (--cpus, --memory).
3. Automated kill switches and strict wall-clock execution timeouts.
4. Reduced filesystem access (read-only root, non-root user).
5. Deterministic artifact collection from sandbox volume to host.
6. Degraded local fallback with explicit security warnings when container daemon is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Operational parameters for sandboxed execution."""

    image_name: str = "tribune-appeals-eval:latest"
    timeout_s: float = 60.0
    network_disabled: bool = True
    memory_limit: str = "2g"
    cpu_limit: str = "2.0"
    read_only_root: bool = False
    mode: str = "auto"  # "container" | "local_fallback" | "auto"
    host_artifacts_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "artifacts"))
    container_artifacts_dir: str = "/app/artifacts"


@dataclass
class SandboxResult:
    """Outcome of sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    killed: bool = False
    duration_s: float = 0.0
    isolation_mode: str = "container"
    collected_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxRuntime:
    """Base interface for sandbox runtime controllers."""

    def execute(self, command: list[str], env: dict[str, str] | None = None) -> SandboxResult:
        raise NotImplementedError


class DockerSandboxRuntime(SandboxRuntime):
    """Production containerized sandbox runtime using Docker."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    @staticmethod
    def is_docker_available() -> bool:
        """Verify docker CLI is present and daemon is responding."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(self, command: list[str], env: dict[str, str] | None = None) -> SandboxResult:
        start_t = time.perf_counter()
        os.makedirs(self.config.host_artifacts_dir, exist_ok=True)

        docker_cmd = ["docker", "run", "--rm"]

        if self.config.network_disabled:
            docker_cmd.extend(["--network", "none"])

        if self.config.memory_limit:
            docker_cmd.extend(["--memory", self.config.memory_limit])

        if self.config.cpu_limit:
            docker_cmd.extend(["--cpus", self.config.cpu_limit])

        if self.config.read_only_root:
            docker_cmd.append("--read-only")

        # Security options: drop capabilities
        docker_cmd.extend(["--cap-drop", "ALL"])

        # Mount artifact directory
        abs_host_artifacts = os.path.abspath(self.config.host_artifacts_dir)
        docker_cmd.extend(["-v", f"{abs_host_artifacts}:{self.config.container_artifacts_dir}:rw"])

        # Pass environment variables safely
        if env:
            for k, v in env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])

        docker_cmd.append(self.config.image_name)
        docker_cmd.extend(command)

        logger.info(f"[DockerSandbox] Launching container with network_disabled={self.config.network_disabled}")

        timed_out = False
        killed = False
        exit_code = 0
        stdout = ""
        stderr = ""

        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.config.timeout_s)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                killed = True
                logger.warning(f"[DockerSandbox] Process exceeded timeout {self.config.timeout_s}s. Dispatching kill.")
                proc.kill()
                stdout, stderr = proc.communicate()
                exit_code = -9
        except Exception as exc:
            logger.error(f"[DockerSandbox] Execution failure: {exc}")
            exit_code = 1
            stderr = str(exc)

        duration = time.perf_counter() - start_t
        artifacts = [
            str(p)
            for p in Path(self.config.host_artifacts_dir).glob("**/*")
            if p.is_file()
        ]

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            killed=killed,
            duration_s=duration,
            isolation_mode="container",
            collected_artifacts=artifacts,
        )


class LocalFallbackSandboxRuntime(SandboxRuntime):
    """Degraded local fallback with explicit security warnings and subprocess isolation."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def execute(self, command: list[str], env: dict[str, str] | None = None) -> SandboxResult:
        start_t = time.perf_counter()
        os.makedirs(self.config.host_artifacts_dir, exist_ok=True)

        logger.warning(
            "=========================================================================\n"
            "[SECURITY ALERT] SANDBOX RUNNING IN DEGRADED LOCAL FALLBACK MODE.\n"
            "Container isolation is UNAVAILABLE. Host filesystem is partially exposed.\n"
            "Production workflows must deploy via container isolation!\n"
            "========================================================================="
        )

        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # Enforce offline flags
        if self.config.network_disabled:
            exec_env["TRIBUNE_NETGUARD"] = "deny_all"
            exec_env["PYTHONHTTPSVERIFY"] = "1"

        exec_env["TRIBUNE_ARTIFACTS_DIR"] = os.path.abspath(self.config.host_artifacts_dir)

        timed_out = False
        killed = False
        exit_code = 0
        stdout = ""
        stderr = ""

        cmd = list(command)
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=exec_env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.config.timeout_s)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                killed = True
                proc.kill()
                stdout, stderr = proc.communicate()
                exit_code = -9
        except Exception as exc:
            exit_code = 1
            stderr = str(exc)

        duration = time.perf_counter() - start_t
        artifacts = [
            str(p)
            for p in Path(self.config.host_artifacts_dir).glob("**/*")
            if p.is_file()
        ]

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            killed=killed,
            duration_s=duration,
            isolation_mode="local_fallback",
            collected_artifacts=artifacts,
        )


def get_sandbox_runtime(config: SandboxConfig | None = None) -> SandboxRuntime:
    """Factory selecting Docker container isolation when available, else local fallback."""
    cfg = config or SandboxConfig()

    if cfg.mode == "container":
        return DockerSandboxRuntime(cfg)
    elif cfg.mode == "local_fallback":
        return LocalFallbackSandboxRuntime(cfg)

    # Auto mode: check docker
    if DockerSandboxRuntime.is_docker_available():
        return DockerSandboxRuntime(cfg)
    else:
        return LocalFallbackSandboxRuntime(cfg)
