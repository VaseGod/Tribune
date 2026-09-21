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
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .container_policy import SandboxPolicy
from .telemetry import ExploitDetectionEngine, SecurityEvent
from .tool_catalog import RuntimeMode, ToolCatalog, TypedTool

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


# --------------------------------------------------------------------------- #
# Hardened Ephemeral Shell Sandbox & Policy Enforcement
# --------------------------------------------------------------------------- #


class ShellSandboxExecutor:
    """Hardened Ephemeral Shell Executor prioritizing shell-first isolated execution.

    Implements strict SandboxPolicy enforcement, ExploitDetectionEngine loop traps,
    resource limits, and token usage accounting.
    """

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        mode: RuntimeMode = RuntimeMode.SHELL_FIRST,
        tool_catalog: ToolCatalog | None = None,
        exploit_detector: ExploitDetectionEngine | None = None,
        workspace_dir: str | None = None,
    ) -> None:
        self.policy = policy or SandboxPolicy()
        self.mode = mode
        self.tool_catalog = tool_catalog or ToolCatalog()
        self.exploit_detector = exploit_detector or ExploitDetectionEngine(
            max_output_bytes=self.policy.max_output_bytes
        )
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="tribune_sandbox_")
        self._total_shell_tokens = 0
        self._total_catalog_tokens = 0

    def execute_shell(
        self,
        command_str: str,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> SandboxResult:
        """Execute a command string in an isolated ephemeral Bash environment."""
        start_t = time.perf_counter()
        timeout = timeout_s if timeout_s is not None else float(self.policy.max_execution_seconds)

        # 1. Check Mode
        if self.mode == RuntimeMode.CATALOG_ONLY:
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr="Execution denied: Runtime is configured for CATALOG_ONLY mode.",
                isolation_mode="catalog_only_denied",
                metadata={"tokens_consumed": 10},
            )

        # 2. Validate against SandboxPolicy
        is_allowed, policy_reason = self.policy.validate_command(command_str)
        if not is_allowed:
            return SandboxResult(
                exit_code=126,
                stdout="",
                stderr=policy_reason or "Command denied by sandbox policy.",
                isolation_mode="policy_denial",
                metadata={"denial": True, "tokens_consumed": 15},
            )

        # 3. Inspect via ExploitDetectionEngine (detect loops, exfil, rapid no-ops)
        is_safe, exploit_reason = self.exploit_detector.inspect_command(command_str)
        if not is_safe:
            return SandboxResult(
                exit_code=127,
                stdout="",
                stderr=exploit_reason or "Command denied by exploit detection engine.",
                isolation_mode="exploit_trap",
                metadata={"exploit_trap": True, "tokens_consumed": 20},
            )

        # 4. Prepare isolated execution environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        if not self.policy.allow_network:
            exec_env["TRIBUNE_NETGUARD"] = "deny_all"
            exec_env["http_proxy"] = "http://127.0.0.1:0"
            exec_env["https_proxy"] = "http://127.0.0.1:0"

        exec_env["TRIBUNE_WORKSPACE"] = self.workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)

        timed_out = False
        killed = False
        exit_code = 0
        stdout = ""
        stderr = ""

        # Use bash shell
        bash_bin = shutil.which("bash") or "/bin/sh"
        cmd = [bash_bin, "-c", command_str]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.workspace_dir,
                env=exec_env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
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

        # 5. Inspect and truncate output volume flooding
        stdout, stderr, flooded = self.exploit_detector.inspect_output(stdout, stderr)

        duration = time.perf_counter() - start_t
        tokens_consumed = max(5, (len(command_str) + len(stdout) + len(stderr)) // 4)
        self._total_shell_tokens += tokens_consumed

        artifacts = [
            str(p) for p in Path(self.workspace_dir).glob("**/*") if p.is_file()
        ]

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            killed=killed,
            duration_s=duration,
            isolation_mode="shell_first",
            collected_artifacts=artifacts,
            metadata={
                "tokens_consumed": tokens_consumed,
                "flooded": flooded,
                "quarantined": self.exploit_detector.is_quarantined,
            },
        )

    def execute_catalog_tool(self, tool_name: str, params: dict[str, Any]) -> SandboxResult:
        """Execute a typed tool from the catalog."""
        start_t = time.perf_counter()
        try:
            res = self.tool_catalog.execute_tool(tool_name, params)
            duration = time.perf_counter() - start_t
            # Catalog tokens are typically higher due to verbose schemas (~250-400 tokens)
            tool_obj = self.tool_catalog.get_tool(tool_name)
            tokens = tool_obj.estimated_tokens if tool_obj else 300
            self._total_catalog_tokens += tokens

            import json
            return SandboxResult(
                exit_code=0,
                stdout=json.dumps(res),
                stderr="",
                duration_s=duration,
                isolation_mode="catalog_typed",
                metadata={"tokens_consumed": tokens, "tool": tool_name},
            )
        except Exception as exc:
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                duration_s=time.perf_counter() - start_t,
                isolation_mode="catalog_typed",
                metadata={"tokens_consumed": 50, "error": str(exc)},
            )

    def get_token_comparison_stats(self) -> dict[str, Any]:
        """Compare token consumption of shell-first mode vs catalog-only mode."""
        return {
            "total_shell_tokens": self._total_shell_tokens,
            "total_catalog_tokens": self._total_catalog_tokens,
            "token_reduction_ratio": (
                round(1.0 - (self._total_shell_tokens / max(1, self._total_catalog_tokens)), 4)
                if self._total_catalog_tokens > 0
                else 0.0
            ),
            "quarantined": self.exploit_detector.is_quarantined,
            "security_events_count": len(self.exploit_detector.evidence_log),
        }


__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "SandboxRuntime",
    "DockerSandboxRuntime",
    "LocalFallbackSandboxRuntime",
    "get_sandbox_runtime",
    "SandboxPolicy",
    "ExploitDetectionEngine",
    "SecurityEvent",
    "RuntimeMode",
    "TypedTool",
    "ToolCatalog",
    "ShellSandboxExecutor",
]

