"""Containerized Shell Runner for the Controlled Execution Tier.

Executes quantized model actions inside an unprivileged, sealed runtime container:
- Non-root user (default: tribune-sandbox / UID 10001)
- Minimal filesystem with read-only root and tmpfs for writable paths
- Capabilities dropped (--cap-drop=ALL)
- Network deny-by-default (--network=none)
- Sanitized environment containing ONLY surrogate tokens (MOCK_*)
- Output stream scanning and secret redaction
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..security.audit import SecurityAuditEvent, SecurityEventType
from ..security.sentinel import CommandDecision, SentinelClient
from ..security.token_broker import TokenBroker, get_token_broker
from .exec_request import ExecRequest, ExecResult, ShellRunner

logger = logging.getLogger(__name__)


@dataclass
class ContainerRunnerConfig:
    """Configuration options for ContainerShellRunner."""

    image: str = "tribune-sandbox:latest"
    user: str = "tribune-sandbox"  # Non-root user
    network_mode: str = "none"  # "none" | "sentinel_proxy"
    memory_limit: str = "1024m"
    cpu_limit: str = "2.0"
    read_only_root: bool = True
    tmpfs_mounts: list[str] = field(default_factory=lambda: ["/tmp", "/workspace:rw,size=512m"])
    drop_capabilities: bool = True
    host_workspace_base: str | None = None


class ContainerShellRunner(ShellRunner):
    """Executes commands inside an isolated, unprivileged container."""

    def __init__(
        self,
        config: ContainerRunnerConfig | None = None,
        sentinel_client: SentinelClient | None = None,
        token_broker: TokenBroker | None = None,
    ) -> None:
        self.config = config or ContainerRunnerConfig()
        self.sentinel = sentinel_client or SentinelClient()
        self.token_broker = token_broker or get_token_broker()

    @staticmethod
    def is_available() -> bool:
        """Verify whether Docker/container daemon is accessible."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
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

    def build_docker_command(
        self,
        cmd_str: str,
        env: dict[str, str],
        host_workspace: str,
        container_workspace: str = "/workspace",
    ) -> list[str]:
        """Construct the hardened docker execution command."""
        docker_cmd = ["docker", "run", "--rm"]

        # 1. Non-root user
        if self.config.user:
            docker_cmd.extend(["--user", self.config.user])

        # 2. Network isolation
        if self.config.network_mode == "none":
            docker_cmd.extend(["--network", "none"])

        # 3. Resource ceilings
        if self.config.memory_limit:
            docker_cmd.extend(["--memory", self.config.memory_limit])
        if self.config.cpu_limit:
            docker_cmd.extend(["--cpus", self.config.cpu_limit])

        # 4. Read-only root filesystem
        if self.config.read_only_root:
            docker_cmd.append("--read-only")

        # 5. Drop all capabilities
        if self.config.drop_capabilities:
            docker_cmd.extend(["--cap-drop", "ALL"])

        # 6. Tmpfs mounts for writable scratch
        for mount in self.config.tmpfs_mounts:
            docker_cmd.extend(["--tmpfs", mount])

        # 7. Bind workspace volume
        docker_cmd.extend(["-v", f"{host_workspace}:{container_workspace}:rw"])
        docker_cmd.extend(["-w", container_workspace])

        # 8. Environment variables (only sanitized surrogate tokens!)
        for k, v in env.items():
            docker_cmd.extend(["-e", f"{k}={v}"])

        docker_cmd.append(self.config.image)
        docker_cmd.extend(["/bin/bash", "-c", cmd_str])
        return docker_cmd

    def execute(self, request: ExecRequest) -> ExecResult:
        start_t = time.perf_counter()
        raw_cmd = request.normalized_command_str()

        # 1. Inspect command via Sentinel
        decision = self.sentinel.check_command(
            command=raw_cmd,
            session_id=request.session_id,
            task_id=request.task_id,
        )

        if not decision.allowed:
            return ExecResult(
                exit_code=126,
                stdout="",
                stderr=f"Sentinel denial: {decision.reason}",
                duration_seconds=time.perf_counter() - start_t,
                redacted_command=decision.redacted_command,
                sentinel_decision=decision.decision_code,
                isolation_mode="container",
            )

        # 2. Sanitize environment: surrogate all authentic secrets
        surrogate_env, modified_keys = self.token_broker.surrogate_environment(
            request.env,
            session_id=request.session_id,
        )

        # 3. Mask command line arguments
        masked_cmd, redacted_display_cmd = self.token_broker.mask_command(
            raw_cmd,
            session_id=request.session_id,
        )
        final_cmd_str = str(masked_cmd)

        # 4. Prepare host workspace directory
        host_workspace = request.cwd or os.path.join(
            self.config.host_workspace_base or "/tmp",
            f"tribune_work_{request.session_id}",
        )
        os.makedirs(host_workspace, exist_ok=True)

        docker_cmd = self.build_docker_command(
            cmd_str=final_cmd_str,
            env=surrogate_env,
            host_workspace=host_workspace,
        )

        # Log container spawn audit event
        if self.sentinel.in_process_broker and self.sentinel.in_process_broker.audit_logger:
            self.sentinel.in_process_broker.audit_logger.record(
                SecurityAuditEvent(
                    event_type=SecurityEventType.CONTAINER_SPAWNED,
                    severity="LOW",
                    source="container_runner",
                    message="Sandbox container spawned",
                    details={"session_id": request.session_id, "image": self.config.image},
                )
            )

        timed_out = False
        killed = False
        exit_code = 0
        raw_stdout = ""
        raw_stderr = ""

        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                raw_stdout, raw_stderr = proc.communicate(timeout=request.timeout_seconds)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                killed = True
                proc.kill()
                raw_stdout, raw_stderr = proc.communicate()
                exit_code = -9
        except Exception as exc:
            exit_code = 1
            raw_stderr = str(exc)

        # Log container exited audit event
        if self.sentinel.in_process_broker and self.sentinel.in_process_broker.audit_logger:
            self.sentinel.in_process_broker.audit_logger.record(
                SecurityAuditEvent(
                    event_type=SecurityEventType.CONTAINER_EXITED,
                    severity="LOW",
                    source="container_runner",
                    message=f"Sandbox container exited with code {exit_code}",
                    details={"session_id": request.session_id, "exit_code": exit_code},
                )
            )

        # 5. Redact stdout and stderr to guarantee no secret leakage
        clean_stdout = self.token_broker.redact_text(raw_stdout)
        clean_stderr = self.token_broker.redact_text(raw_stderr)

        duration = time.perf_counter() - start_t
        return ExecResult(
            exit_code=exit_code,
            stdout=clean_stdout,
            stderr=clean_stderr,
            duration_seconds=duration,
            redacted_command=redacted_display_cmd,
            surrogate_env_keys=modified_keys,
            sentinel_decision="ALLOWED",
            isolation_mode="container",
            timed_out=timed_out,
            killed=killed,
            unsafe_for_production=False,
            metadata={"image": self.config.image, "user": self.config.user},
        )
