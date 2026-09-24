"""Local Fallback Runner with Surrogate Token Brokering and Sentinel Verification.

Provides subprocess-isolated execution for development and test environments where
Docker/container daemons are unavailable.
Enforces:
- Full surrogate token brokering (real secrets are never passed into child processes)
- Sentinel command validation and allowlist enforcement
- Automatic output stream redaction
- Explicit metadata marking: unsafe_for_production = True
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from ..security.sentinel import SentinelClient
from ..security.token_broker import TokenBroker, get_token_broker
from .exec_request import ExecRequest, ExecResult, ShellRunner

logger = logging.getLogger(__name__)


class LocalFallbackRunner(ShellRunner):
    """Subprocess runner for local development with surrogate token protection."""

    def __init__(
        self,
        sentinel_client: SentinelClient | None = None,
        token_broker: TokenBroker | None = None,
    ) -> None:
        self.sentinel = sentinel_client or SentinelClient()
        self.token_broker = token_broker or get_token_broker()

    def execute(self, request: ExecRequest) -> ExecResult:
        start_t = time.perf_counter()
        raw_cmd = request.normalized_command_str()

        # 1. Sentinel validation
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
                isolation_mode="local_fallback",
                unsafe_for_production=True,
            )

        # 2. Surrogate token brokering on environment
        surrogate_env, modified_keys = self.token_broker.surrogate_environment(
            request.env,
            session_id=request.session_id,
        )

        # 3. Mask command arguments
        masked_cmd, redacted_display_cmd = self.token_broker.mask_command(
            raw_cmd,
            session_id=request.session_id,
        )
        final_cmd_str = str(masked_cmd)

        # 4. Prepare execution environment
        exec_env = os.environ.copy()
        exec_env.update(surrogate_env)
        # Deny outbound network by default in environment flags
        exec_env["TRIBUNE_NETGUARD"] = "deny_all"
        exec_env["http_proxy"] = "http://127.0.0.1:0"
        exec_env["https_proxy"] = "http://127.0.0.1:0"

        cwd = request.cwd or os.getcwd()
        os.makedirs(cwd, exist_ok=True)

        bash_bin = shutil.which("bash") or "/bin/sh"
        cmd = [bash_bin, "-c", final_cmd_str]

        timed_out = False
        killed = False
        exit_code = 0
        raw_stdout = ""
        raw_stderr = ""

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=exec_env,
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

        # 5. Redact stdout and stderr
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
            isolation_mode="local_fallback",
            timed_out=timed_out,
            killed=killed,
            unsafe_for_production=True,
            metadata={"degraded_mode": True},
        )
