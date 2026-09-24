"""Hardened Dual-Zone Runtime and Shell Execution Subsystem."""

from __future__ import annotations

import os

from .container_runner import ContainerRunnerConfig, ContainerShellRunner
from .exec_request import ExecRequest, ExecResult, ShellRunner
from .local_runner import LocalFallbackRunner


def create_shell_runner(
    mode: str | None = None,
    sentinel_client=None,
    token_broker=None,
    container_config: ContainerRunnerConfig | None = None,
) -> ShellRunner:
    """Factory creating appropriate shell runner based on mode or environment."""
    selected_mode = mode or os.environ.get("TRIBUNE_EXECUTION_MODE", "auto").lower()

    if selected_mode == "container":
        return ContainerShellRunner(
            config=container_config,
            sentinel_client=sentinel_client,
            token_broker=token_broker,
        )
    elif selected_mode == "local_fallback":
        return LocalFallbackRunner(
            sentinel_client=sentinel_client,
            token_broker=token_broker,
        )
    elif selected_mode == "legacy":
        # Legacy fallback still uses local runner for safety but with legacy flag
        return LocalFallbackRunner(
            sentinel_client=sentinel_client,
            token_broker=token_broker,
        )

    # Auto mode: use container if Docker daemon responds, otherwise local fallback
    if ContainerShellRunner.is_available():
        return ContainerShellRunner(
            config=container_config,
            sentinel_client=sentinel_client,
            token_broker=token_broker,
        )
    return LocalFallbackRunner(
        sentinel_client=sentinel_client,
        token_broker=token_broker,
    )


__all__ = [
    "ExecRequest",
    "ExecResult",
    "ShellRunner",
    "ContainerShellRunner",
    "ContainerRunnerConfig",
    "LocalFallbackRunner",
    "create_shell_runner",
]
