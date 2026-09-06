"""Hardened Execution Sandbox, Host Defenses, & Anti-Tamper Heartbeat Watchdog.

Enforces out-of-band authorization verification with HMAC-SHA256 tokens,
masks /proc and /sys virtual filesystems, blocks process enumeration utilities
(ps, top, pgrep, etc.), and runs an asynchronous heartbeat watchdog that triggers
immediate SIGKILL-equivalent containment termination upon runtime tampering.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from ..governance.action_gate import SecurityViolationError
from ..governance.gates import GatekeeperSupervisor
from ..instrumentation import tracing
from ..types import HMACAuthorizationToken

logger = logging.getLogger(__name__)

# Virtual filesystems masked from execution context
MASKED_FILESYSTEM_PATTERNS = [
    re.compile(r"(?:^|/)(?:proc|sys)(?:/|$)", re.IGNORECASE),
    re.compile(r"\.\./\.\./.*(?:proc|sys)", re.IGNORECASE),
]

# Process enumeration utilities blocked unconditionally
BLOCKED_PROCESS_UTILITIES = {
    "ps",
    "top",
    "pgrep",
    "pstree",
    "htop",
    "tasklist",
    "pidof",
    "prstat",
}


class ContainmentTerminationError(SecurityViolationError):
    """Raised when immediate SIGKILL-equivalent containment termination is triggered."""
    pass


class HeartbeatWatchdog:
    """Asynchronous heartbeat watchdog monitoring runtime environment variables,

    logger pipelines, and supervisor process trees for inspection or tampering.
    """

    def __init__(
        self,
        check_interval_s: float = 0.1,
        termination_callback: Callable[[], None] | None = None,
    ) -> None:
        self.check_interval_s = check_interval_s
        self.termination_callback = termination_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        # Capture baseline environment and logger configurations
        self._baseline_environ = dict(os.environ)
        self._baseline_logger_handlers_count = len(logging.getLogger().handlers)
        self._tamper_detected = False
        self._tamper_reason = ""

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._watchdog_loop,
                name="tribune-heartbeat-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)

    def verify_integrity(self) -> None:
        """Explicit single-shot check of runtime environment and logger integrity."""
        # 1. Check environment variable tampering
        for key in ("PATH", "PYTHONPATH", "OPENAI_API_KEY", "TRIBUNE_ENV"):
            if key in self._baseline_environ:
                if key not in os.environ:
                    self._trigger_containment_termination(f"Environment variable '{key}' was deleted")
                elif os.environ[key] != self._baseline_environ[key]:
                    self._trigger_containment_termination(f"Environment variable '{key}' was modified")

        # 2. Check logger pipeline suppression
        current_handlers = logging.getLogger().handlers
        if self._baseline_logger_handlers_count > 0 and len(current_handlers) == 0:
            self._trigger_containment_termination("Root logging handlers were wiped / suppressed")

        # Check for sys.modules tampering on critical security modules
        if "tribune.security.sandbox" not in sys.modules or sys.modules["tribune.security.sandbox"] is None:
            self._trigger_containment_termination("Security module sandbox hijacked in sys.modules")

    def _watchdog_loop(self) -> None:
        while self._running:
            try:
                self.verify_integrity()
            except ContainmentTerminationError:
                break
            except Exception as exc:
                logger.error(f"[WATCHDOG] Unexpected error in heartbeat: {exc}")
            time.sleep(self.check_interval_s)

    def _trigger_containment_termination(self, reason: str) -> None:
        self._tamper_detected = True
        self._tamper_reason = reason

        # Emit critical security telemetry event
        tracing.log(
            "critical_security_tamper_detected",
            event_type="RUNTIME_TAMPERING",
            severity="CRITICAL",
            reason=reason,
            timestamp=time.time(),
        )

        logger.critical(
            f"[SECURITY-TAMPER] Containment termination triggered: {reason}. Emitting telemetry and terminating."
        )

        if self.termination_callback:
            self.termination_callback()

        raise ContainmentTerminationError(
            f"SIGKILL containment termination triggered: runtime tampering detected ({reason})."
        )


class HardenedExecutionSandbox:
    """Hardened execution boundary enforcing HMAC authorization tokens and host defenses."""

    def __init__(
        self,
        supervisor: GatekeeperSupervisor | None = None,
        enable_watchdog: bool = True,
    ) -> None:
        self.supervisor = supervisor or GatekeeperSupervisor()
        self._received_tokens: dict[str, HMACAuthorizationToken] = {}
        self._lock = threading.RLock()

        self.watchdog = HeartbeatWatchdog()
        if enable_watchdog:
            self.watchdog.start()

    def receive_authorization_token(self, token: HMACAuthorizationToken) -> None:
        """Receive out-of-band authorization token minted by supervisor."""
        with self._lock:
            self._received_tokens[token.proposal_id] = token

    def execute_high_stakes_tool(
        self,
        tool_name: str,
        tool_fn: Callable[..., Any],
        kwargs: dict[str, Any],
        authorization_token: HMACAuthorizationToken | None = None,
        proposal_id: str = "",
        target_resource: str = "",
        action_type: str = "formal_filing",
    ) -> Any:
        """Verify supervisor HMAC token, enforce host defenses, and execute high-stakes tool."""
        with self._lock:
            # Check heartbeat watchdog integrity first
            self.watchdog.verify_integrity()

            # 1. Verify HMAC Authorization Token
            token = authorization_token or self._received_tokens.get(proposal_id)
            if token is None:
                tracing.log("security_rejection", reason="missing_authorization_token", tool=tool_name)
                raise SecurityViolationError(
                    f"Execution rejected: Missing HMAC authorization token for high-stakes tool '{tool_name}'."
                )

            is_valid = self.supervisor.verify_token(
                token=token,
                proposal_id=proposal_id,
                action_type=action_type,
                target_resource=target_resource,
            )
            if not is_valid:
                tracing.log("security_rejection", reason="invalid_authorization_token", tool=tool_name)
                raise SecurityViolationError(
                    f"Execution rejected: Invalid, expired, or mismatched HMAC token for proposal '{proposal_id}'."
                )

            # 2. Host Defense: Mask /proc and /sys virtual filesystems
            self._enforce_filesystem_masking(tool_name, target_resource, kwargs)

            # 3. Host Defense: Block process enumeration utilities
            self._enforce_process_enumeration_blocking(tool_name, kwargs)

            # Safe execution
            try:
                return tool_fn(**kwargs)
            except Exception as exc:
                logger.error(f"[SANDBOX] Execution error in {tool_name}: {exc}")
                raise

    def _enforce_filesystem_masking(
        self, tool_name: str, target_resource: str, kwargs: dict[str, Any]
    ) -> None:
        """Inspect all inputs and block any access to /proc or /sys virtual filesystems."""
        candidates = [tool_name, target_resource]
        for val in kwargs.values():
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list | dict):
                candidates.append(str(val))

        for text in candidates:
            for pattern in MASKED_FILESYSTEM_PATTERNS:
                if pattern.search(text):
                    tracing.log(
                        "host_defense_violation",
                        violation_type="MASKED_FILESYSTEM_ACCESS",
                        matched_text=text,
                    )
                    raise SecurityViolationError(
                        f"Host defense violation: Access to /proc or /sys virtual filesystems is strictly masked ({text})."
                    )

    def _enforce_process_enumeration_blocking(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> None:
        """Inspect all command arguments and block process enumeration utilities."""
        candidates = [tool_name]
        for val in kwargs.values():
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list):
                candidates.extend([str(item) for item in val])

        for text in candidates:
            tokens = re.split(r"[\s\|\;&]+", text.strip())
            for tok in tokens:
                clean_tok = os.path.basename(tok).lower()
                if clean_tok in BLOCKED_PROCESS_UTILITIES:
                    tracing.log(
                        "host_defense_violation",
                        violation_type="PROCESS_ENUMERATION_BLOCKED",
                        command=clean_tok,
                    )
                    raise SecurityViolationError(
                        f"Host defense violation: Process enumeration utility '{clean_tok}' is strictly blocked."
                    )

    def close(self) -> None:
        self.watchdog.stop()


__all__ = [
    "HardenedExecutionSandbox",
    "HeartbeatWatchdog",
    "ContainmentTerminationError",
    "SecurityViolationError",
]
