"""Sentinel Security Daemon and Out-of-Band Policy Broker.

Intercepts execution requests, enforces deny-by-default egress allowlists,
manages authentic credentials strictly out-of-band, injects host-side authentication
headers invisibly to the sandboxed runtime, and records structured audit events.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .allowlist import AllowlistPolicy, default_allowlist
from .audit import SecurityAuditEvent, SecurityAuditLogger, SecurityEventType
from .token_broker import TokenBroker, get_token_broker

logger = logging.getLogger(__name__)

# Patterns blocked unconditionally by Sentinel command inspection
DISALLOWED_COMMAND_PATTERNS = [
    re.compile(r"\b(?:sudo|su|doas)\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+[+]?s\b", re.IGNORECASE),
    re.compile(r"/(?:etc/shadow|etc/gshadow|etc/sudoers)", re.IGNORECASE),
    re.compile(r"/var/run/docker\.sock", re.IGNORECASE),
    re.compile(r"\brm\s+-(?:r[fF]|fr)\s+/(?:\s|$)", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", re.IGNORECASE),  # Fork bomb
    re.compile(r"(?:curl|wget)\s+[^|]+\|\s*(?:bash|sh|python)", re.IGNORECASE),  # Remote script pipe
    re.compile(r"\b(?:printenv|export\s+-p|env\b\s*$)", re.IGNORECASE),  # Environment dumping
]

# Network commands that require explicit allowlist verification
NETWORK_COMMAND_PATTERNS = [
    re.compile(r"\b(?:curl|wget|nc|ncat|netcat|telnet|ssh|scp|sftp|ftp)\s+([^\s]+)", re.IGNORECASE),
]


@dataclass
class CommandDecision:
    """Decision outcome of a shell command policy evaluation."""

    allowed: bool
    reason: str
    redacted_command: str
    decision_code: str = "ALLOWED"  # "ALLOWED" | "DENIED_DISALLOWED_PATTERN" | "DENIED_NETWORK_VIOLATION"


@dataclass
class EgressDecision:
    """Decision outcome of a network egress policy evaluation."""

    allowed: bool
    reason: str
    auth_profile: str = "none"
    matched_rule_id: str | None = None
    injected_headers: dict[str, str] = field(default_factory=dict)


class SentinelBroker:
    """Core Sentinel security logic running in the host tier."""

    def __init__(
        self,
        allowlist: AllowlistPolicy | None = None,
        token_broker: TokenBroker | None = None,
        audit_logger: SecurityAuditLogger | None = None,
        host_vault: dict[str, str] | None = None,
    ) -> None:
        self.allowlist = allowlist or default_allowlist()
        self.token_broker = token_broker or get_token_broker()
        self.audit_logger = audit_logger or SecurityAuditLogger()
        self.host_vault = dict(host_vault or {})
        self._lock = threading.RLock()

    def register_host_credential(self, secret_ref: str, real_secret: str) -> None:
        """Store an authentic credential in the host-only vault."""
        with self._lock:
            self.host_vault[secret_ref] = real_secret

    def evaluate_command(
        self,
        command: str,
        session_id: str = "default_session",
        task_id: str = "default_task",
    ) -> CommandDecision:
        """Inspect and decide whether a shell command is allowed to execute."""
        redacted_cmd = self.token_broker.redact_text(command)

        # Audit: command requested
        self.audit_logger.record(
            SecurityAuditEvent(
                event_type=SecurityEventType.COMMAND_REQUESTED,
                severity="LOW",
                source="sentinel",
                message=f"Command requested: {redacted_cmd[:100]}",
                details={"session_id": session_id, "task_id": task_id, "command_preview": redacted_cmd[:200]},
            )
        )

        # 1. Check disallowed command patterns
        for pattern in DISALLOWED_COMMAND_PATTERNS:
            if pattern.search(command):
                reason = f"Disallowed security pattern matched: {pattern.pattern}"
                self.audit_logger.record(
                    SecurityAuditEvent(
                        event_type=SecurityEventType.COMMAND_DENIED,
                        severity="HIGH",
                        source="sentinel",
                        message=f"Command denied: {reason}",
                        details={"session_id": session_id, "task_id": task_id, "pattern": pattern.pattern},
                    )
                )
                return CommandDecision(
                    allowed=False,
                    reason=reason,
                    redacted_command=redacted_cmd,
                    decision_code="DENIED_DISALLOWED_PATTERN",
                )

        # 2. Check for unauthorized network egress commands embedded in shell
        for net_pattern in NETWORK_COMMAND_PATTERNS:
            match = net_pattern.search(command)
            if match:
                target_url_or_host = match.group(1).lstrip("-").strip()
                # Extract host
                host_candidate = target_url_or_host.split("/")[0].split(":")[0]
                if host_candidate and not host_candidate.startswith("-"):
                    allowed, reason, rule = self.allowlist.check_egress(
                        destination_host=host_candidate,
                        task_id=task_id,
                    )
                    if not allowed:
                        self.audit_logger.record(
                            SecurityAuditEvent(
                                event_type=SecurityEventType.NETWORK_DENIED,
                                severity="HIGH",
                                source="sentinel",
                                message=f"Network command denied for target {host_candidate}: {reason}",
                                details={"target": host_candidate, "task_id": task_id},
                            )
                        )
                        return CommandDecision(
                            allowed=False,
                            reason=f"Command denied by network egress policy: {reason}",
                            redacted_command=redacted_cmd,
                            decision_code="DENIED_NETWORK_VIOLATION",
                        )

        # Allowed
        self.audit_logger.record(
            SecurityAuditEvent(
                event_type=SecurityEventType.COMMAND_ALLOWED,
                severity="LOW",
                source="sentinel",
                message=f"Command allowed: {redacted_cmd[:100]}",
                details={"session_id": session_id, "task_id": task_id},
            )
        )
        return CommandDecision(
            allowed=True,
            reason="Command authorized by Sentinel policy",
            redacted_command=redacted_cmd,
            decision_code="ALLOWED",
        )

    def evaluate_egress(
        self,
        destination_host: str,
        port: int = 443,
        scheme: str = "https",
        method: str = "GET",
        path: str = "",
        task_id: str = "default_task",
    ) -> EgressDecision:
        """Validate network egress and resolve host-side authentication injection."""
        self.audit_logger.record(
            SecurityAuditEvent(
                event_type=SecurityEventType.NETWORK_REQUESTED,
                severity="LOW",
                source="sentinel",
                message=f"Network egress requested to {scheme}://{destination_host}:{port}",
                details={"destination_host": destination_host, "port": port, "method": method, "task_id": task_id},
            )
        )

        allowed, reason, rule = self.allowlist.check_egress(
            destination_host=destination_host,
            port=port,
            scheme=scheme,
            method=method,
            path=path,
            task_id=task_id,
        )

        if not allowed:
            self.audit_logger.record(
                SecurityAuditEvent(
                    event_type=SecurityEventType.NETWORK_DENIED,
                    severity="HIGH",
                    source="sentinel",
                    message=f"Network egress denied: {reason}",
                    details={"destination_host": destination_host, "port": port, "task_id": task_id},
                )
            )
            return EgressDecision(
                allowed=False,
                reason=reason,
                auth_profile="none",
                matched_rule_id=rule.id if rule else None,
            )

        # Build host-injected authentication headers based on profile
        auth_profile = rule.auth_profile if rule else "none"
        injected_headers: dict[str, str] = {}

        if auth_profile == "bearer_host_secret_ref":
            secret = self.host_vault.get(destination_host) or self.host_vault.get("default_bearer")
            if secret:
                injected_headers["Authorization"] = f"Bearer {secret}"
        elif auth_profile == "api_key_header_host_secret_ref":
            secret = self.host_vault.get(destination_host) or self.host_vault.get("default_api_key")
            if secret:
                injected_headers["X-API-Key"] = secret
        elif auth_profile == "mock_internal_api":
            injected_headers["X-Tribune-Internal-Auth"] = "sentinel_verified"

        self.audit_logger.record(
            SecurityAuditEvent(
                event_type=SecurityEventType.NETWORK_ALLOWED,
                severity="LOW",
                source="sentinel",
                message=f"Network egress allowed to {destination_host} (rule: {rule.id if rule else 'default'})",
                details={"destination_host": destination_host, "auth_profile": auth_profile, "task_id": task_id},
            )
        )

        return EgressDecision(
            allowed=True,
            reason=reason,
            auth_profile=auth_profile,
            matched_rule_id=rule.id if rule else None,
            injected_headers=injected_headers,
        )


class SentinelServer:
    """Unix Domain Socket server for the Sentinel security broker."""

    def __init__(
        self,
        broker: SentinelBroker | None = None,
        socket_path: str = "/tmp/tribune_sentinel.sock",
    ) -> None:
        self.broker = broker or SentinelBroker()
        self.socket_path = socket_path
        self._running = False
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start Sentinel socket listener in background thread."""
        if self._running:
            return

        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        self._server_sock.listen(5)
        self._server_sock.settimeout(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="SentinelUDS")
        self._thread.start()
        logger.info(f"[SentinelServer] Listening on {self.socket_path}")

    def _serve_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            client_thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            client_thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            try:
                data = conn.recv(65536)
                if not data:
                    return
                req = json.loads(data.decode("utf-8"))
                action = req.get("action")

                if action == "ping":
                    res = {"status": "pong", "time": time.time()}
                elif action == "check_command":
                    dec = self.broker.evaluate_command(
                        command=req.get("command", ""),
                        session_id=req.get("session_id", "default"),
                        task_id=req.get("task_id", "default"),
                    )
                    res = {
                        "allowed": dec.allowed,
                        "reason": dec.reason,
                        "decision_code": dec.decision_code,
                        "redacted_command": dec.redacted_command,
                    }
                elif action == "check_egress":
                    dec = self.broker.evaluate_egress(
                        destination_host=req.get("destination_host", ""),
                        port=int(req.get("port", 443)),
                        scheme=req.get("scheme", "https"),
                        method=req.get("method", "GET"),
                        path=req.get("path", ""),
                        task_id=req.get("task_id", "default"),
                    )
                    res = {
                        "allowed": dec.allowed,
                        "reason": dec.reason,
                        "auth_profile": dec.auth_profile,
                        "matched_rule_id": dec.matched_rule_id,
                        # Note: We do NOT expose injected_headers to the sandbox socket client!
                    }
                else:
                    res = {"allowed": False, "reason": f"Unknown action '{action}'"}

                conn.sendall(json.dumps(res).encode("utf-8"))
            except Exception as exc:
                err_res = {"allowed": False, "reason": f"Sentinel error: {str(exc)}"}
                try:
                    conn.sendall(json.dumps(err_res).encode("utf-8"))
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop Sentinel socket listener and remove socket file."""
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        logger.info(f"[SentinelServer] Stopped {self.socket_path}")


class SentinelClient:
    """Client for querying Sentinel via Unix Domain Socket or in-process fallback."""

    def __init__(
        self,
        socket_path: str = "/tmp/tribune_sentinel.sock",
        in_process_broker: SentinelBroker | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.in_process_broker = in_process_broker

    def check_command(
        self,
        command: str,
        session_id: str = "default",
        task_id: str = "default",
    ) -> CommandDecision:
        """Query Sentinel to check if command is allowed."""
        # Fast path: in-process broker
        if self.in_process_broker is not None:
            return self.in_process_broker.evaluate_command(command, session_id, task_id)

        # IPC path: Unix domain socket
        if os.path.exists(self.socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client_sock:
                    client_sock.settimeout(3.0)
                    client_sock.connect(self.socket_path)
                    req = {
                        "action": "check_command",
                        "command": command,
                        "session_id": session_id,
                        "task_id": task_id,
                    }
                    client_sock.sendall(json.dumps(req).encode("utf-8"))
                    data = client_sock.recv(65536)
                    resp = json.loads(data.decode("utf-8"))
                    return CommandDecision(
                        allowed=bool(resp.get("allowed", False)),
                        reason=str(resp.get("reason", "")),
                        redacted_command=str(resp.get("redacted_command", command)),
                        decision_code=str(resp.get("decision_code", "DENIED")),
                    )
            except Exception as exc:
                logger.warning(f"[SentinelClient] UDS communication failed: {exc}")

        # Fallback safe: fail closed or local evaluation
        return CommandDecision(
            allowed=False,
            reason="Sentinel daemon unavailable (fail-closed)",
            redacted_command=command,
            decision_code="DENIED_SENTINEL_UNAVAILABLE",
        )
