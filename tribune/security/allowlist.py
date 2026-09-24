"""Sentinel Allowlist Policy and Egress Verification Engine.

Enforces deny-by-default egress controls, destination host and port validation,
HTTP method restrictions, and per-task request limits.
"""

from __future__ import annotations

import fnmatch
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class AllowlistRule:
    """A granular network egress authorization rule."""

    id: str
    destination_host: str
    port: int = 443
    scheme: str = "https"
    methods: tuple[str, ...] = ("GET",)
    path_prefix: str = ""
    auth_profile: str = "none"
    max_requests_per_task: int = 50


class AllowlistPolicy:
    """Manages and enforces network egress allowlists."""

    def __init__(
        self,
        rules: list[AllowlistRule] | None = None,
        default_policy: str = "deny",
        version: int = 1,
    ) -> None:
        self.version = version
        self.default_policy = default_policy.lower()
        self.rules = list(rules or [])
        self._task_request_counts: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()

    @classmethod
    def load_from_yaml(cls, path: str) -> AllowlistPolicy:
        """Load allowlist policy from a YAML configuration file."""
        if not os.path.exists(path):
            # Fallback safe default deny
            return cls(default_policy="deny")

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        rules_raw = data.get("rules", [])
        rules: list[AllowlistRule] = []
        for r in rules_raw:
            rules.append(
                AllowlistRule(
                    id=str(r.get("id", f"rule_{len(rules)}")),
                    destination_host=str(r.get("destination_host", "")).lower(),
                    port=int(r.get("port", 443)),
                    scheme=str(r.get("scheme", "https")).lower(),
                    methods=tuple(m.upper() for m in r.get("methods", ["GET"])),
                    path_prefix=str(r.get("path_prefix", "")),
                    auth_profile=str(r.get("auth_profile", "none")),
                    max_requests_per_task=int(r.get("max_requests_per_task", 50)),
                )
            )

        return cls(
            rules=rules,
            default_policy=str(data.get("default_policy", "deny")),
            version=int(data.get("version", 1)),
        )

    def check_egress(
        self,
        destination_host: str,
        port: int = 443,
        scheme: str = "https",
        method: str = "GET",
        path: str = "",
        task_id: str = "default",
    ) -> tuple[bool, str, AllowlistRule | None]:
        """Validate an egress network request against allowlist rules.

        Returns (is_allowed, reason, matched_rule).
        """
        host = destination_host.lower().strip()
        scheme_norm = scheme.lower().strip()
        method_norm = method.upper().strip()

        with self._lock:
            for rule in self.rules:
                # Match host (supports wildcards, e.g. *.internal.tribune)
                if not (rule.destination_host == host or fnmatch.fnmatch(host, rule.destination_host)):
                    continue

                # Match port
                if rule.port != port:
                    continue

                # Match scheme
                if rule.scheme != scheme_norm:
                    continue

                # Match HTTP method
                if method_norm not in rule.methods:
                    return False, f"Method {method_norm} not allowed by rule '{rule.id}' (allowed: {rule.methods})", rule

                # Match path prefix if defined
                if rule.path_prefix and not path.startswith(rule.path_prefix):
                    continue

                # Check task-level rate limit / max requests
                task_key = (task_id, rule.id)
                current_count = self._task_request_counts.get(task_key, 0)
                if current_count >= rule.max_requests_per_task:
                    return (
                        False,
                        f"Task '{task_id}' exceeded max_requests_per_task ({rule.max_requests_per_task}) for rule '{rule.id}'",
                        rule,
                    )

                # Request is allowed; increment counter
                self._task_request_counts[task_key] = current_count + 1
                return True, f"Allowed by rule '{rule.id}'", rule

        # No rule matched
        if self.default_policy == "allow":
            return True, "Allowed by default policy", None

        return False, f"Destination {scheme_norm}://{host}:{port} denied by default policy", None

    def reset_task_counters(self, task_id: str) -> None:
        """Reset request counters for a specific task."""
        with self._lock:
            keys_to_del = [k for k in self._task_request_counts if k[0] == task_id]
            for k in keys_to_del:
                del self._task_request_counts[k]


def default_allowlist() -> AllowlistPolicy:
    """Return default secure allowlist policy with internal mock endpoints allowed."""
    rules = [
        AllowlistRule(
            id="allow_local_mock_api",
            destination_host="mock.internal.tribune",
            port=443,
            scheme="https",
            methods=("GET", "POST"),
            auth_profile="mock_internal_api",
            max_requests_per_task=50,
        ),
        AllowlistRule(
            id="allow_static_dataset",
            destination_host="datasets.internal.tribune",
            port=443,
            scheme="https",
            methods=("GET",),
            auth_profile="none",
            max_requests_per_task=20,
        ),
        AllowlistRule(
            id="allow_localhost_mock",
            destination_host="127.0.0.1",
            port=8080,
            scheme="http",
            methods=("GET", "POST"),
            auth_profile="none",
            max_requests_per_task=100,
        ),
    ]
    return AllowlistPolicy(rules=rules, default_policy="deny")
