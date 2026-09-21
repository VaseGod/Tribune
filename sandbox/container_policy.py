"""Hardened Container Security Policy & Command Sanitization Engine.

Defines strict resource limits, path confinement, and command denial patterns
for sandboxed execution.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_DENY_PATTERNS = [
    r"\bsudo\b",
    r"\bsu\s+",
    r"\bchmod\s+.*(?:[+a-zA-Z]*s|[0-7]*[4-7][0-7]{2})",  # setuid / suid bit attempts
    r"\bchown\b",
    r"/etc/shadow",
    r"/etc/passwd",
    r"/etc/sudoers",
    r"\.dockerenv\b",
    r"/var/run/docker\.sock",
    r"\brm\s+-rf\s+/(?:\s|$|\*)",  # rm -rf /
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # fork bomb
    r"\bcurl\b.*\|\s*bash\b",
    r"\bwget\b.*\|\s*bash\b",
    r"\bprintenv\b",
    r"\bexport\s+-p\b",
]


@dataclass
class SandboxPolicy:
    """Explicit security policy enforcing container isolation constraints."""

    allow_shell: bool = True
    allow_python: bool = True
    allow_network: bool = False
    allowed_egress_hosts: list[str] = field(default_factory=list)
    max_execution_seconds: int = 120
    max_memory_mb: int = 1024
    max_output_bytes: int = 1048576  # 1 MB
    read_only_paths: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_DENY_PATTERNS))

    def validate_command(self, command_str: str) -> tuple[bool, str | None]:
        """Verify command does not violate denial patterns or shell prohibitions."""
        if not self.allow_shell:
            return False, "Execution denied: shell access is disabled by policy."

        for pattern in self.deny_patterns:
            if re.search(pattern, command_str, re.IGNORECASE):
                logger.warning(f"[SandboxPolicy] Denied command matching pattern '{pattern}': {command_str}")
                return False, f"Execution denied: command matched security deny pattern '{pattern}'"

        return True, None

    def validate_path_access(self, target_path: str, for_writing: bool = False) -> tuple[bool, str | None]:
        """Validate filesystem path confinement against read-only and writable paths."""
        abs_target = os.path.abspath(target_path)

        if for_writing:
            if not self.writable_paths:
                return True, None
            is_writable = any(
                abs_target == os.path.abspath(p) or abs_target.startswith(os.path.abspath(p) + os.sep)
                for p in self.writable_paths
            )
            if not is_writable:
                return False, f"Path '{target_path}' is not within permitted writable paths."
        else:
            # Check read-only / disallowed paths
            sensitive = ["/etc", "/root", "/proc/sys", "/sys"]
            for s in sensitive:
                if abs_target == s or abs_target.startswith(s + os.sep):
                    return False, f"Path '{target_path}' is a protected system directory."

        return True, None


__all__ = [
    "DEFAULT_DENY_PATTERNS",
    "SandboxPolicy",
]
