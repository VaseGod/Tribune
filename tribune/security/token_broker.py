"""Surrogate Token Broker and Credential Redaction Engine.

Ensures real authentication material remains strictly on the host. The model-facing
runtime only ever sees synthetic surrogate tokens (e.g. MOCK_ACCESS_HANDLE_ALPHA,
MOCK_API_KEY_BRAVO, MOCK_STORAGE_TOKEN_CHARLIE), and stdout/stderr/prompts/logs
are automatically scanned and redacted.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

# Standard surrogate names for deterministic mapping
SURROGATE_NAMES = [
    "MOCK_ACCESS_HANDLE_ALPHA",
    "MOCK_API_KEY_BRAVO",
    "MOCK_STORAGE_TOKEN_CHARLIE",
    "MOCK_AUTH_SECRET_DELTA",
    "MOCK_CLIENT_TOKEN_ECHO",
    "MOCK_SIGNING_KEY_FOXTROT",
    "MOCK_DATABASE_PASS_GOLF",
    "MOCK_SERVICE_CRED_HOTEL",
]

# Regex patterns detecting common credential/secret formats
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),  # OpenAI / generic secret key
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),  # Anthropic key
    re.compile(r"AIzaSy[a-zA-Z0-9_-]{33}"),  # Google API key
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub token
    re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]{20,})", re.IGNORECASE),  # Bearer token
    re.compile(r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?", re.IGNORECASE),
    re.compile(r"(?:postgres|mysql|mongodb(?:\+srv)?):\/\/[^:\s]+:([^@\s]+)@", re.IGNORECASE),  # Connection strings
    re.compile(r"https?:\/\/[^:\s]+:([^@\s]+)@", re.IGNORECASE),  # URL with basic auth
    re.compile(r"\"private_key\"\s*:\s*\"-----BEGIN[^\"]+\"", re.IGNORECASE),  # Service account JSON
    re.compile(r"DefaultEndpointsProtocol=https?;AccountName=[^;]+;AccountKey=([a-zA-Z0-9+/=]{20,})", re.IGNORECASE),  # Azure storage
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
]

# Sensitive environment variable keys to surrogate unconditionally
SENSITIVE_ENV_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "GITHUB_TOKEN",
    "SLACK_BOT_TOKEN",
    "TRIBUNE_HMAC_SECRET",
    "TRIBUNE_REAL_API_KEY",
    "SECRET_KEY",
    "DATABASE_URL",
    "DB_PASSWORD",
    "API_TOKEN",
}


@dataclass
class SecretBinding:
    """Host-only binding between a surrogate token and a real secret."""

    surrogate_id: str
    surrogate_name: str
    real_secret: str
    secret_ref: str
    injection_policy: str = "bearer_host_secret_ref"
    session_id: str = "default"
    created_at: float = field(default_factory=lambda: 0.0)


class TokenBroker:
    """Manages synthetic surrogate tokens, secret isolation, and bidirectional redaction.

    Maintains a host-only mapping between surrogate tokens and authentic secrets.
    Ensures model contexts, logs, containers, and reports never observe raw secrets.
    """

    def __init__(self, surrogate_prefix: str = "MOCK_") -> None:
        self.surrogate_prefix = surrogate_prefix
        self._surrogate_to_real: dict[str, SecretBinding] = {}
        self._real_to_surrogate: dict[str, str] = {}
        self._session_counters: dict[str, int] = {}
        self._lock = threading.RLock()

    def register_secret(
        self,
        real_secret: str,
        secret_ref: str = "default_secret",
        injection_policy: str = "bearer_host_secret_ref",
        session_id: str = "default",
        preferred_surrogate_name: str | None = None,
    ) -> str:
        """Register a real secret and return its corresponding synthetic surrogate token."""
        if not real_secret or len(real_secret.strip()) < 4:
            return real_secret

        with self._lock:
            # Check if this exact secret is already registered in this session
            cache_key = f"{session_id}::{real_secret}"
            if cache_key in self._real_to_surrogate:
                return self._real_to_surrogate[cache_key]

            idx = self._session_counters.get(session_id, 0)
            self._session_counters[session_id] = idx + 1

            if preferred_surrogate_name:
                surrogate_name = preferred_surrogate_name
            elif idx < len(SURROGATE_NAMES):
                surrogate_name = SURROGATE_NAMES[idx]
            else:
                h = hashlib.sha256(f"{session_id}:{idx}".encode()).hexdigest()[:8].upper()
                surrogate_name = f"{self.surrogate_prefix}SURROGATE_TOKEN_{h}"

            binding = SecretBinding(
                surrogate_id=f"surr_{idx}_{session_id}",
                surrogate_name=surrogate_name,
                real_secret=real_secret,
                secret_ref=secret_ref,
                injection_policy=injection_policy,
                session_id=session_id,
            )

            self._surrogate_to_real[surrogate_name] = binding
            self._real_to_surrogate[cache_key] = surrogate_name
            # Also index by real_secret globally for fast redaction
            self._real_to_surrogate[real_secret] = surrogate_name
            return surrogate_name

    def resolve_surrogate(self, surrogate_name: str) -> SecretBinding | None:
        """Resolve a surrogate token to its real secret on the host. NEVER expose to sandbox."""
        with self._lock:
            return self._surrogate_to_real.get(surrogate_name)

    def surrogate_environment(
        self,
        env: Mapping[str, str],
        session_id: str = "default",
    ) -> tuple[dict[str, str], list[str]]:
        """Sanitize an environment dictionary by replacing sensitive values with surrogate tokens.

        Returns (sanitized_env, surrogate_keys_modified).
        """
        sanitized = dict(env)
        modified_keys: list[str] = []

        with self._lock:
            for key, val in list(sanitized.items()):
                upper_key = key.upper()
                is_sensitive = (
                    upper_key in SENSITIVE_ENV_KEYS
                    or any(s in upper_key for s in ("SECRET", "TOKEN", "API_KEY", "PASSWORD", "CREDENTIAL"))
                )

                # Check if value matches known secret regex patterns
                pattern_matched = any(p.search(val) for p in SECRET_PATTERNS if len(val) >= 8)

                if is_sensitive or pattern_matched:
                    surrogate = self.register_secret(
                        real_secret=val,
                        secret_ref=key,
                        session_id=session_id,
                    )
                    sanitized[key] = surrogate
                    modified_keys.append(key)

        return sanitized, modified_keys

    def redact_text(self, text: str) -> str:
        """Scrub all known real secrets and regex-matched credential patterns from text.

        Used to sanitize stdout, stderr, exception messages, and prompt injections.
        """
        if not text:
            return text

        scrubbed = text

        with self._lock:
            # 1. Exact match replacements for all registered real secrets
            for real_secret, surrogate in self._real_to_surrogate.items():
                if "::" in real_secret:
                    _, real = real_secret.split("::", 1)
                else:
                    real = real_secret
                if len(real) >= 4 and real in scrubbed:
                    scrubbed = scrubbed.replace(real, surrogate)

            # 2. Pattern-based redaction for unregistered credentials
            for pattern in SECRET_PATTERNS:
                def _repl(match: re.Match) -> str:
                    matched_str = match.group(0)
                    # If match has capturing group, replace the captured secret
                    if match.groups():
                        secret_part = match.group(1)
                        if len(secret_part) >= 6:
                            return matched_str.replace(secret_part, "[REDACTED_SECRET]")
                    return "[REDACTED_SECRET]"

                scrubbed = pattern.sub(_repl, scrubbed)

        return scrubbed

    def mask_command(self, command: str | list[str], session_id: str = "default") -> tuple[str | list[str], str]:
        """Scan command string or list for credentials, replace with surrogates.

        Returns (sanitized_command, redacted_display_string).
        """
        if isinstance(command, list):
            cmd_str = " ".join(command)
            redacted_str = self.redact_text(cmd_str)
            sanitized_list = [self.redact_text(arg) for arg in command]
            return sanitized_list, redacted_str
        else:
            redacted_str = self.redact_text(command)
            return redacted_str, redacted_str

    def reset_session(self, session_id: str) -> None:
        """Reset state for a given session."""
        with self._lock:
            self._session_counters.pop(session_id, None)
            keys_to_del = [k for k, v in self._surrogate_to_real.items() if v.session_id == session_id]
            for k in keys_to_del:
                del self._surrogate_to_real[k]


_GLOBAL_TOKEN_BROKER: TokenBroker | None = None
_BROKER_LOCK = threading.RLock()


def get_token_broker() -> TokenBroker:
    """Return the global default TokenBroker singleton."""
    global _GLOBAL_TOKEN_BROKER
    with _BROKER_LOCK:
        if _GLOBAL_TOKEN_BROKER is None:
            _GLOBAL_TOKEN_BROKER = TokenBroker()
        return _GLOBAL_TOKEN_BROKER
