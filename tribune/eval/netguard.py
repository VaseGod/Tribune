"""In-process network egress guard: deny-by-default with an explicit allowlist.

Reproducible, stateful agent evals must not silently reach the network — and for
a benefits tool they must *never* touch a real government system. This guard
patches the socket layer so that during an eval run any outbound TCP connection
to a host/port not on the allowlist raises :class:`NetworkEgressBlocked`.

Two layers protect an eval:

* the container runs with egress denied (``--network=none`` when no remote model
  is configured; see ``sandbox/``), and
* this in-process guard, which also works outside a container (in CI, on a
  laptop) and produces a precise, attributable error naming the blocked host.

The allowlist is intentionally tiny: the remote model endpoint only, if one is
configured. Everything else — including any agency endpoint — is denied. Local
IPC (AF_UNIX) is not egress and is left alone.
"""

from __future__ import annotations

import json
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


class NetworkEgressBlocked(RuntimeError):
    pass


class PromptInjectionDetected(ValueError):
    """Raised when an incoming unstructured data stream contains malicious prompt injections."""
    pass


# --------------------------------------------------------------------------- #
# Real-time Prompt Injection Detection & Ingestion Safeguards
# --------------------------------------------------------------------------- #

_INJECTION_PATTERNS = [
    re.compile(r"(?i)\b(?:ignore|disregard|override|forget)\s+(?:all\s+)?(?:previous|prior|system|above)\s+(?:instructions|prompts|rules|commands|directives|context)\b"),
    re.compile(r"(?i)\b(?:system\s+override|admin\s+override|jailbreak|DAN\s+mode|developer\s+mode\s+enabled)\b"),
    re.compile(r"(?i)\b(?:you\s+are\s+now\s+(?:an?\s+)?unrestricted|act\s+as\s+(?:an?\s+)?unfiltered)\b"),
    re.compile(r"(?i)(?:<system>|\[SYSTEM\]|---END SYSTEM---|```system)"),
    re.compile(r"(?i)\b(?:exfiltrate|leak\s+all\s+(?:ssn|pii|data)|disregard\s+(?:the\s+)?eligibility\s+rules)\b"),
    re.compile(r"(?i)\b(?:mark\s+this\s+household\s+as\s+likely_eligible\s+for\s+every\s+program)\b"),
]

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
_CC_PATTERN = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """Inspect text for prompt injection, jailbreak attempts, or system instruction overrides."""
    if not isinstance(text, str) or not text:
        return False, []
    findings = []
    for pat in _INJECTION_PATTERNS:
        matches = pat.findall(text)
        if matches:
            findings.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])
    return len(findings) > 0, findings


def filter_prompt_injection(text: str, raise_on_detection: bool = False) -> str:
    """Filter out or neutralize detected prompt injections in incoming data streams."""
    if not isinstance(text, str) or not text:
        return text
    is_injected, findings = detect_prompt_injection(text)
    if is_injected and raise_on_detection:
        raise PromptInjectionDetected(f"Prompt injection detected in input stream: {findings}")
    cleaned = text
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[FILTERED_INJECTION_DIRECTIVE]", cleaned)
    return cleaned


def redact_ssn(text: str) -> str:
    """Scrub Social Security Numbers (formatted and unformatted) from claimant text."""
    if not isinstance(text, str):
        return text
    # Formatted XXX-XX-XXXX
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    # SSN followed by 9 digits
    text = re.sub(r"(?i)\b(ssn[:\s]+)(\d{9})\b", r"\1[REDACTED_SSN]", text)
    return text


def redact_pii(text: str) -> str:
    """Scrub emails, phone numbers, credit card numbers, and SSNs from claimant files."""
    if not isinstance(text, str):
        return text
    text = redact_ssn(text)
    text = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    text = _CC_PATTERN.sub("[REDACTED_FINANCIAL]", text)
    return text


def scrub_claimant_payload(payload: Any) -> Any:
    """Recursively scrub PII, SSN, and filter prompt injections from raw claimant structures."""
    if isinstance(payload, str):
        cleaned = redact_pii(payload)
        return filter_prompt_injection(cleaned)
    if isinstance(payload, dict):
        return {k: scrub_claimant_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [scrub_claimant_payload(item) for item in payload]
    return payload


def inspect_inbound_stream(data: Any, raise_on_injection: bool = False) -> Any:
    """Ingestion safeguard: inspects inbound unstructured stream, sanitizing PII and blocking injection."""
    if isinstance(data, str):
        is_inj, findings = detect_prompt_injection(data)
        if is_inj and raise_on_injection:
            raise PromptInjectionDetected(f"Prompt injection detected: {findings}")
    return scrub_claimant_payload(data)


# --------------------------------------------------------------------------- #
# Egress Guard
# --------------------------------------------------------------------------- #

@dataclass
class _GuardState:
    allowlist: set[tuple[str, int]]
    allow_hosts: set[str]
    blocked: list[str] = field(default_factory=list)

    def permitted(self, host: str, port: int) -> bool:
        return host in self.allow_hosts or (host, port) in self.allowlist


_active: _GuardState | None = None
_orig_connect = None
_orig_connect_ex = None
_orig_create_connection = None


def _check(address) -> None:
    if _active is None:
        return
    if not isinstance(address, tuple) or len(address) < 2:
        return  # AF_UNIX or unusual family: not IP egress, leave it alone
    host, port = str(address[0]), int(address[1])
    if not _active.permitted(host, port):
        _active.blocked.append(f"{host}:{port}")
        raise NetworkEgressBlocked(
            f"network egress to {host}:{port} is denied by the eval sandbox "
            f"(allowlist: {sorted(_active.allowlist) or 'empty — fully offline'})"
        )


def inspect_outbound_payload(payload: dict | str) -> None:
    """Inspect outbound API request payloads and flag/block unauthorized transmission of encrypted reasoning blocks."""
    if _active is None:
        return
    data_str = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    forbidden_keys = ["encrypted_content", "thought_signature", "raw_base64_blobs"]
    for key in forbidden_keys:
        if key in data_str:
            _active.blocked.append(f"payload_violation:{key}")
            raise NetworkEgressBlocked(
                f"Unauthorized transmission of encrypted reasoning blocks ({key}) detected in outbound API request payload."
            )


def install(allowlist: set[tuple[str, int]] | None = None, allow_hosts: set[str] | None = None) -> None:
    global _active, _orig_connect, _orig_connect_ex, _orig_create_connection
    if _active is not None:
        raise RuntimeError("netguard already installed")
    _active = _GuardState(allowlist=allowlist or set(), allow_hosts=allow_hosts or set())
    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex
    _orig_create_connection = socket.create_connection

    def guarded_connect(self, address):
        _check(address)
        return _orig_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(address)
        return _orig_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        _check(address)
        return _orig_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection


def uninstall() -> list[str]:
    """Restore the socket layer; return the list of blocked destinations seen."""
    global _active, _orig_connect, _orig_connect_ex, _orig_create_connection
    blocked = list(_active.blocked) if _active else []
    if _orig_connect is not None:
        socket.socket.connect = _orig_connect
    if _orig_connect_ex is not None:
        socket.socket.connect_ex = _orig_connect_ex
    if _orig_create_connection is not None:
        socket.create_connection = _orig_create_connection
    _active = None
    _orig_connect = _orig_connect_ex = _orig_create_connection = None
    return blocked


def blocked_so_far() -> list[str]:
    return list(_active.blocked) if _active else []


def _hostport(url: str) -> tuple[str, int] | None:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"//{url}")
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (parsed.hostname, port)


def allowlist_from_settings(settings) -> set[tuple[str, int]]:
    """Only the configured remote model endpoint is allowlisted (if any).

    A hosted vector store or OCR endpoint would be added here too when
    configured, but the default appeals eval runs fully offline, so the
    allowlist is empty and egress is total-deny.
    """
    allow: set[tuple[str, int]] = set()
    if getattr(settings, "provider", "") == "openai_compat" or getattr(
        settings, "verifier_provider", ""
    ) == "openai_compat":
        hp = _hostport(getattr(settings, "openai_base_url", ""))
        if hp:
            allow.add(hp)
    if getattr(settings, "rule_store", "") == "hosted":
        hp = _hostport(getattr(settings, "hosted_vector_url", ""))
        if hp:
            allow.add(hp)
    if getattr(settings, "doc_ingest", "") == "ocr":
        hp = _hostport(getattr(settings, "ocr_endpoint", ""))
        if hp:
            allow.add(hp)
    return allow


@contextmanager
def deny_egress(
    allowlist: set[tuple[str, int]] | None = None, allow_hosts: set[str] | None = None
) -> Iterator[_GuardState]:
    install(allowlist=allowlist, allow_hosts=allow_hosts)
    try:
        assert _active is not None
        yield _active
    finally:
        uninstall()
