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

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse


class NetworkEgressBlocked(RuntimeError):
    pass


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
