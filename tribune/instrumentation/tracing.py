"""Tracing hooks.

When ``TRIBUNE_TRACING=weave`` and ``weave`` is installed, runs are traced to
Weights & Biases Weave. Otherwise every hook is a silent no-op so the system runs
with zero observability dependencies. Nothing here ever raises into the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ..config import get_settings

_initialized = False
_weave = None


def init_tracing() -> None:
    global _initialized, _weave
    if _initialized:
        return
    _initialized = True
    settings = get_settings()
    if settings.tracing != "weave":
        return
    try:  # pragma: no cover - exercised only when weave is installed/configured
        import weave  # type: ignore

        weave.init(settings.weave_project)
        _weave = weave
    except Exception:
        # Tracing is best-effort; never let it break a run.
        _weave = None


@contextmanager
def span(name: str, **attrs: object) -> Iterator[None]:
    """A tracing span that degrades to a no-op when weave is unavailable."""
    init_tracing()
    if _weave is None:
        yield
        return
    try:  # pragma: no cover - requires weave
        with _weave.attributes(attrs):
            yield
    except Exception:
        yield


def log(event: str, **fields: object) -> None:  # pragma: no cover - thin shim
    init_tracing()
    if _weave is None:
        return
    try:
        _weave.publish({"event": event, **fields})  # type: ignore[attr-defined]
    except Exception:
        pass
