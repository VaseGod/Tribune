"""Pluggable token counting: real tokenizers when available, safe fallback otherwise.

`estimate_tokens` keeps the repo-standard ~4-chars-per-token heuristic as the
default (consistent with `ContextPilot`), but `TRIBUNE_TOKEN_COUNTER=tiktoken`
routes through tiktoken when installed. Production deployments that need
billing-grade prefill accounting set the env var and install the package;
everything else is untouched.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharEstimator:
    """Heuristic fallback: ~4 chars per token, min 1 for non-empty text."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class TiktokenCounter:
    """Real tokenizer counts via tiktoken (optional dependency)."""

    def __init__(self, model: str = "cl100k_base") -> None:
        try:
            import tiktoken  # type: ignore
        except ImportError as err:
            raise ImportError(
                "TRIBUNE_TOKEN_COUNTER=tiktoken requires the 'tiktoken' package. "
                "Install it or unset the variable to use the char estimator."
            ) from err
        self._enc = tiktoken.get_encoding(model)
        self.model = model

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


_COUNTER_CACHE: dict[tuple[str, str], TokenCounter] = {}


def get_token_counter() -> TokenCounter:
    backend = os.getenv("TRIBUNE_TOKEN_COUNTER", "char").strip().lower()
    model = os.getenv("TRIBUNE_TOKEN_COUNTER_MODEL", "cl100k_base")
    key = (backend, model)
    cached = _COUNTER_CACHE.get(key)
    if cached is not None:
        return cached
    if backend == "tiktoken":
        try:
            counter: TokenCounter = TiktokenCounter(model)
            _COUNTER_CACHE[key] = counter
            return counter
        except ImportError:
            logger.warning(
                "[TOKENS] tiktoken requested but not installed; falling back to char estimator."
            )
            fallback = CharEstimator()
            _COUNTER_CACHE[key] = fallback
            return fallback
    if backend != "char":
        logger.warning("[TOKENS] Unknown counter '%s'; using char estimator.", backend)
    counter = CharEstimator()
    _COUNTER_CACHE[key] = counter
    return counter


def estimate_tokens(text: str) -> int:
    """Count tokens with the configured backend (default: char heuristic)."""
    return get_token_counter().count(text)


__all__ = [
    "TokenCounter",
    "CharEstimator",
    "TiktokenCounter",
    "get_token_counter",
    "estimate_tokens",
]
