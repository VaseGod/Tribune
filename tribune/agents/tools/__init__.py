"""Programmatic & MCP agent tools package."""

from .context_ops import (
    analyzeText,
    checkBudget,
    compressContext,
    foldHistory,
    get_context_manager,
    set_context_manager,
)

__all__ = [
    "analyzeText",
    "checkBudget",
    "foldHistory",
    "compressContext",
    "get_context_manager",
    "set_context_manager",
]
