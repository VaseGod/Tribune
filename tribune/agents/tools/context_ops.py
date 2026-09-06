"""MCP & Programmatic Context Operations Toolset.

Provides agent-executable and MCP-callable operations for proactive context engineering:
- analyzeText: Computes token count, Shannon entropy, and information density
- checkBudget: Returns quota, consumption velocity, and compaction urgency
- foldHistory: Discards resolved spans and embeds semantic indexing headers
- compressContext: Extractive/distillation compression preserving entities, citations, and causal predicates
"""

from __future__ import annotations

import threading

from ...context.manager import ProactiveContextManager
from ...types import (
    BudgetStatus,
    ContextAnalysis,
    FoldResult,
)

_LOCK = threading.RLock()
_ACTIVE_CONTEXT_MANAGER: ProactiveContextManager | None = None


def get_context_manager() -> ProactiveContextManager:
    """Get the active ProactiveContextManager instance, creating a default one if needed."""
    global _ACTIVE_CONTEXT_MANAGER
    with _LOCK:
        if _ACTIVE_CONTEXT_MANAGER is None:
            _ACTIVE_CONTEXT_MANAGER = ProactiveContextManager()
        return _ACTIVE_CONTEXT_MANAGER


def set_context_manager(manager: ProactiveContextManager) -> None:
    """Set the active ProactiveContextManager instance."""
    global _ACTIVE_CONTEXT_MANAGER
    with _LOCK:
        _ACTIVE_CONTEXT_MANAGER = manager


def analyzeText(span: str) -> ContextAnalysis:
    """Calculates exact token count, information density, and Shannon entropy across the working memory span."""
    manager = get_context_manager()
    return manager.analyze_text(span)


def checkBudget() -> BudgetStatus:
    """Returns remaining context window quota, consumption velocity, and compaction urgency flags."""
    manager = get_context_manager()
    return manager.check_budget()


def foldHistory(span_id: str, summarize: bool = True) -> FoldResult:
    """Discards resolved interaction spans and stores structured semantic indexing headers in place."""
    manager = get_context_manager()
    return manager.fold_history(span_id=span_id, summarize=summarize)


def compressContext(target_text: str, ratio: float = 0.5) -> str:
    """Integrates an extractive/distillation compression pipeline (modeled on LLMLingua-2 principles).

    Targets evidentiary discovery filings and deposition transcripts without dropping
    named entities, citations, or causal predicates.
    """
    manager = get_context_manager()
    return manager.compress_context(target_text=target_text, ratio=ratio)


__all__ = [
    "analyzeText",
    "checkBudget",
    "foldHistory",
    "compressContext",
    "get_context_manager",
    "set_context_manager",
]
