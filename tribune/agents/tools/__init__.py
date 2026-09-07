"""Programmatic & MCP agent tools package."""

from .context_ops import (
    analyzeText,
    checkBudget,
    compressContext,
    foldHistory,
    get_context_manager,
    set_context_manager,
)
from .escalation import (
    ESCALATE_DEFECT_TOOL_SCHEMA,
    DefectType,
    VALID_DEFECT_TYPES,
    escalate_defect,
)

__all__ = [
    "analyzeText",
    "checkBudget",
    "foldHistory",
    "compressContext",
    "get_context_manager",
    "set_context_manager",
    "escalate_defect",
    "ESCALATE_DEFECT_TOOL_SCHEMA",
    "DefectType",
    "VALID_DEFECT_TYPES",
]
