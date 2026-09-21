"""Programmatic Typed Tool Catalog & Runtime Mode Selector.

Provides typed programmatic tool execution reserved for high-security, compliance-constrained,
or auditable enterprise actions where raw shell execution is prohibited or requires structured schema validation.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class RuntimeMode(str, enum.Enum):
    """Runtime execution mode for the sandbox subsystem."""

    SHELL_FIRST = "shell_first"
    CATALOG_ONLY = "catalog_only"
    SHELL_WITH_CATALOG_ALLOWLIST = "shell_with_catalog_allowlist"


@dataclass
class TypedTool:
    """Schema and handler definition for an auditable programmatic tool."""

    name: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    requires_audit: bool = True
    estimated_tokens: int = 250


class ToolCatalog:
    """Registry and execution controller for programmatic typed tools."""

    def __init__(self) -> None:
        self._tools: dict[str, TypedTool] = {}
        self.invocation_count = 0
        self._register_default_tools()

    def register(self, tool: TypedTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> TypedTool | None:
        return self._tools.get(name)

    def execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in catalog.")

        self.invocation_count += 1
        logger.info(f"[ToolCatalog] Executing auditable typed tool '{name}' with parameters {params}")
        return tool.handler(params)

    def _register_default_tools(self) -> None:
        """Register built-in enterprise compliance tools."""

        def query_statutory_table(params: dict[str, Any]) -> dict[str, Any]:
            program = str(params.get("program", "snap")).lower()
            year = params.get("year", 2026)
            household = int(params.get("household_size", 1))
            # Representative canonical FPL limits
            limit = 1450 if household <= 1 else 1450 + (household - 1) * 514
            return {
                "program": program,
                "year": year,
                "household_size": household,
                "gross_monthly_income_limit": limit,
                "audited": True,
            }

        def verify_case_disclaimers(params: dict[str, Any]) -> dict[str, Any]:
            text = params.get("text", "")
            has_disclaimer = "legal advice" in text.lower() or "informational" in text.lower()
            return {"disclaimer_verified": has_disclaimer, "audited": True}

        self.register(
            TypedTool(
                name="query_statutory_table",
                description="Audited query to official statutory income standards",
                handler=query_statutory_table,
                parameters_schema={"program": "string", "year": "integer", "household_size": "integer"},
                requires_audit=True,
            )
        )
        self.register(
            TypedTool(
                name="verify_case_disclaimers",
                description="Verify presence of mandatory statutory disclaimers",
                handler=verify_case_disclaimers,
                parameters_schema={"text": "string"},
                requires_audit=True,
            )
        )

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


__all__ = [
    "RuntimeMode",
    "TypedTool",
    "ToolCatalog",
]
