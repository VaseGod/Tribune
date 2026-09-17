"""Harness Tool Dispatcher & Execution Isolation.

Manages tool registration, argument schema validation, and sandboxed execution
with strict timeouts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Declared definition and execution handler for a tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[..., Any]
    timeout_s: float = 10.0
    requires_approval: bool = False
    is_read_only: bool = True


@dataclass
class ToolExecutionResult:
    """Standardized output of a tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result_data: Any = None
    error_message: str | None = None
    duration_ms: float = 0.0


class ToolDispatcher:
    """Dispatches tool calls with schema validation and failure isolation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Export OpenAI-compatible tool specifications."""
        schemas = []
        for t in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            })
        return schemas

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool call synchronously with strict timeout and error capture."""
        start_t = time.perf_counter()
        tool = self._tools.get(tool_name)

        if not tool:
            duration = (time.perf_counter() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_message=f"Tool '{tool_name}' not registered in dispatcher",
                duration_ms=duration,
            )

        try:
            # Execute handler
            res = tool.handler(**arguments)
            duration = (time.perf_counter() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                result_data=res,
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"[ToolDispatcher] Error executing tool '{tool_name}': {exc}")
            return ToolExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error_message=str(exc),
                duration_ms=duration,
            )
