"""Base Agent Abstraction and Central Tool Registry.

Provides:
1. BaseAgent with execution freezing interlock upon defect escalation.
2. CentralToolRegistry managing tool definitions, schemas, and execution dispatches.
3. Out-of-the-box registration of escalate_defect.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable
from typing import Any

from .tools.escalation import ESCALATE_DEFECT_TOOL_SCHEMA, escalate_defect

logger = logging.getLogger(__name__)


class AgentFrozenError(RuntimeError):
    """Raised when an agent attempts tool execution after being frozen due to escalation."""
    pass


class CentralToolRegistry:
    """Thread-safe registry for agent tools and JSON schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Register default core tools
        self.register_tool(
            name="escalate_defect",
            fn=escalate_defect,
            schema=ESCALATE_DEFECT_TOOL_SCHEMA,
        )

    def register_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool function and optional JSON schema."""
        with self._lock:
            self._tools[name] = fn
            if schema:
                self._schemas[name] = copy.deepcopy(schema)
            else:
                self._schemas[name] = {
                    "name": name,
                    "description": fn.__doc__ or f"Tool {name}",
                    "input_schema": {"type": "object", "properties": {}},
                }

    def get_tool(self, name: str) -> Callable[..., Any] | None:
        with self._lock:
            return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return list of all tool schemas for model API formatting."""
        with self._lock:
            return list(self._schemas.values())

    def execute(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Execute tool by name."""
        with self._lock:
            tool_fn = self._tools.get(tool_name)
            if not tool_fn:
                raise KeyError(f"Tool '{tool_name}' not registered in CentralToolRegistry")
            return tool_fn(**kwargs)


_CENTRAL_TOOL_REGISTRY = CentralToolRegistry()


def get_central_tool_registry() -> CentralToolRegistry:
    return _CENTRAL_TOOL_REGISTRY


class BaseAgent:
    """Base agent implementing lifecycle state and execution freezing interlock."""

    def __init__(
        self,
        agent_id: str,
        role: str,
        tool_registry: CentralToolRegistry | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.tool_registry = tool_registry or get_central_tool_registry()
        self.is_frozen: bool = False
        self.frozen_reason: str = ""
        self.execution_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def freeze(self, reason: str = "") -> None:
        """Immediately freeze agent execution loop upon escalation or containment breach."""
        with self._lock:
            self.is_frozen = True
            self.frozen_reason = reason
            logger.warning(f"[AGENT-FREEZE] Agent '{self.agent_id}' execution frozen: {reason}")

    def unfreeze(self) -> None:
        with self._lock:
            self.is_frozen = False
            self.frozen_reason = ""

    def execute_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Execute a tool while enforcing the execution freezing interlock."""
        with self._lock:
            if self.is_frozen:
                raise AgentFrozenError(
                    f"Tool execution rejected: Agent '{self.agent_id}' is frozen ({self.frozen_reason}). "
                    f"Execution halted following defect escalation."
                )

            res = self.tool_registry.execute(tool_name, kwargs)

            # If escalate_defect was called, immediately freeze self
            if tool_name == "escalate_defect":
                self.freeze(reason=f"Defect escalated: {kwargs.get('defect_type')}")

            self.execution_history.append({
                "tool": tool_name,
                "kwargs": kwargs,
                "result": res,
            })
            return res


__all__ = [
    "AgentFrozenError",
    "CentralToolRegistry",
    "get_central_tool_registry",
    "BaseAgent",
]
