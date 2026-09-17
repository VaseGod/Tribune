"""Harness Runtime Policy Enforcement Hooks.

Validates operations at loop boundaries before dispatching inference or tool execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .state import HarnessState, StepType

logger = logging.getLogger(__name__)


@dataclass
class PolicyGateEvaluation:
    """Outcome of a harness policy check."""

    allowed: bool
    policy_name: str
    violation_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class HarnessPolicyEnforcer:
    """Enforces runtime guardrails before each harness step."""

    def __init__(
        self,
        forbidden_tools: set[str] | None = None,
        max_cost_usd: float = 1.00,
        max_tokens: int = 200_000,
        allow_network_tools: bool = False,
    ) -> None:
        self.forbidden_tools = forbidden_tools or {
            "execute_shell",
            "eval_code",
            "modify_security_rules",
            "delete_file_permanent",
            "raw_socket_bind",
        }
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.allow_network_tools = allow_network_tools

    def evaluate_step(self, state: HarnessState, next_step_type: StepType, details: dict[str, Any]) -> PolicyGateEvaluation:
        """Run all policy gates prior to dispatching next step."""

        # 1. Kill switch check
        if state.kill_requested:
            return PolicyGateEvaluation(
                allowed=False,
                policy_name="KILL_SWITCH_ACTIVE",
                violation_reason=state.failure_reason or "Emergency kill switch is active",
            )

        # 2. Cumulative budget checks
        if state.cumulative_cost_usd >= self.max_cost_usd:
            return PolicyGateEvaluation(
                allowed=False,
                policy_name="MAX_COST_CAP_EXCEEDED",
                violation_reason=f"Cumulative cost ${state.cumulative_cost_usd:.4f} >= limit ${self.max_cost_usd:.4f}",
            )

        if state.cumulative_tokens >= self.max_tokens:
            return PolicyGateEvaluation(
                allowed=False,
                policy_name="MAX_TOKEN_CAP_EXCEEDED",
                violation_reason=f"Cumulative tokens {state.cumulative_tokens} >= limit {self.max_tokens}",
            )

        # 3. Tool boundary checks
        if next_step_type == StepType.TOOL:
            tool_name = details.get("tool_name", "")
            if tool_name in self.forbidden_tools:
                return PolicyGateEvaluation(
                    allowed=False,
                    policy_name="FORBIDDEN_TOOL_INVOCATION",
                    violation_reason=f"Tool '{tool_name}' is prohibited by safety policy",
                    details={"forbidden_tool": tool_name},
                )

        return PolicyGateEvaluation(allowed=True, policy_name="ALL_POLICIES_PASSED")
