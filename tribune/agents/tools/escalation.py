"""Escalation Defect Tool Subsystem.

Provides the escalate_defect tool enabling agents to report unresolvable
environmental defects, syntax errors in fixtures, missing dependencies, or
ambiguous specifications without resorting to reward-hacking or test tampering.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from ...security.audit import SecurityEventType, record_security_event

logger = logging.getLogger(__name__)

DefectType = Literal[
    "broken_test_assertion",
    "environment_dependency_missing",
    "ambiguous_specification",
    "syntax_error_in_fixture",
]

VALID_DEFECT_TYPES: set[str] = {
    "broken_test_assertion",
    "environment_dependency_missing",
    "ambiguous_specification",
    "syntax_error_in_fixture",
}


def escalate_defect(
    defect_type: Literal[
        "broken_test_assertion",
        "environment_dependency_missing",
        "ambiguous_specification",
        "syntax_error_in_fixture"
    ],
    target_file: str,
    reproduction_trace: str,
    proposed_rationale: str,
) -> dict[str, Any]:
    """Halts execution loop and reports an unresolvable upstream or environmental defect."""
    if defect_type not in VALID_DEFECT_TYPES:
        raise ValueError(
            f"Invalid defect_type '{defect_type}'. Must be one of: {sorted(VALID_DEFECT_TYPES)}"
        )

    defect_id = f"esc_{defect_type}_{int(time.time() * 1000)}"
    logger.critical(
        f"[ESCALATE-DEFECT] Defect '{defect_type}' reported on '{target_file}': {proposed_rationale}"
    )

    report = {
        "defect_id": defect_id,
        "defect_type": defect_type,
        "target_file": target_file,
        "reproduction_trace": reproduction_trace,
        "proposed_rationale": proposed_rationale,
        "timestamp": time.time(),
        "status": "ESCALATED",
        "execution_halted": True,
    }

    # Dispatch immediate structured event to security audit
    record_security_event(
        event_type=SecurityEventType.DEFECT_ESCALATED,
        source="tribune.agents.tools.escalate_defect",
        message=f"Defect escalated: {defect_type} in {target_file}",
        severity="HIGH",
        details=report,
    )

    return report


ESCALATE_DEFECT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "escalate_defect",
    "description": (
        "Halts execution loop and reports an unresolvable upstream or environmental defect "
        "(e.g., broken test assertion, missing environment dependency, ambiguous specification, "
        "or syntax error in fixture). Use immediately instead of tampering with tests."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "defect_type": {
                "type": "string",
                "enum": [
                    "broken_test_assertion",
                    "environment_dependency_missing",
                    "ambiguous_specification",
                    "syntax_error_in_fixture",
                ],
                "description": "Category of unresolvable environmental or fixture defect.",
            },
            "target_file": {
                "type": "string",
                "description": "The file path where the defect or broken assertion resides.",
            },
            "reproduction_trace": {
                "type": "string",
                "description": "Command line, stack trace, or reproduction steps demonstrating the issue.",
            },
            "proposed_rationale": {
                "type": "string",
                "description": "Technical analysis of why this cannot be resolved without test/spec tampering.",
            },
        },
        "required": ["defect_type", "target_file", "reproduction_trace", "proposed_rationale"],
    },
}

__all__ = [
    "DefectType",
    "VALID_DEFECT_TYPES",
    "escalate_defect",
    "ESCALATE_DEFECT_TOOL_SCHEMA",
]
