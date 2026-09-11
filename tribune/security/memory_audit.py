"""Memory-Mapped Knowledge Store & Ngram Hot-Swap Auditing Protocol.

Audits runtime memory-mapped knowledge stores (e.g., llama.cpp Ngram hot-swapping
and dynamic Engram RAM lookups) to ensure runtime injected tables cannot overwrite,
shadow, or collide with canonical statutory constraints registered in rule_store.py.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import Field

from ..corpus.rule_store import LocalRuleStore
from ..types import StrictModel
from .audit import SecurityEventType, record_security_event

logger = logging.getLogger(__name__)


class ShadowCollisionViolationError(PermissionError):
    """Raised when an injected memory-mapped table attempts to shadow canonical safety constraints."""

    def __init__(
        self,
        conflicting_key: str,
        canonical_value: Any,
        attempted_value: Any,
        table_name: str,
        message: str | None = None,
    ) -> None:
        msg = (
            message
            or f"Memory audit violation: Injected key '{conflicting_key}' in table '{table_name}' "
            f"attempts to shadow canonical value '{canonical_value}' with '{attempted_value}'."
        )
        super().__init__(msg)
        self.conflicting_key = conflicting_key
        self.canonical_value = canonical_value
        self.attempted_value = attempted_value
        self.table_name = table_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "ShadowCollisionViolationError",
            "table_name": self.table_name,
            "conflicting_key": self.conflicting_key,
            "canonical_value": str(self.canonical_value),
            "attempted_value": str(self.attempted_value),
        }


class MemoryAuditReport(StrictModel):
    """Report summarizing security verification of memory-mapped knowledge tables and Ngram maps."""

    is_secure: bool
    tables_audited: int
    shadow_collisions_detected: int
    flagged_keys: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


# Reserved statutory keys that may never be shadowed or overwritten dynamically
_CANONICAL_SAFETY_KEYS = {
    "gross_income_fpl_pct",
    "net_income_fpl_pct",
    "standard_deduction",
    "asset_limit_standard",
    "expansion_adult",
    "extremely_low_pct",
    "min_base_earnings",
    "window_days",
    "ActionGate",
    "POL-ANTI-REWARD-HACK-001",
    "escalate_defect",
    "HumanSignoff",
}

# Dangerous Ngram substitution targets
_RESTRICTED_NGRAM_TARGETS = [
    re.compile(r"\b(?:POL-ANTI-REWARD-HACK-001|ActionGate|HumanSignoff)\b", re.IGNORECASE),
    re.compile(r"\b(?:7\s*CFR\s*273|42\s*CFR\s*435|24\s*CFR\s*982|20\s*CFR\s*604)\b", re.IGNORECASE),
    re.compile(r"\b(?:likely_eligible|likely_ineligible|abstain_and_escalate)\b", re.IGNORECASE),
    re.compile(r"\b(?:escalate_defect|raise\s+ActionBlocked)\b", re.IGNORECASE),
]


class MemoryAuditProtocol:
    """Auditing protocol for runtime memory-mapped tables and Ngram hot-swapping."""

    def __init__(self, canonical_rule_store: LocalRuleStore | None = None) -> None:
        self.rule_store = canonical_rule_store or LocalRuleStore()
        self._registered_tables: dict[str, dict[str, Any]] = {}
        self._registered_ngrams: dict[str, str] = {}

    def audit_memory_mapped_table(
        self,
        table_name: str,
        key_value_map: dict[str, Any],
        allow_override: bool = False,
    ) -> MemoryAuditReport:
        """Audit a dynamic memory-mapped table against canonical statutory rules.

        Raises ShadowCollisionViolationError if any key attempts to shadow or mutate
        canonical safety constraints without explicit authorization.
        """
        flagged: list[str] = []
        collisions = 0
        details: dict[str, Any] = {}

        # 1. Check against reserved canonical safety keys
        for key, attempted_val in key_value_map.items():
            if key in _CANONICAL_SAFETY_KEYS:
                flagged.append(key)
                collisions += 1
                details[key] = {
                    "reason": "Reserved canonical safety key collision",
                    "attempted_value": attempted_val,
                }
                if not allow_override:
                    record_security_event(
                        event_type=SecurityEventType.SECURITY_VIOLATION,
                        source="tribune.security.memory_audit.MemoryAuditProtocol",
                        message=f"Shadow collision attempt on reserved safety key '{key}' in table '{table_name}'",
                        severity="CRITICAL",
                        details={"table": table_name, "key": key, "attempted_value": attempted_val},
                    )
                    raise ShadowCollisionViolationError(
                        conflicting_key=key,
                        canonical_value="CANONICAL_PROTECTED_SPECIFICATION",
                        attempted_value=attempted_val,
                        table_name=table_name,
                    )

            # 2. Check against statutory criterion IDs
            existing_rule = None
            for r in self.rule_store._rule_lookup.values():
                if r.criterion_id == key:
                    existing_rule = r
                    break

            if existing_rule is not None:
                flagged.append(key)
                collisions += 1
                details[key] = {
                    "reason": "Statutory rule criterion ID collision",
                    "attempted_value": attempted_val,
                }
                if not allow_override:
                    record_security_event(
                        event_type=SecurityEventType.SECURITY_VIOLATION,
                        source="tribune.security.memory_audit.MemoryAuditProtocol",
                        message=f"Shadow collision attempt on statutory rule '{key}'",
                        severity="CRITICAL",
                        details={"table": table_name, "key": key, "attempted_value": attempted_val},
                    )
                    raise ShadowCollisionViolationError(
                        conflicting_key=key,
                        canonical_value=existing_rule.title,
                        attempted_value=attempted_val,
                        table_name=table_name,
                    )

        # Store audited table
        self._registered_tables[table_name] = dict(key_value_map)

        is_sec = collisions == 0
        return MemoryAuditReport(
            is_secure=is_sec,
            tables_audited=1,
            shadow_collisions_detected=collisions,
            flagged_keys=flagged,
            details=details,
        )

    def audit_ngram_hot_swap(
        self,
        ngram_map: dict[str, str],
        namespace: str = "default",
    ) -> MemoryAuditReport:
        """Audit hot-swapped Ngram maps before activation.

        Ensures that Ngram substitutions cannot rewrite statutory citations or policy phrases.
        """
        flagged: list[str] = []
        collisions = 0
        details: dict[str, Any] = {}

        for src_ngram, target_ngram in ngram_map.items():
            # Check if source or target tampers with restricted policy patterns
            for pat in _RESTRICTED_NGRAM_TARGETS:
                if pat.search(src_ngram) or pat.search(target_ngram):
                    flagged.append(src_ngram)
                    collisions += 1
                    details[src_ngram] = {
                        "reason": "Ngram map attempts to alter restricted safety/citation phrase",
                        "target_replacement": target_ngram,
                    }
                    record_security_event(
                        event_type=SecurityEventType.SECURITY_VIOLATION,
                        source="tribune.security.memory_audit.MemoryAuditProtocol",
                        message=f"Restricted Ngram replacement detected: '{src_ngram}' -> '{target_ngram}'",
                        severity="CRITICAL",
                        details={"namespace": namespace, "source": src_ngram, "target": target_ngram},
                    )
                    raise ShadowCollisionViolationError(
                        conflicting_key=src_ngram,
                        canonical_value=src_ngram,
                        attempted_value=target_ngram,
                        table_name=f"ngram_hot_swap::{namespace}",
                    )

        self._registered_ngrams.update(ngram_map)

        is_sec = collisions == 0
        return MemoryAuditReport(
            is_secure=is_sec,
            tables_audited=1,
            shadow_collisions_detected=collisions,
            flagged_keys=flagged,
            details=details,
        )


__all__ = [
    "ShadowCollisionViolationError",
    "MemoryAuditReport",
    "MemoryAuditProtocol",
]
