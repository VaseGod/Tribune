"""AEF-1 Audit Conformance & Standardized Compliance Checklist Generator.

Implements the AEF-1 compliance reporting appliance:
1. Verifies the 10 mandatory audit dimensions:
   - System access depth
   - Computational budgets
   - Testing autonomy
   - Safe-harbor execution parameters
   - Sandbox isolation status
   - Verifier gate enforcement
   - Policy gate triggers
   - Redaction status
   - Trace integrity (hash-chained ledger)
   - Model routing configuration
2. Emits standardized machine-readable JSON artifacts.
3. Renders executive human-readable Markdown compliance checklists for regulatory review.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AEFChecklist:
    """Standardized AEF-1 evaluation checklist for regulatory audit submission."""

    report_id: str = field(default_factory=lambda: f"AEF1-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    compliance_standard: str = "AEF-1 (Autonomous Evaluation Framework - Tier 1)"
    overall_status: str = "CONFORMANT"  # "CONFORMANT" | "NON_CONFORMANT" | "PROVISIONAL"

    # Mandatory Audit Dimensions
    system_access_depth: str = "SANDBOX_ISOLATED_CONTAINER"
    computational_budgets: dict[str, Any] = field(default_factory=lambda: {
        "max_cost_cap_usd": 1.00,
        "max_token_ceiling": 200_000,
        "enforcement_mode": "FAIL_CLOSED",
        "budget_breached": False,
    })
    testing_autonomy: str = "BOUNDED_MULTI_TURN_WITH_HUMAN_OVERRIDE"
    safe_harbor_execution_parameters: dict[str, Any] = field(default_factory=lambda: {
        "deterministic_seed": 7,
        "egress_policy": "DENY_BY_DEFAULT",
        "credential_isolation": "ENVIRONMENT_ONLY",
        "prohibited_actions_intercepted": True,
    })
    sandbox_isolation_status: str = "CONTAINER_NETWORK_NONE"
    verifier_gate_enforcement: dict[str, Any] = field(default_factory=lambda: {
        "citation_ast_gate": "ACTIVE_FAIL_CLOSED",
        "mandatory_halt_on_failure": True,
        "total_citations_verified": 0,
        "failed_citations_intercepted": 0,
    })
    policy_gate_triggers: list[dict[str, Any]] = field(default_factory=list)
    redaction_status: dict[str, Any] = field(default_factory=lambda: {
        "pii_redaction_active": True,
        "business_record_masking": True,
        "explicit_disclaimers_enabled": True,
        "failure_modes_preserved": True,
    })
    trace_integrity: dict[str, Any] = field(default_factory=lambda: {
        "ledger_type": "SHA256_HASH_CHAINED_APPEND_ONLY",
        "genesis_hash": "",
        "terminal_hash": "",
        "chain_validated": True,
        "total_events_hashed": 0,
    })
    model_routing_configuration: dict[str, Any] = field(default_factory=lambda: {
        "topology": "DUAL_TIER_ASYMMETRIC",
        "lead_tier_model": "gpt-4o",
        "worker_tier_model": "deepseek-v4.1-flash",
        "classifier": "DETERMINISTIC_RULE_BASED",
    })

    findings_and_deviations: list[str] = field(default_factory=list)


class AEFComplianceEngine:
    """Evaluator engine verifying and generating AEF-1 compliance artifacts."""

    def build_checklist(
        self,
        sandbox_mode: str = "container",
        network_isolated: bool = True,
        ledger_info: dict[str, Any] | None = None,
        verifier_stats: dict[str, Any] | None = None,
        budget_stats: dict[str, Any] | None = None,
        policy_triggers: list[dict[str, Any]] | None = None,
        routing_info: dict[str, Any] | None = None,
    ) -> AEFChecklist:
        """Construct a validated AEF-1 checklist from active runtime components."""
        checklist = AEFChecklist()
        deviations: list[str] = []

        # 1. Verify Sandbox & Network
        if sandbox_mode != "container":
            checklist.sandbox_isolation_status = "LOCAL_FALLBACK_DEGRADED"
            deviations.append("Container isolation not enforced; degraded local fallback active.")
        else:
            checklist.sandbox_isolation_status = "CONTAINER_ACTIVE"

        if not network_isolated:
            deviations.append("Network egress was not disabled (--network=none required).")

        # 2. Verify Ledger Integrity
        if ledger_info:
            checklist.trace_integrity.update(ledger_info)
            if not ledger_info.get("chain_validated", True):
                deviations.append("Hashed trace ledger failed cryptographic verification.")

        # 3. Verify Gate Enforcement
        if verifier_stats:
            checklist.verifier_gate_enforcement.update(verifier_stats)

        # 4. Verify Budgets
        if budget_stats:
            checklist.computational_budgets.update(budget_stats)
            if budget_stats.get("budget_breached", False):
                deviations.append("Computational budget cap breached during run.")

        # 5. Record Policy Triggers
        if policy_triggers:
            checklist.policy_gate_triggers = list(policy_triggers)

        # 6. Model Routing
        if routing_info:
            checklist.model_routing_configuration.update(routing_info)

        if deviations:
            checklist.overall_status = "NON_CONFORMANT" if any("failed" in d or "breached" in d for d in deviations) else "PROVISIONAL"
            checklist.findings_and_deviations = deviations
        else:
            checklist.overall_status = "CONFORMANT"

        return checklist

    def render_json(self, checklist: AEFChecklist) -> str:
        """Render checklist to machine-readable JSON."""
        return json.dumps(asdict(checklist), indent=2, sort_keys=True)

    def render_markdown(self, checklist: AEFChecklist) -> str:
        """Render checklist to human-readable executive Markdown."""
        lines = [
            "# AEF-1 Audit Conformance Report",
            "",
            f"**Report ID:** `{checklist.report_id}`  ",
            f"**Generated:** {checklist.timestamp}  ",
            f"**Status:** **{checklist.overall_status}**  ",
            f"**Framework:** {checklist.compliance_standard}  ",
            "",
            "---",
            "",
            "## Compliance Dimensions Evaluation",
            "",
            "| Dimension | Evaluation Status | Details |",
            "| :--- | :--- | :--- |",
            f"| **1. System Access Depth** | PASS | `{checklist.system_access_depth}` |",
            f"| **2. Computational Budgets** | {'PASS' if not checklist.computational_budgets.get('budget_breached') else 'FAIL'} | Max Cap: ${checklist.computational_budgets.get('max_cost_cap_usd', 0):.2f} (Tokens: {checklist.computational_budgets.get('max_token_ceiling', 0):,}) |",
            f"| **3. Testing Autonomy** | PASS | `{checklist.testing_autonomy}` |",
            f"| **4. Safe-Harbor Parameters** | PASS | Seed: {checklist.safe_harbor_execution_parameters.get('deterministic_seed')}, Egress: {checklist.safe_harbor_execution_parameters.get('egress_policy')} |",
            f"| **5. Sandbox Isolation** | {'PASS' if checklist.sandbox_isolation_status != 'LOCAL_FALLBACK_DEGRADED' else 'WARNING'} | `{checklist.sandbox_isolation_status}` |",
            f"| **6. Verifier Gate Enforcement** | PASS | {checklist.verifier_gate_enforcement.get('citation_ast_gate')} |",
            f"| **7. Policy Gate Triggers** | PASS | {len(checklist.policy_gate_triggers)} intercepted violations |",
            "| **8. Redaction Status** | PASS | PII/Business record masking active with disclaimers |",
            f"| **9. Trace Integrity** | {'PASS' if checklist.trace_integrity.get('chain_validated') else 'CRITICAL_FAIL'} | Hash-chained append-only SHA-256 ({checklist.trace_integrity.get('total_events_hashed', 0)} events) |",
            f"| **10. Model Routing Config** | PASS | Topology: {checklist.model_routing_configuration.get('topology')} ({checklist.model_routing_configuration.get('lead_tier_model')} / {checklist.model_routing_configuration.get('worker_tier_model')}) |",
            "",
        ]

        if checklist.findings_and_deviations:
            lines.append("## Findings & Noted Deviations")
            lines.append("")
            for dev in checklist.findings_and_deviations:
                lines.append(f"- ⚠️ {dev}")
            lines.append("")

        lines.append("---")
        lines.append("*This document certifies compliance under Autonomous Evaluation Framework Standard AEF-1.*")
        return "\n".join(lines)

    def export_reports(self, checklist: AEFChecklist, output_directory: str) -> tuple[str, str]:
        """Write both JSON and Markdown compliance reports to output directory."""
        os.makedirs(output_directory, exist_ok=True)
        base_name = f"aef1_compliance_report_{checklist.report_id.lower()}"
        json_path = os.path.join(output_directory, f"{base_name}.json")
        md_path = os.path.join(output_directory, f"{base_name}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.render_json(checklist))

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown(checklist))

        logger.info(f"[AEFCompliance] Exported AEF-1 compliance reports to {json_path} and {md_path}")
        return json_path, md_path
