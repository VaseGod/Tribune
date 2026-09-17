# AEF-1 Compliance Specification & Audit Verification Guide

## 1. Overview of the AEF-1 Standard

The **Autonomous Evaluation Framework - Tier 1 (AEF-1)** defines standardized compliance, isolation, and observability criteria for autonomous legal reasoning systems. Tribune is designed to produce verifiable evidence across all 10 AEF-1 compliance dimensions:

1. **System Access Depth**: Restricted strictly to isolated sandboxes; direct host filesystem access is blocked.
2. **Computational Budgets**: Enforces pre-execution token ceilings and cumulative dollar cost caps.
3. **Testing Autonomy**: Multi-turn autonomous trajectories must respect human kill switches and stop signals.
4. **Safe-Harbor Execution Parameters**: Offline execution with deterministic seeds, local mock rule stores, and credential isolation.
5. **Sandbox Isolation Status**: Zero-egress containerization (`--network=none`), non-root user execution, and capability dropping.
6. **Verifier Gate Enforcement**: Deterministic AST-based citation verification with mandatory fail-closed halting.
7. **Policy Gate Triggers**: Interception and audit recording of forbidden tool calls, dangerous shell scripts, and path traversals.
8. **Redaction Status**: PII and proprietary business records are masked with explicit stable markers (`[REDACTED_PII]`, `[REDACTED_BUSINESS_RECORD]`) while preserving failure modes.
9. **Trace Integrity**: Every trajectory step is appended to a cryptographically sealed SHA-256 hash-chained ledger.
10. **Model Routing Configuration**: Documented asymmetric dual-tier model assignments (Lead vs. Worker tiers).

---

## 2. Generating Compliance Reports

To generate machine-readable JSON and executive Markdown compliance reports:

```python
from tribune.redteam.aef_compliance import AEFComplianceEngine

engine = AEFComplianceEngine()
checklist = engine.build_checklist(
    sandbox_mode="container",
    network_isolated=True,
)

# Export reports
json_path, md_path = engine.export_reports(checklist, output_directory="artifacts/aef1_compliance")
print(f"Compliance artifacts saved to {json_path} and {md_path}")
```

---

## 3. Cryptographic Trace Ledger Verification

Every step executed by `HarnessLoop` is recorded in `HashedTraceLedger` (`tribune/instrumentation/tracing.py`).

### Verification Command

Run the trace verification utility:
```bash
python -m tribune.instrumentation.verify_traces artifacts/trace_ledger.json
```

### Verification Logic

The verifier executes a formal mathematical induction check:
1. Event 0 must link to the canonical genesis hash (`0000000000000000000000000000000000000000000000000000000000000000`).
2. For each Event $i$:
   $$\text{previous\_hash}_i \stackrel{?}{=} \text{current\_hash}_{i-1}$$
   $$\text{current\_hash}_i \stackrel{?}{=} \text{SHA256}(\text{CanonicalJSON}(\text{Event}_i))$$
3. Any missing step, deleted turn, or modified argument breaks the chain, causing immediate verification failure.
