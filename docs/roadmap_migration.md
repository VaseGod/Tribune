# Tribune Architectural Roadmap Migration Guide

## Executive Summary

This document describes the architectural upgrade of the Tribune repository from a monolithic execution model to a modular, container-isolated, verifier-gated, dual-tier orchestrated, and AEF-1 audit-conformant platform.

---

## 1. Roadmap Phase Mapping

| Roadmap Phase | Core Requirements | Implemented Modules |
| :--- | :--- | :--- |
| **Phase 1: Modular Harness Decoupling & Container Isolation** | - Unified inference provider layer<br>- Execution harness separating inference, tools, verifier, state<br>- Hardened container sandbox with local fallback | - `tribune/inference/base.py`<br>- `tribune/inference/openai_compatible.py`<br>- `tribune/inference/commercial.py`<br>- `tribune/inference/registry.py`<br>- `tribune/harness/loop.py`<br>- `tribune/harness/state.py`<br>- `tribune/harness/context.py`<br>- `sandbox/sandbox_runtime.py`<br>- `sandbox/Dockerfile` |
| **Phase 2: Deterministic Verifier Gates & Dual-Tier Orchestration** | - AST citation verifier gate<br>- Mandatory gate fail-closed policy<br>- Dual-tier orchestrator (lead vs. worker)<br>- Standardized compact Markdown schemas | - `tribune/corpus/citation_ast.py`<br>- `tribune/corpus/citations.py`<br>- `tribune/casegen/simulation.py`<br>- `tribune/casegen/markdown_schema.py`<br>- `tribune/casegen/synthetic.py` |
| **Phase 3: Calibrated Reasoning Budgets** | - Deterministic task complexity classifier<br>- Dynamic routing between worker & lead tiers<br>- Reasoning budget metadata & cap enforcement | - `tribune/casegen/task_classifier.py`<br>- `tribune/casegen/reasoning_budget.py`<br>- `tribune/inference/costs.py`<br>- `docs/quant_sensitivity.md`<br>- `docs/cost_accounting.md` |
| **Phase 4: AEF-1 Audit Conformance & Immutable Tracing** | - Standardized AEF-1 compliance checklist<br>- Explicit redaction disclaimers preserving failure evidence<br>- Cryptographic SHA-256 hash-chained trace ledger<br>- Runtime policy gates with fail-closed enforcement | - `tribune/redteam/aef_compliance.py`<br>- `tribune/security/sanitization.py`<br>- `tribune/security/policies.py`<br>- `tribune/instrumentation/tracing.py`<br>- `tribune/instrumentation/verify_traces.py`<br>- `docs/aef1_compliance.md` |

---

## 2. Configuration & Feature Flags

All new features honor the configuration hierarchy:
`Hardcoded Defaults` $\rightarrow$ `config.yaml` $\rightarrow$ `Environment Variables` $\rightarrow$ `Runtime Arguments`.

| Setting / Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `lead_model_name` | `TRIBUNE_LEAD_MODEL` | `gpt-4o` | Frontier/lead reasoning model name. |
| `worker_model_name` | `TRIBUNE_WORKER_MODEL` | `deepseek-v4.1-flash` | Low-cost worker model name. |
| `lead_provider` | `TRIBUNE_LEAD_PROVIDER` | `openai_compatible` | Lead tier provider backend. |
| `worker_provider` | `TRIBUNE_WORKER_PROVIDER` | `openai_compatible` | Worker tier provider backend. |
| `reasoning_budget_cap_usd` | `TRIBUNE_REASONING_BUDGET_CAP_USD` | `1.00` | Max dollar budget per task trajectory. |
| `reasoning_budget_cap_tokens`| `TRIBUNE_REASONING_BUDGET_CAP_TOKENS`| `200000` | Max token limit per task trajectory. |
| `citation_verifier_strict` | `TRIBUNE_CITATION_VERIFIER_STRICT` | `true` | When true, ungrounded citations immediately fail the task. |
| `sandbox_mode` | `TRIBUNE_SANDBOX_MODE` | `container` | Execution mode: `container`, `local_fallback`, `disabled`. |
| `sandbox_network_isolated` | `TRIBUNE_SANDBOX_NETWORK_ISOLATED` | `true` | Enforces `--network=none` at container boundary. |
| `trace_hashing_enabled` | `TRIBUNE_TRACE_HASHING_ENABLED` | `true` | Emits SHA-256 chained trace ledger. |
| `redaction_enabled` | `TRIBUNE_REDACTION_ENABLED` | `true` | Masks PII/business data with explicit disclaimers. |
| `legacy_monolithic_execution`| `TRIBUNE_LEGACY_MONOLITHIC_EXECUTION`| `false` | Fallback flag for legacy monolithic execution. |

---

## 3. Migration Instructions for Existing Callers

1. **Inference Migration**:
   - Legacy: Calling internal LLM wrappers directly.
   - New: Use `get_inference_registry().get_provider()` or `registry.complete(InferenceRequest(...))`.
2. **Citation Verification**:
   - Legacy: `LateInteractionRetriever` or raw text matching.
   - New: `CitationVerifierGate.verify_citation(citation_text)`.
3. **Task Trajectory Execution**:
   - Legacy: Monolithic custom loops.
   - New: `HarnessLoop(provider=..., verifier_callback=...).run(...)`.
4. **Case Serialization**:
   - Legacy: Verbose JSON strings.
   - New: `SyntheticCase.to_markdown()` and `SyntheticCase.from_markdown()`.
