# DeepSeek-V4.1-Flash Ingestion Engine & Dual-Tier Routing Architecture

## Overview

This document details the production integration of **DeepSeek-V4.1-Flash** into Tribune as the primary ingestion and context-engineering engine. The implementation restructures Tribune's provider routing layer and three core agent abstractions — **Preparer**, **Navigator**, and **Verifier** — to establish extreme economic efficiency while mitigating verbosity loops, multi-agent coordination fragility, and trajectory sprawl.

---

## 1. Dual-Tier Routing Architecture

Tribune employs an asymmetric dual-tier provider routing topology:

```
                          ┌──────────────────────────┐
                          │ Incoming Task / Case Run │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Economic Router Decision    │
                        │    (tribune.providers.router) │
                        └───────┬───────────────┬───────┘
                                │               │
          Ingestion / Context / │               │ Final Adjudication /
          PARSER / Delta Repair │               │ Complex Disambiguation
                                ▼               ▼
                    ┌──────────────────┐ ┌──────────────────┐
                    │ DeepSeek-V4.1-   │ │ Frontier Models  │
                    │ Flash Engine     │ │ (Claude Sonnet / │
                    │ ($0.006-$0.30/M) │ │  GPT-4o / O3)    │
                    └──────────────────┘ └──────────────────┘
```

### Task Classification & Policy Matrix

1. **Ingestion Tasks** (`DOCUMENT_INGESTION`, `CORPUS_EXTRACTION`, `CHUNKING`):
   - **Engine**: DeepSeek-V4.1-Flash exclusively.
   - **Constraint**: Strict cost-aware fallback protection ensures ingestion is *never* escalated to frontier models, preventing massive token billing spikes.
2. **Context Engineering** (`CONTEXT_PREPARATION`, `PARSER_SCATTER_GATHER`, `PREFIX_COMPACTION`):
   - **Engine**: DeepSeek-V4.1-Flash with invariant prefix caching and PARSER scatter-gather parallelism.
3. **Exploration & Navigation** (`TRAJECTORY_NAVIGATION`, `EVIDENCE_RETRIEVAL`):
   - **Engine**: DeepSeek-V4.1-Flash governed by the Elastic Horizon Controller and 512-token thought ceiling.
4. **Verification & Quality Gate** (`DEFECT_VERIFICATION`, `SINGLE_TURN_REWRITE`):
   - **Engine**: DeepSeek-V4.1-Flash for targeted delta patch synthesis, backed by ToolGrad assertion checkers.
5. **Frontier Adjudication** (`LEGAL_INTERPRETATION`, `AMBIGUITY_RESOLVE`):
   - **Engine**: Reserved frontier models (Claude 3.5 Sonnet, GPT-4o) only for high-entropy ambiguity resolution and final legal certification.

---

## 2. Prompt Caching Economics

DeepSeek-V4.1-Flash features industry-leading cache read economics that unlock >85% cost reduction when system prompts and static corpora are preserved across turns.

### Price Schedule (tribune/eval/pricing.json)

| Token Category | Rate per Million Tokens ($/1M) | Relative Cost Multiplier |
| :--- | :--- | :--- |
| **Cached Input** (`cache_read_tokens`) | **$0.006** | 1.0× (Baseline) |
| **Uncached Input** (`tokens_input` miss) | **$0.300** | 50.0× |
| **Output / Completion** (`tokens_output`) | **$1.200** | 200.0× |

### Maximizing Cache Hit Rate (>95%)

1. **Deterministic Static Anchoring**: Static context (system instructions, rules, statutes, schemas) must remain identical and positioned strictly at the start of prompts.
2. **Separation of Concerns**: Dynamic user turns, variable timestamps, and applicant facts must never be interpolated into the invariant prefix block.
3. **Cache Key Headers**: The `DeepSeekProvider` automatically passes `X-Prompt-Cache-Key: <sha256>` computed from the invariant prefix to ensure upstream server-side KV reuse.
4. **Cache Observability**: Per-call `ProviderUsage` tracks `cached_tokens`, `prompt_tokens`, `completion_tokens`, and calculates `cache_hit_rate = cached_tokens / (cached_tokens + uncached_tokens)`.

---

## 3. Invariant Prefix Structure & Canonical Serialization

To guarantee consistent server-side cache hits across multi-turn sessions, `tribune.agents.preparer.InvariantPrefixSerializer` enforces a deterministic 7-stage prefix layout:

```
┌────────────────────────────────────────────────────────┐
│ 1. System Identity & Protocol Specification            │
├────────────────────────────────────────────────────────┤
│ 2. Canonical JSON Schema Definitions (Sorted Keys)     │
├────────────────────────────────────────────────────────┤
│ 3. Statutory Corpus Rules & Citations (Sorted CIDs)    │
├────────────────────────────────────────────────────────┤
│ 4. Fixed Jurisdiction Context & Base Definitions       │
├────────────────────────────────────────────────────────┤
│ 5. Tool / Function Calling Declarations (Sorted Names) │
├────────────────────────────────────────────────────────┤
│ 6. Guardrail Invariants & Operational Constraints      │
├────────────────────────────────────────────────────────┤
│ 7. Verification Rubrics & Legal Proof Criteria         │
└────────────────────────────────────────────────────────┘
```

### Serialization Rules

- **Canonical JSON Formatting**: All dictionary objects are serialized with `json.dumps(obj, sort_keys=True, separators=(',', ':'))` with `ensure_ascii=False`.
- **Alphabetical Ordering**: Citations, program keys, tool definitions, and rubric clauses are sorted lexicographically before rendering.
- **SHA-256 Digest Verification**: Every serialized prefix carries a computed `sha256` checksum (`prefix_hash`). Tests assert bitwise equality between independent serialization invocations.
- **Dynamic Tail Isolation**: Dynamic user state, transient session history, and runtime evidence are isolated into `DynamicTailSpec` and appended strictly after the invariant boundary.

---

## 4. PARSER Scatter-Gather Ingestion Engine

Large-scale document ingestion is executed through `PARSERScatterGatherEngine` (`tribune/agents/preparer.py`).

### Architecture

```
                       Raw Document List
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Chunk 0 (Worker 0)                    Chunk 1 (Worker 1)
     PARSER Prompt (Budgeted)              PARSER Prompt (Budgeted)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                    Deterministic Gathering
                    (Sorted by Chunk Index)
                               │
                               ▼
                   GatheredIngestionResult
```

### Key Capabilities

1. **Scatter Phase**: Partitions raw multi-document inputs into discrete chunks (default: 8KB per chunk, max 8 concurrent threads).
2. **Per-Chunk Token Budgeting**: Enforces strict input/output token envelopes (`parser_chunk_token_budget`, default 4,000 tokens) to prevent unbounded fan-out.
3. **Structured Entity Extraction**: Each worker extracts citations, applicant facts, monetary values, and metadata in structured JSON format.
4. **Gather Phase**: Sorts chunks deterministically by index and merges extracted entities, deduplicating citations while preserving source provenance.
5. **1M Context Safeguard**: Total accumulated tokens are verified against `preparer_context_token_limit` (default 1,000,000 tokens).

---

## 5. Elastic Horizon Controller & Navigator Anti-Verbosity

To eliminate runaway trajectories, cycle loops, and intermediate token bloat in `tribune.agents.navigator`, Tribune implements dynamic trajectory bounds and intermediate thought pruning.

### Elastic Horizon Calculation

The `ElasticHorizonController` maintains a rolling window of past successful trajectory lengths ($N=50$):

$$\text{Horizon} = \text{clamp}\left(\lceil p_{90}(\text{history}) \times 1.25 \rceil, \; \text{min\_turns}=8, \; \text{max\_turns}=24\right)$$

If history is empty, the controller falls back to the default bound (`navigator_max_horizon=16`).

### Anti-Verbosity Controls

- **512-Token Intermediate Thought Ceiling**: Intermediate scratchpad reasoning within tool-calling steps is strictly limited to 512 tokens. Thoughts exceeding 512 tokens are cleanly truncated with a marker `[... reasoning truncated to 512-token ceiling ...]` to protect cache efficiency and prevent verbosity drift.
- **Loop & Stagnation Detection**: The `LoopDetector` analyzes sliding windows ($W=3$) of tool calls, arguments, and reasoning hashes. If repeated identical calls or no-op actions are detected, the loop detector triggers an early termination signal.
- **Unified Single-Agent Topology**: Disables redundant subagent handoffs by default (`single_agent_topology=True`), preventing multi-agent coordination latency and trajectory sprawl.

---

## 6. Single-Turn Delta Rewriter & ToolGrad Assertions

The `tribune.agents.verifier` module replaces full-trajectory regeneration with surgical delta rewriting and enforces deterministic execution boundaries.

### Single-Turn Delta Rewriter

Instead of re-executing entire multi-turn trajectories upon finding a defect:

1. **DefectTurnLocator**: Pinpoints the exact turn index and failure category (`COMPILATION_SYNTAX`, `LINTING_POLICY`, `TEST_ASSERTION`, `SCHEMA_VIOLATION`, `RUNTIME_EXCEPTION`, `PATH_TRAVERSAL`).
2. **Isolated Turn Patching**: Synthesizes a targeted correction for the failing turn only, feeding the defect diagnosis and local context into DeepSeek-V4.1-Flash.
3. **Trajectory Preservation**: Replaces the single defective turn in-place while keeping preceding context and subsequent valid states unaltered.
4. **Trajectory Regeneration Count = 0**: Full trajectory re-executions are completely averted.

### ToolGrad Assertion Checker

Before any generated tool invocation is committed or dispatched, `ToolGradAssertionChecker.validate_tool_call(tool_name, arguments)` validates safety constraints:

1. **Schema Typing & Enum Validation**: Checks required parameters, types (`string`, `integer`, `boolean`, `array`), and permissible enum values against tool signatures.
2. **Path Traversal Confinement**: Rejects paths containing `..` or absolute paths outside the designated workspace sandbox root (`SecurityViolationError`).
3. **Command Safety Gate**: Prohibits dangerous shell constructs (`rm -rf`, `mkfs`, `dd`, piped downloads `curl ... | sh`, fork bombs, unauthorized network bindings).

---

## 7. Configuration Reference (`tribune/config.py`)

| Setting Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `deepseek_model_name` | `str` | `"deepseek-chat"` | Model identifier for DeepSeek API calls. |
| `deepseek_flash_endpoint` | `str` | `"https://api.deepseek.com/v1"` | Base URL for DeepSeek API endpoints. |
| `enable_prompt_caching` | `bool` | `True` | Enables invariant prefix caching and header tracking. |
| `deepseek_cache_headers` | `bool` | `True` | Emits `X-Prompt-Cache-Key` in completion requests. |
| `deepseek_temperature_override`| `float` | `1.0` | Calibrated generation temperature (standard for DeepSeek reasoning). |
| `deepseek_reasoning_effort` | `str` | `"high"` | High reasoning effort profile for complex legal reasoning. |
| `deepseek_ingestion_role` | `str` | `"primary"` | Role assignment (`"primary"`, `"secondary"`, `"disabled"`). |
| `router_task_policy` | `dict` | Policy Table | Dynamic routing mapping task types to provider tiers. |
| `frontier_providers` | `list[str]`| `["anthropic", "openai"]` | Providers reserved for frontier escalation tasks. |
| `preparer_context_token_limit` | `int` | `1_000_000` | Maximum context budget for the Preparer engine. |
| `parser_chunk_token_budget` | `int` | `4_000` | Token budget ceiling per PARSER scatter chunk. |
| `parser_concurrency_workers` | `int` | `8` | Maximum concurrent threads for scatter-gather ingestion. |
| `navigator_max_horizon` | `int` | `16` | Static fallback horizon if rolling history is absent. |
| `navigator_p90_safety_multiplier`| `float` | `1.25` | Safety factor applied to rolling p90 trajectory steps. |
| `navigator_min_horizon` | `int` | `8` | Lower bound clamp on the elastic horizon. |
| `navigator_absolute_max_horizon`| `int` | `24` | Upper bound clamp on the elastic horizon. |
| `navigator_intermediate_thought_token_ceiling` | `int` | `512` | Token limit on scratchpad reasoning per tool call. |
| `navigator_single_agent_topology` | `bool` | `True` | Enforces single-agent loop to prevent multi-agent sprawl. |
| `verifier_single_turn_rewrite_enabled` | `bool` | `True` | Enables targeted delta repair vs. full regeneration. |
| `verifier_toolgrad_strict_mode` | `bool` | `True` | Rejects tool calls violating schema or path safety. |

---

## 8. Benchmarking Methodology & Metric Harness

### Metric Instrumentation (`tribune/metrics.py`)

Every task run records structured roadmap metrics through `RoadmapMetricsCollector`:

- `tokens_input_uncached`, `tokens_input_cached`, `tokens_output`
- `cost_usd`: Computed with exact rate multipliers ($0.30 uncached / $0.006 cached / $1.20 output per 1M).
- `cache_hit_rate`: Measured ratio of cached tokens to total input tokens.
- `elastic_horizon_breaches`: Count of tasks exceeding dynamic horizon bounds.
- `intermediate_verbosity_violations`: Count of turns whose reasoning exceeded 512 tokens.
- `single_turn_delta_rewrites`: Count of isolated repairs executed.
- `full_trajectory_regenerations`: Asserted to remain strictly 0.
- `toolgrad_schema_violations` & `toolgrad_path_violations`: Intercepted security violations.

### Running the Evaluation Suite

Execute tests and verification locally without live network dependencies:

```bash
# Run all DeepSeek roadmap tests
PYTHONPATH=. .venv/bin/pytest tests/test_deepseek_roadmap.py -v

# Run the complete provider and routing test suite
PYTHONPATH=. .venv/bin/pytest tests/test_deepseek_roadmap.py tests/test_model_router.py tests/test_prompt_caching_and_cost.py tests/test_metrics.py

# Verify code formatting and linting
.venv/bin/ruff check tribune/providers/deepseek.py tribune/providers/router.py tribune/agents/preparer.py tribune/agents/navigator.py tribune/agents/verifier.py tribune/config.py tribune/metrics.py tests/test_deepseek_roadmap.py
```

### Production Evaluation & Benchmark Interpretation

When running against live serving endpoints (with `DEEPSEEK_API_KEY` set):

1. **Ingestion Throughput ($11\times$ Target)**: Compare wall-clock execution time of `PARSERScatterGatherEngine` against sequential single-pass extraction across multi-page statutory PDF bundles.
2. **Cost Reduction ($>85\%$ Target)**: Aggregate `cost_usd` across 1,000 synthetic case runs under dual-tier routing vs. legacy frontier-only baseline.
3. **Cache Hit Efficiency ($>95\%$ Target)**: Verify that multi-turn sessions against invariant static corpora achieve $\ge 95\%$ `cache_hit_rate` starting from Turn 2 onwards.
4. **Trajectory Token Reduction ($25\%$ Target)**: Measure intermediate token usage per trajectory turn before and after applying the 512-token thought ceiling and single-turn delta rewriting.
5. **Task Retention ($+12$ to $+30$ Target)**: Measure task success rate under tight timeout constraints; single-turn delta rewriting prevents timeouts caused by re-running 10+ turns from scratch.
