# Tribune Hardened Three-Tier Evaluation Architecture: Implementation Plan

**Author:** Autonomous Systems & Benchmark-Infrastructure Engineering  
**Date:** 2026-09-24  
**Status:** Approved & In-Progress  
**Target:** Tribune Hardened Roadmap Integration

---

## 1. Executive Summary & Project Goal

The Tribune evaluation harness evaluates model sensitivity and abstention calibration across quantization depths (FP16, FP8, 4-bit AWQ, GGUF Q4/Q3/Q2/IQ1). In low-bit and distilled models, degraded attention causes severe operational decay: models lose track of working directories, repeatedly reissue failed commands with identical syntax, hallucinate missing file paths, lose subgoal state, or prematurely declare completion before completing synthetic verification loops.

This project refactors the evaluation harness from a monolithic shell-execution loop into a hardened **three-tier architecture**:

1. **Host Tier (Orchestration & Security Core):**
   - Runs evaluation ladder orchestration (`ladder.py`).
   - Hosts the **Sentinel security broker** daemon on a local Unix domain socket.
   - Hosts the **Proactive Memory Sidecar engine** and state bank.
   - Holds authentic credentials and evaluation tokens strictly out-of-band.
2. **Controlled Execution Tier (Sealed Sandbox):**
   - An unprivileged, sealed runtime container (`ContainerShellRunner`) or isolated fallback (`LocalFallbackRunner`).
   - Executes the quantized action model and sandboxed shell environment with dropped capabilities and non-root execution.
   - Receives only synthetic surrogate tokens (`MOCK_ACCESS_HANDLE_ALPHA`, `MOCK_API_KEY_BRAVO`, etc.), never real credentials.
   - Network egress is deny-by-default; outbound traffic must be brokered and allowlist-gated by Sentinel.
3. **Auxiliary Intelligence Tier (Proactive Memory Sidecar):**
   - Uses an auxiliary fast model or calibrated inference path to maintain a structured 5-track execution memory state bank.
   - Intervenes only under sparse, policy-gated conditions (e.g. repeated failed command loops, non-existent paths, premature completion declarations).
   - Returns explicit silence (`MEMORY_INTERVENTION: SILENCE`) by default to prevent context pollution.

---

## 2. Current Execution Flow vs. Target Execution Flow

### 2.1 Current Monolithic Flow

```
[Ladder / Evaluation Run]
       │
       ▼
[CasePipeline / Runner] ──(raw shell / direct tools)──► [Local OS / Sandbox Subprocess]
       │                                                         │
       ◄────────────────(raw stdout/stderr/exit code)───────────┘
       │
   (Unfiltered tokens, raw environment, possible secret leaks)
       │
       ▼
[Action Model Context] (Accumulates error loops, directory disorientation, hallucinated paths)
```

**Limitations:**
- No dual-zone trust separation between orchestrator and untrusted action model execution.
- Real credentials in environment variables can leak into stdout/stderr or model prompts.
- Degraded action models loop endlessly on identical command errors or declare success prematurely.
- Monolithic cost model only tracks single-agent token counts and fails to account for auxiliary sidecars or loop avoidance economics.

### 2.2 Target Hardened Three-Tier Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ HOST TIER (Trusted Zone)                                                               │
│                                                                                        │
│  [run_ladder()] ──► [TrajectoryBuffer] ◄────► [ProactiveMemorySidecar]                 │
│         │                                             │                                │
│         ▼                                             ▼ (every k turns, default k=2)   │
│   [ExecRequest]                              [Sparse Intervention Gate]                │
│         │                                             │                                │
│         │ (Surrogate Tokens only)                     ▼                                │
│         │                                   {SILENCE | System Reminder}                │
│         ▼                                             │                                │
│  ┌──────────────┐     UDS Socket IPC                  │                                │
│  │   SENTINEL   │ ◄─────────────────────┐             │ (Injected into next prompt)    │
│  │ POLICY BROKER│                       │             │                                │
│  │ & TOKEN VAULT│ ──► [Audit Logger]   │             ▼                                │
│  └──────┬───────┘                       │    [Quantized Action Model]                  │
│         │ (Host Auth Injection)         │             ▲                                │
│         ▼                               │             │                                │
│   [Outbound Net                         │             │                                │
│    Allowlist Gateway]                   │             │                                │
└─────────┼───────────────────────────────┼─────────────┼────────────────────────────────┘
          │                               │             │
          │ Proxy egress                  │ Check/Exec  │ Prompt & Tool Invocations
          ▼                               │             ▼
┌─────────────────────────────────────────┴──────────────────────────────────────────────┐
│ CONTROLLED EXECUTION TIER (Unprivileged Sandbox Zone)                                  │
│                                                                                        │
│  [ContainerShellRunner / LocalFallbackRunner]                                          │
│  - Non-root user (`tribune-sandbox`)                                                   │
│  - Read-only root filesystem + tmpfs for writable paths                                │
│  - Capabilities dropped (`--cap-drop=ALL`)                                             │
│  - Environment contains ONLY `MOCK_*` surrogate tokens                                 │
│  - Outbound network deny-by-default                                                    │
│  - Redacted stdout/stderr before returning to model context                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modules to Modify and New Modules to Add

### 3.1 Existing Modules to Modify

| Module | Location | Planned Modifications |
|---|---|---|
| `backends.py` | `tribune/eval/quant_sensitivity/backends.py` | Integrate runner selection, execution mode flags (`container`, `local_fallback`, `legacy`), Sentinel coordination, and surrogate token injection. |
| `ladder.py` | `tribune/eval/quant_sensitivity/ladder.py` | Integrate `TrajectoryBuffer`, invoke `ProactiveMemorySidecar` at interval $k=2$, inject sparse reminders/challenges, record intervention metadata into `RungResult`. |
| `seedset.py` | `tribune/eval/quant_sensitivity/seedset.py` | Extend synthetic cases and tasks with structured, machine-readable task requirements (`expected_artifacts`, `expected_files`, `completion_criteria`, `expected_command_success_criteria`, `forbidden_actions`). |
| `synthetic_env_verifier.py` | `tribune/eval/synthetic_env_verifier.py` | Add structured verification outputs (`requirement_id`, `satisfied`, `evidence`, `missing_artifacts`, `failed_checks`, `severity`, `reproducible_command_hint`) to feed `unmet_task_requirements`. |
| `costmodel.py` | `tribune/eval/costmodel.py` | Implement dual-agent trajectory cost formula: $Cost = \sum [C_{action} + I_{eval} \cdot C_{mem}]$; track auxiliary overhead, avoided-loop savings, net efficiency. |
| `report.py` | `tribune/eval/quant_sensitivity/report.py` | Add comparative reporting across Baseline, Sentinel-only, and Sentinel+Memory; report turns, cost delta, loop reduction, and net trajectory efficiency score. |
| `audit.py` | `tribune/security/audit.py` | Support Sentinel audit event types (`command_requested`, `command_allowed`, `command_denied`, `network_requested`, etc.) with automatic secret scrubbing. |
| Documentation | `docs/security/sandbox_shell_execution.md`, `evidence.json` | Update architecture, operational runbooks, socket specs, threat models, and verifiable milestone evidence. |

### 3.2 New Modules to Add

| Subsystem | New Module | Primary Purpose |
|---|---|---|
| Runtime & Execution | `tribune/runtime/exec_request.py` | Data contracts: `ExecRequest`, `ExecResult`, `ShellRunner` protocol. |
| Runtime & Execution | `tribune/runtime/container_runner.py` | `ContainerShellRunner`: Docker/containerized unprivileged runner with capability dropping and surrogate env. |
| Runtime & Execution | `tribune/runtime/local_runner.py` | `LocalFallbackRunner`: Development fallback with token surrogateing, allowlist enforcement, and `unsafe_for_production=True`. |
| Runtime & Execution | `tribune/runtime/__init__.py` | Export runner classes and factory functions. |
| Security | `tribune/security/token_broker.py` | `TokenBroker`: Host-only secret mapping, surrogate token generation (`MOCK_*`), bidirectional scrubbing and stdout/stderr redaction. |
| Security | `tribune/security/allowlist.py` | `AllowlistPolicy`: Destination parsing, CIDR/domain/port/method validation, max requests per task enforcement. |
| Security | `tribune/security/sentinel.py` | `SentinelDaemon`: Local Unix domain socket listener & policy broker; handles command checks, egress checks, and host-side auth injection. |
| Memory Sidecar | `tribune/memory/state_bank.py` | 5-track operational state bank: `EnvironmentState`, `MutationRecord`, `TaskRequirement`, `Subgoal`, `FailedCommandCache`, `MemoryStateBank`. |
| Memory Sidecar | `tribune/memory/intervention_policy.py` | Sparse intervention rules (failed command recurrence, invalid path, premature completion, repeated loops, silence default). |
| Memory Sidecar | `tribune/memory/trajectory_buffer.py` | Multi-turn trajectory event log tracking commands, outputs, exits, and turns. |
| Memory Sidecar | `tribune/memory/aux_backend.py` | Auxiliary intelligence paths (heuristic fast mock, local 8-bit, API-based). |
| Memory Sidecar | `tribune/memory/sidecar.py` | `ProactiveMemorySidecar`: Orchestrates asynchronous/synchronous trajectory updates every $k$ turns. |
| Configuration | `tribune/eval/quant_sensitivity/hardening_config.py` | Unified configuration dataclass supporting args, env, yaml/json, and safe defaults. |
| Configuration | `config/sentinel_allowlist.yaml` | Production/evaluation allowlist rules and auth injection profiles. |
| Re-exports | `tribune/eval/quant_sensitivity/costmodel.py` | Clean re-export of cost model for quant_sensitivity namespace. |

---

## 4. Data Contracts

### 4.1 Execution Contracts (`tribune/runtime/exec_request.py`)

```python
@dataclass
class ExecRequest:
    command: str | list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    session_id: str = "default_session"
    task_id: str = "default_task"
    model_backend_id: str = "quant_model"
    requested_network_destinations: list[str] = field(default_factory=list)

@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    redacted_command: str = ""
    surrogate_env_keys: list[str] = field(default_factory=list)
    sentinel_decision: str = "ALLOWED"
    intervention_metadata: dict[str, Any] | None = None
```

### 4.2 State Bank Contracts (`tribune/memory/state_bank.py`)

```python
@dataclass
class EnvironmentState:
    working_directory: str = "/workspace"
    path_variables: list[str] = field(default_factory=list)
    active_virtual_environment: str | None = None
    visible_files: list[str] = field(default_factory=list)
    relevant_environment_variables: dict[str, str] = field(default_factory=dict)
    last_observed_filesystem_state: dict[str, Any] = field(default_factory=dict)

@dataclass
class MutationRecord:
    timestamp: float
    mutation_type: str
    target: str
    verified: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskRequirement:
    requirement_id: str
    description: str
    satisfied: bool = False
    evidence: str = ""
    severity: str = "HIGH"

@dataclass
class Subgoal:
    subgoal_id: str
    description: str
    completed: bool = False
    next_commands: list[str] = field(default_factory=list)

@dataclass
class FailedCommandCache:
    failed_commands: dict[str, dict[str, Any]] = field(default_factory=dict)
    # keyed by normalized command fingerprint: {count, last_turn, exit_code, stderr_summary}

@dataclass
class MemoryStateBank:
    current_environment: EnvironmentState
    successful_mutations: list[MutationRecord]
    unmet_task_requirements: list[TaskRequirement]
    active_subgoals: list[Subgoal]
    failed_commands_cache: FailedCommandCache
    updated_at_turn: int = 0
    schema_version: str = "1.0.0"
```

---

## 5. Security Assumptions & Threat Model

1. **Untrusted Model Output:** The quantized action model's output cannot be trusted; it may generate prompt injections, attempt directory traversal, execute destructive bash commands, or probe environment variables.
2. **Strict Host Isolation:** The host process possesses root authority, real API credentials, and network routing. The container/sandbox tier must run unprivileged (`nobody` or `tribune-sandbox`), drop all capabilities, and deny host networking.
3. **Surrogate Token Masking:** Model environments only ever see synthetic tokens (`MOCK_*`). Real secrets are mapped in memory strictly on the host and never written to logs, artifacts, stdout/stderr, or model prompts.
4. **Deny-by-Default Egress:** Sandbox outbound traffic is blocked (`--network=none` or filtered). Outbound calls to allowlisted APIs must be forwarded through Sentinel, which injects authentic credentials host-side.
5. **Fail-Closed Policy:** Any unhandled exception, syntax failure, or timeout in Sentinel or Allowlist enforcement defaults to rejection (exit code 126/127).

---

## 6. Backward Compatibility & Rollout Strategy

- **`TRIBUNE_EXECUTION_MODE`:**
  - `container`: Production default. Requires Docker/container runtime.
  - `local_fallback`: Development default when Docker daemon is not active. Implements token surrogateing and allowlist enforcement; marked `unsafe_for_production=True`.
  - `legacy`: Preserves previous direct shell execution for baseline benchmark comparison.
- **`TRIBUNE_MEMORY_SIDECAR_ENABLED`:**
  - Default: `true`.
  - If set to `false`, `ladder.py` bypasses the sidecar and records zero auxiliary overhead, matching baseline behavior.
- **Seed Set & Verifier Compatibility:**
  - `seedset.py` keeps existing generation logic while enriching returned `SyntheticCase` with `.task_requirements` and `.task_metadata`. Existing tests continue passing without modification.

---

## 7. Testing Strategy

1. **Unit Tests:**
   - `test_sentinel_policy.py`: Allowlist allow/deny, UDS socket communication, host-side auth injection, audit log validation.
   - `test_token_broker.py`: Secret detection, surrogate token mapping, bi-directional scrubbing, stdout/stderr redaction.
   - `test_runners.py`: Container config generation, local fallback runner, timeout enforcement, exit code handling.
   - `test_memory_sidecar.py`: 5-track state bank operations, interval $k=2$, silence default, trigger conditions (repeated failed command, non-existent path, premature completion), cooldown and cap limits.
   - `test_dual_agent_costmodel.py`: Trajectory cost equation, indicator function, auxiliary cost calculation, Pareto frontier under dual agents.
2. **Integration & Smoke Tests:**
   - Full end-to-end evaluation run on synthetic test cases using mock auxiliary backend and local fallback runner.
   - Verify that `evidence.json` is updated with all verification milestones and metrics.
