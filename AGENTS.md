# AGENTS.md: Architectural Contracts, Invariant Protocols, & Operational Guidelines

This document specifies the technical architecture, symbolic ontology, runtime verification contracts, and agent working rules for the `Tribune` system. All AI agents, contributors, and downstream automated workflows must adhere strictly to the protocols defined herein.

---

## 1. World Model Invariants & Symbolic Ontology

The world model (`tribune.casegen.world_model`) enforces symbolic invariants across benefit programs (SNAP, Medicaid, Unemployment, Housing, Appeals) and courtroom evidence admission workflows.

### 1.1 Statutory Predicates & Bounds

1. **Gross Monthly Income Invariant:**
   For household size $H \ge 1$ and Federal Poverty Level parameters ($\text{base} = \$1,255.00/\text{mo}$, $\text{increment} = \$438.00/\text{person}$), the statutory cutoff is:
   $$\text{Limit}(H) = (\text{base} + (H - 1) \times \text{increment}) \times 1.30$$
   Any state transition asserting eligibility where $I_{\text{gross}} > \text{Limit}(H)$ incurs an immediate confidence penalty ($\Delta V_t \ge 0.40$), driving the transition score $V_t$ below the calibrated threshold.

2. **Temporal Appeal Window Invariant:**
   Statutory appeal postmark elapsed days $D$ must satisfy:
   $$0 \le D \le \tau_{\text{deadline}} \quad (\text{where } \tau_{\text{deadline}} = 90 \text{ days})$$
   Appeals with $D > 90$ (time-barred) or $D < 0$ (chronological impossibility) are hard invariant violations ($V_t \le 0.15$).

3. **Liquid Resource Invariant:**
   Total liquid assets $A$ must satisfy $0 \le A \le A_{\max}$ (where $A_{\max} = \$2,750.00$ for SNAP non-elderly households). Asset balances exceeding $A_{\max}$ or negative asset assertions trigger immediate breach flags.

4. **Latent Contradiction & Discovery Invariant:**
   Contradictions uncovered during legal discovery (e.g., self-reported $\$0$ gig earnings contradicted by unverified $\$850$ 1099 wages, or self-reported $\$500$ liquid assets contradicted by a $\$3,200$ bank audit) degrade transition confidence to $V_t \le 0.25$.

5. **Courtroom Evidentiary Procedural State Machine:**
   Exhibits in `CourtroomWorldModel` must follow strict sequential transitions:
   $$\text{UNMARKED} \xrightarrow{\text{mark\_exhibit}} \text{MARKED} \xrightarrow{\text{offer\_exhibit}} \text{OFFERED} \xrightarrow{\text{admit\_exhibit}} \{\text{ADMITTED} \mid \text{REJECTED}\}$$
   Attempting to admit an unoffered or unmarked exhibit breaches procedural rules ($V_t \le 0.10$).

---

## 2. Simulation Lifecycle & Conformal Risk Control (CRC) Protocol

`SimulationEngine` in `tribune.casegen.simulation` enforces inline runtime verification using finite-sample corrected Conformal Risk Control.

### 2.1 Calibration Formulae & Bounds

Given $n$ historical simulation traces under nominal compliant conditions:
1. **Finite-Sample Quantile Index:**
   $$\hat{q} = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}$$
   where $\alpha \in (0, 1)$ is the nominal false-alarm tolerance (default $\alpha = 0.10$).
2. **Statistical Padding Term:**
   $$\epsilon_n = \sqrt{\frac{\ln(2/\delta)}{2n}}$$
   where $\delta \in (0, 1)$ is the confidence tolerance (default $\delta = 0.05$ for $95\%$ confidence).
3. **Certified Cutoff Thresholds:**
   * **Conformity Scores ($V_t \in [0, 1]$, higher is valid):**
     $$\hat{\lambda} = V_{(\lfloor (n + 1)\alpha \rfloor)}, \quad \hat{\lambda}_{\text{padded}} = \max(0.0, \, \hat{\lambda} - \epsilon_n)$$
   * **Non-Conformity Scores ($S_t \in [0, 1]$, higher is error):**
     $$\hat{\lambda} = S_{(\lceil (n + 1)(1 - \alpha) \rceil)}, \quad \hat{\lambda}_{\text{padded}} = \min(1.0, \, \hat{\lambda} + \epsilon_n)$$
   * **Finite-Sample Guarantee:** $\mathbb{P}(\text{False Alarm}) \le \alpha + \epsilon_n$.

### 2.2 Turn Interception & State Rollback Protocol

At each turn $t \in \{1, \dots, T_{\max}\}$:
```
Turn Proposed (t)
       │
       ▼
Take Pre-Turn Snapshot ─────────► snapshot = state.snapshot()
       │
       ▼
Stage Turn Mutation ────────────► state.stage_mutation(turn)
       │
       ▼
Evaluate Transition Score ──────► Vt = world_model.score_transition(state, turn)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Vt >= lambda_hat_padded]      [Vt < lambda_hat_padded]
   (Invariant Valid)               (CRC Fault Breached)
       │                                 │
       ▼                                 ▼
 Commit Mutation:               Clean State Rollback:
 state.commit()                 state.rollback(snapshot)
 tokens_burned += 500                    │
 Proceed to t + 1                        ▼
                                Log Telemetry to UsageRecorder:
                                early_exit_step = t
                                tokens_saved = (T_max - t) * 500
                                status = "Interrupted (CRC Fault at Step t)"
                                         │
                                         ▼
                                Terminate Immediately
```

On invalid premises, this intercepts execution at Step 3 or 4, burning only $\sim 35\%$ of trajectory tokens and saving $\sim 65\%$.

---

## 3. Usage & Telemetry Schema

All model calls and simulation runs report telemetry through `tribune.instrumentation.usage` and `tribune.types`.

### 3.1 ModelCallUsage Schema
* `role: str`: `"proposer"` | `"verifier"` | `"simulation"` | `"general"`.
* `model: str`: Model identifier (e.g., `"glm-5.3-flash"`, `"qwen3.8-27b"`).
* `tokenizer_id: str`: Tokenizer identifier.
* `tokens_input: int`: Prompt token count.
* `tokens_output: int`: Generation token count.
* `cache_read_tokens: int`: Tokens served from prompt/KV cache.
* `cache_write_tokens: int`: Tokens written to prompt cache.
* `estimated: bool`: Whether tokens were deterministically estimated.
* `active_experts: Optional[int]`: Active expert count for MoE backends (e.g., `8` for GLM-5.3-Flash).
* `timestamp: str`: High-precision ISO-8601 UTC timestamp with microseconds (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`).

### 3.2 TaskUsage Schema
* `cache_hit_ratio: float`: Ratio of cache read tokens to total prompt tokens ($[0.0, 1.0]$).
* `active_experts: Optional[int]`: Active experts utilized in execution trace.
* `early_exit_step: Optional[int]`: Step index where simulation was aborted; `None` if completed.
* `tokens_saved_estimate: int`: Spared tokens projected from aborted trajectory turns.
* `crc_breach_events: list[dict[str, Any]]`: Detailed log of breach steps, fault scores, thresholds, and actions.
* `cost_usd: Optional[float]`: Total priced cost calculated against `CostModel`.

---

## 4. Tiered Routing Architecture

Model invocation is abstracted behind `TieredRoutingGateway` (`tribune.clients.routing`):

1. **Bulk Scenario Generation (`RoutingTier.BULK_GENERATION`):**
   * Default backend: `glm-5.3-flash` (open MoE architecture).
   * Configuration: 1,048,576 token context window, 8 active experts per token, cost rates $\$0.15/\text{M}$ input, $\$0.50/\text{M}$ output.
   * Purpose: High-throughput generation of bulk variations, narrative permutations, and counterfactual constraints.
2. **Statutory Invariant Verification (`RoutingTier.STATUTORY_VERIFICATION`):**
   * Default backend: `qwen3.8-27b` / fine-tuned legal process verifier.
   * Configuration: 131,072 token context window, dense reasoning, cost rates $\$0.10/\text{M}$ input, $\$0.20/\text{M}$ output.
   * Purpose: Deterministic citation verification, symbolic ontology checks, and step-level advantage calculation.
3. **Frontier Escalation (`RoutingTier.FRONTIER_REASONING`):**
   * Default backend: `gemini-3.7-flash` / `gpt-5.6-sol-ultrafast`.
   * Purpose: Unresolvable statutory ambiguities, novel appellate claims, and governance escalation.

---

## 5. Agent Working Rules & Compliance Boundaries

Downstream agents and developers working on `Tribune` must observe the following constraints:

1. **Never Bypass Runtime CRC Verification:**
   All multi-turn simulation trajectories must be executed via `SimulationEngine.run_trajectory()`. Direct programmatic mutations of `SimulationState` without passing through `stage_mutation -> score_transition -> commit` are strictly forbidden.

2. **Atomic Rollback Inviolability:**
   Whenever a CRC fault is triggered, the engine must revert all uncommitted mutations immediately. No partial, corrupt, or unverified state variables may persist into subsequent turns.

3. **No Unjustified Mocks:**
   Process verifiers and statutory world models must not be replaced with permissive dummy mocks in integration tests or production paths. Fuzzing suites must assert that invalid scenarios are rejected.

4. **Telemetry Integrity:**
   All early-termination events must log `early_exit_step`, `tokens_saved_estimate`, and ISO-8601 timestamps to `UsageRecorder`. Modifying usage counters manually to mask token expenditure is prohibited.

5. **Interface Stability:**
   Preserve existing courtroom methods (`CourtroomWorldModel.mark_exhibit`, `offer_exhibit`, etc.) and streaming client interfaces (`FalMiniMaxStreamingClient`) to maintain backwards compatibility across the test suite.
