# TRIBUNE

**TRIBUNE is a verification-first, self-hostable AI assistant that helps people access the public benefits and essential services they are legally entitled to — food assistance (SNAP), unemployment insurance, Medicaid/health coverage, housing assistance, and the appeals attached to them.**
It is safe-by-construction: every eligibility claim is cited to the governing rule, it abstains and routes to a human whenever a case is genuinely uncertain, and it never issues a binding determination or auto-submits anything.

> Billions in benefits people are legally owed go unclaimed every year because the systems are too complex to navigate. A tool that helps people understand what they qualify for has real welfare impact — but a tool that *confidently hallucinates* eligibility is actively harmful. TRIBUNE's verification, abstention, provenance, and audit machinery exist precisely so an AI can be put near someone's access to food, healthcare, or housing without that risk.

TRIBUNE is the successor to ARCHON, a proposer/verifier verification stack built for corporate financial back-office work. TRIBUNE reuses that machinery and points it the other way: at helping ordinary people.

---

## The safety stance (hard rules, enforced in code and tested)

1. **Never a binding determination.** TRIBUNE surfaces what the rules say and what a situation suggests — always informational, always subject to review by the person, a navigator, or the agency. It is not legal advice and says so.
2. **Calibrated abstention is mandatory.** On any close, ambiguous, or under-evidenced question, TRIBUNE abstains and escalates to a human. Being confidently wrong is the cardinal harm; abstaining when uncertain is the *rewarded* outcome.
3. **Every substantive claim is cited.** An eligibility assertion without a verifiable citation to the governing rule is *impossible to construct* (enforced by a pydantic validator on `Assessment`).
4. **Action-gating on anything binding.** TRIBUNE prepares applications, appeals, and checklists, but submission requires an explicit human sign-off routed through the action-gate.
5. **A human path is always offered**, and TRIBUNE never discourages seeking official or in-person help.
6. **Privacy is absolute.** PII never needs to leave the deploying organization. TRIBUNE is self-hostable and the model backend is swappable, so a clinic, library, legal-aid office, or community navigator can run it sovereignly.

See [LIMITATIONS.md](LIMITATIONS.md) for scope and known failure modes.

---

## Quickstart

Requires Python 3.10+. No API keys, no GPU — the defaults run entirely offline.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

tribune demo       # walk full synthetic cases end-to-end (prepare, replan, and abstentions)
tribune eval       # run the evaluation harness and print the metrics table
tribune canary     # adversarial canary + rule-citation-drift sentinel
tribune disclose   # plain-language "here's why" explanation of the last demo run
tribune web        # serve the web UI at http://127.0.0.1:8000
```

`tribune demo` and `tribune eval` run clean and exit 0 with the default local providers.

### The web UI

`tribune web` serves a single-page app (FastAPI backend in `tribune/server.py`, static frontend in `tribune/static/`) built for benefits applicants, caseworkers, and community navigators:

- **Demo cases** — run the preset simulations and watch prepare / replan / abstain outcomes.
- **New Case wizard** — a stepper intake form (household, income, status, programs) plus a simulated intake-document box (`key: value` lines become provenance-tagged evidence).
- **Assessment results** — per-program status cards with confidence gauges, an expandable rules accordion where every criterion links to its governing citation (e.g. 7 CFR 273.9), and honest housing waitlist notes (*eligibility ≠ access*).
- **Preparer & action gate** — document checklists and drafted fields for eligible programs; the "Review & authorize submission" flow requires a named human sign-off and returns a signed `SubmissionReceipt`. Nothing is ever auto-submitted.
- **Navigator chat** — a grounded assistant that explains *why* a decision was made and what is missing. It only reports what the run actually found; it never invents eligibility facts.
- **Caseworker audit view** — a toggle that reveals the hash-chained audit log (PLAN → GATHER → ASSESS → VERIFY → …) with record hashes.
- **Settings** — switch model backend (local rules engine vs. a self-hosted vLLM/SGLang endpoint), jurisdiction, and the abstention threshold.

The web layer needs the `web` extra (`pip install -e ".[web]"`, included in `[dev]`).

### What the demo shows

```
demo-parent-multi     snap          likely_eligible -> prepared (not submitted)
demo-parent-multi     medicaid      likely_eligible -> prepared (not submitted) (replans=1)
demo-parent-multi     housing       ABSTAIN -> human (replans=1)        # eligible, but waitlist unknown: eligibility != access
demo-quit-good-cause  unemployment  ABSTAIN -> human                    # a "good cause" quit is fact-intensive
demo-coverage-gap     medicaid      ABSTAIN -> human (replans=1)        # non-expansion coverage gap
demo-appeal           appeals       likely_eligible -> prepared (not submitted)
demo-snap-ineligible  snap          likely_ineligible -> recommend abstain_and_escalate
```

---

## Architecture

```
tribune/
  config.py              pydantic-settings; selects providers/stores/ingest by env
  types.py               ALL shared strict typed models (the single source of truth)
  providers/             ModelProvider protocol: local_rules (default) + openai_compat (vLLM/SGLang)
  orchestration/         manager/subagent DAG, the PLAN->GATHER->ASSESS->VERIFY->{PREPARE|ABSTAIN|REPLAN}
                         state machine, complexity router, and the CasePipeline
  agents/                navigator (manager), eligibility & preparer (proposers), verifier
  abstention/            calibrated confidence -> assert / abstain-and-escalate
  memory/                typed, per-case, access-controlled partitions + consolidation/lifecycle
  corpus/                RuleStore (local + hosted), typed citations, ColBERT-class late-interaction
                         retrieval, anonymization + provenance, and the program rule sets
  governance/            never-auto-submit action gate, hash-chained audit log, plain-language disclosure
  ingestion/             DocIngest interface: structured (default) + OCR adapter
  casegen/               privacy-preserving synthetic case generator + per-program ground-truth labelers
  eval/                  task x harness x verifier matrix, metrics, canary/drift sentinel
  instrumentation/       W&B Weave hooks with a no-op fallback
  server.py              FastAPI backend for the web UI (wraps CasePipeline + ActionGate)
  static/                the single-page web app (index.html, styles.css, app.js)
  cli.py                 tribune demo | eval | canary | disclose | web
```

### The core loop

For each program in a person's situation:

1. **Navigator** decomposes the situation into a sub-task DAG and routes by complexity.
2. The **eligibility proposer** retrieves the governing rules (late-interaction retrieval), evaluates each rule's predicate against the documented evidence, and synthesizes a typed, fully-cited `Assessment` (status + per-criterion results + recommended action + self-reported confidence).
3. The **verifier** independently re-derives the assessment from the *cited* rules and checks citation integrity, rule **coverage**, and support. It can reject → **REPLAN** (the router widens retrieval and retries — a real retrieval-recall recovery).
4. **Calibrated abstention**: if calibrated confidence is below threshold — or the case hits a structural ambiguity signal, or verification failed, or the status is indeterminate — TRIBUNE **abstains and routes to a human**. Abstention is logged as a first-class, correct-when-uncertain outcome.
5. If the assessment clears verification and confidence, the **preparer** assembles materials and a document checklist — but the **action-gate** enforces mandatory epistemic citation verification, ensuring no uncited claims or invalid statutory references can be submitted.
6. Every step writes a record to an append-only, **hash-chained audit log**; `disclosure.py` renders it as a plain-language "here's why."

### Core Architectural Pillars

TRIBUNE is structured around 4 failure-resilient, governance-first architectural pillars:

1. **Subagent Worktree Isolation (`tribune/memory/`, `tribune/orchestration/`):**
   - Independent subagents run inside isolated `SubagentMemoryPartition` boundaries keyed by `(case_id, subagent_id, program)` with cross-subagent read protection (`AccessDenied`).
   - Supports branching (`fork()`) and safe reconciliation (`merge_into()`), while `AsyncDAGRunner` schedules independent tasks into parallel topological waves.
2. **Two-Stage Evaluator-Generator Pattern (`tribune/agents/`, `tribune/governance/`):**
   - **Pass 1 (Milestone Verification):** `Verifier.evaluate_and_certify` performs independent statutory re-derivation, arithmetic checks, citation validity, and rule coverage checks to yield a certified `VerificationReport`.
   - **Gating & Pass 2 (Notice Generation):** `ActionGate` enforces verification certification before `generate_determination_notice` can draft formal legal disclosure notices with statutory basis and fair hearing appeal rights.
3. **Atomic Transaction Logging & Crash Recovery (`tribune/governance/`, `tribune/orchestration/`):**
   - Step-level milestones are written atomically via `CheckpointManager` using temporary files and `os.replace` to prevent corrupted partial writes on unexpected process termination.
   - On startup, `Router.inspect_checkpoint` detects interrupted cases and generates a `RecoveryPlan`; `CasePipeline.resume_from_checkpoint` restores memory partitions, replays audit trails, and executes only remaining pending tasks without duplicate computation.
4. **Quantization Parity & Pareto Frontier Benchmarking (`tribune/eval/`):**
   - Evaluation ladders track statutory citation precision/recall, false positive rate (FPR), false negative rate (FNR), and accuracy delta vs. full-precision reference backends.
   - `CostModel.compute_pareto_frontier` runs multi-objective optimization (minimizing cost vs. maximizing accuracy/parity) across local quant tiers and cloud endpoints.

---

## Swapping in real backends (run it sovereignly)

Everything external sits behind an interface with a working local fallback. Flip one environment variable or backend profile to run sovereignly.

**Local Sovereign Endpoint (Meta's Muse Glimmer 30B GGUF with Speculative Decoding):**
Register local model backends in [backends/registry.yaml](backends/registry.yaml):
```yaml
muse_glimmer_local:
  provider: openai_compat
  base_url: "http://localhost:8080/v1"
  model_name: "Muse-Glimmer-30B-GGUF"
  parameters:
    temperature: 0.1
    max_tokens: 4096
    extra_body:
      speculative_drafter: "DFlash"
      sliding_window_attention: true
      kv_cache_type: "q8_0"
```
The `openai_compat` provider forwards `extra_body` parameters (such as `speculative_drafter`, `sliding_window_attention`, and `kv_cache_type`) directly in completion requests to local servers (vLLM / llama.cpp).

**Self-hosted model (vLLM / SGLang / any OpenAI-compatible endpoint):**
```bash
export TRIBUNE_PROVIDER=openai_compat
export TRIBUNE_OPENAI_BASE_URL=http://localhost:8000/v1
export TRIBUNE_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
# the verifier can run on a stronger model:
export TRIBUNE_VERIFIER_PROVIDER=openai_compat
export TRIBUNE_VERIFIER_MODEL=Qwen/Qwen2.5-32B-Instruct
```

**Low-Latency Document Ingestion Fast Path:**
Document intake for wage stubs, lease agreements, and decision notices executes a high-speed heuristic regex/layout parser (`fast_heuristic_parse`) before falling back to heavier OCR or VLM processing, minimizing intake latency and token consumption.

**Programmatic Python Agent Stubs:**
Agent tool interfaces (`tribune/agents/`) use strongly-typed Python executable class stubs (`ProgrammaticEligibilityTools`, `ProgrammaticPreparerTools`, `ProgrammaticNavigatorTools`, `ProgrammaticVerifierTools`) with explicit docstrings and type hints exposed directly to model prompts.

**Epistemic Citation Verification Gates:**
`ActionGate` and `Verifier` enforce a mandatory citation gate for local model outputs. Every eligibility determination and legal claim must match an active entry in `rule_store.py`. Outputs with uncited claims or invalid statutory references are intercepted, blocked via `ActionBlocked`, and flagged for human review.

**Hosted vector store** for retrieval:
```bash
export TRIBUNE_RULE_STORE=hosted
export TRIBUNE_HOSTED_VECTOR_URL=https://your-vector-service
```

**OCR document intake** (self-hosted, e.g. an Unlimited-OCR server):
```bash
export TRIBUNE_DOC_INGEST=ocr
export TRIBUNE_OCR_ENDPOINT=http://localhost:9000
```

**Tracing** (optional): `export TRIBUNE_TRACING=weave` (silent no-op if `weave` isn't installed).

Other knobs: `TRIBUNE_ABSTENTION_THRESHOLD` (default `0.70`), `TRIBUNE_DEFAULT_JURISDICTION`, `TRIBUNE_SEED`.

---

## Eval suites

Beyond `tribune eval` and `tribune canary`:

- **Quantization sensitivity** (`tribune quant-eval`, [docs](docs/quant_sensitivity.md)) —
  runs the verifier across a quantization ladder featuring 4-bit and 5-bit GGUF targets (`Q4_K_XL`, `Q5_K_M`, `Q8_0`, `Q4_K_M`, `Q2_K`, `IQ1_S`) evaluated across all standard benefit datasets (SNAP, Medicaid, Housing, Unemployment, Appeals) over a frozen 50-case seed set.
- **Multilingual parity** (`tribune parity`, [docs](docs/multilingual_parity.md)) —
  EN/ES twins of the seed set through the full pipeline; deltas beyond
  configurable thresholds are filed as **equity bugs** (distinct tracker label),
  and machine-drafted legal terms are flagged for human review, never silently
  deployed.
- **Ingestion injection probe** (`tribune redteam`, [docs](docs/injection_probe.md)) —
  adversarial benefit notices through the real OCR path; asserts the action gate
  never fires from document instructions, provenance never cites injected text,
  and injected content is flagged or inert.
- **Backend registry** ([backends/registry.yaml](backends/registry.yaml),
  [criteria](backends/README.md)) — a validated catalog of proposer/verifier
  candidates (license, real vs. announced-only weights, serving support, quant
  formats, and empty measured-kappa / cost-per-task slots filled from eval runs);
  `python scripts/validate_registry.py` checks it in CI.
- **Sandboxed appeals eval** (`make sandbox-appeals-eval`, [docs](docs/sandbox_eval.md)) —
  the appeals workflow end-to-end in a network-isolated container with pinned
  deps and deterministic seeds; a **deny-by-default egress guard** (allowlist =
  the configured model endpoint only) proves no eval run touches a live system.

---

## Model Context Protocol (MCP) Server Endpoint

TRIBUNE exposes a standard, stateless **MCP server interface** (`POST /mcp`) allowing external agentic environments (e.g. Cursor, Claude Desktop, ChatGPT Agent Plugins) to natively inspect resources and call tools:

- **JSON-RPC 2.0 Interface:** Endpoint at `/mcp` handling `initialize`, `resources/list`, `resources/read`, `tools/list`, and `tools/call`.
- **Exposed Resources:**
  - `tribune://programs` — Supported public benefit programs (SNAP, Medicaid, Unemployment, Housing, Appeals).
  - `tribune://jurisdictions` — Known jurisdiction codes (e.g. EX, CA, NY, TX).
  - `tribune://demo-cases` — Synthetic benchmark cases.
  - `tribune://rules/{program}` — Governing rulesets and criteria.
  - `tribune://cases/{case_id}` — Outcomes for an active case run.
- **Exported Tools:**
  - `tribune_run_case` — Run eligibility assessment for applicant parameters or document uploads.
  - `tribune_search_rules` — Search program rules, criteria, and legal citations.
  - `tribune_extract_fields` — Extract structured situation fields from raw text via OCR.
  - `tribune_explain_assessment` — Grounded explanation of case results.
- **Authentication & RBAC:** Enforces Bearer token / API key authentication when `TRIBUNE_MCP_AUTH_TOKEN` is set, and validates `X-Tribune-Role` headers for role-based permissions (e.g., `read_only` restrictions).
- **OpenAI Agent Plugin Compatibility:** Exposes `/.well-known/ai-plugin.json` (plugin manifest) and `/api/openai-tools` (OpenAI Chat Completions compatible tool schemas).

---

## Tiered Model Routing Architecture

TRIBUNE includes a central **`ModelRouter`** service layer that sits between system requests and LLM provider execution to optimize latency, cost, and reasoning quality:

- **Tier 1 (Fast / Cheap / Utility):** Lightweight models (defaults: `GPT-5.6 Luna` / `Qwen/Qwen2.5-7B-Instruct`) for routine tasks such as parsing, classification, formatting, and query generation.
- **Tier 2 (Deep Reasoning / Complex Synthesis):** Frontier models (defaults: `GPT-5.6 Sol` / `Qwen/Qwen2.5-32B-Instruct` / `Claude Opus` class) for multi-step reasoning, verifier checks, and complex synthesis.
- **Task Classifier Heuristic:** Automatically determines task tier based on intent, context token length (>4,000 tokens -> Tier 2), role (`verifier` -> Tier 2), and program complexity (Medicaid/Housing -> Tier 2).
- **Automatic Fallback:** If a Tier 1 model fails with an exception, outputs low confidence (`< 0.50`), or produces unparseable JSON/tool calls, `ModelRouter` automatically retries execution on Tier 2.
- **Configuration:** Set environment variables to customize model assignments:
  ```bash
  export DEFAULT_TIER1_MODEL="GPT-5.6 Luna"
  export DEFAULT_TIER2_MODEL="GPT-5.6 Sol"
  export TRIBUNE_PROVIDER="router"
  ```

---

## Cost accounting: cost per completed verification

TRIBUNE meters every task — tokens in/out, proposer/verifier turns, and a cost
computed from a configurable pricing file where **pricing is data, not code**
(API rates or amortized self-hosted $/GPU-hour, with effective dates so
promotional rates expire on schedule). `tribune eval` prints the breakdown by
outcome type; a **correct abstention is a completed success at its actual (low)
cost**, never a failure. See [docs/cost_accounting.md](docs/cost_accounting.md).

---

## Contributing a new program or jurisdiction

This is the **primary contribution path** and requires **no changes to core code**. See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

1. Add `tribune/corpus/programs/<program>.py` with a `RULESET` — one `Rule` per criterion, each carrying its **citation** and a pure **predicate**. Register it in `corpus/programs/__init__.py`.
2. Add `tribune/casegen/programs/<program>.py` with a `ground_truth(...)` labeler and the program's relevant evidence. Register it in `casegen/programs/__init__.py`.
3. New jurisdictions are added by parameters in `corpus/programs/jurisdictions.py`.

The **citation standard**: every rule claim must carry a verifiable source. The review bar for eligibility logic is high — see CONTRIBUTING.

---

## Why MIT vs. Apache-2.0

TRIBUNE ships under **Apache-2.0** ([LICENSE](LICENSE)): permissive *with an explicit patent grant*, which is the safest choice for broad adoption by under-resourced organizations and downstream builders. MIT would also work and is marginally simpler, but lacks the patent grant; for a tool meant to be deployed widely by clinics, libraries, and legal-aid offices, the patent grant is worth the slightly longer license text.

The value of TRIBUNE is **trust and public good, not a private moat**. The trace/eval corpus — correctly-cited eligibility logic for specific programs and jurisdictions — is meant to be open and community-maintained.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
