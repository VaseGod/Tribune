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
5. If the assessment clears verification and confidence, the **preparer** assembles materials and a document checklist — but the **action-gate** ensures nothing binding is submitted.
6. Every step writes a record to an append-only, **hash-chained audit log**; `disclosure.py` renders it as a plain-language "here's why."

---

## Swapping in real backends (run it sovereignly)

Everything external sits behind an interface with a working local fallback. Flip one environment variable to use a real backend.

**Self-hosted model (vLLM / SGLang / any OpenAI-compatible endpoint):**
```bash
export TRIBUNE_PROVIDER=openai_compat
export TRIBUNE_OPENAI_BASE_URL=http://localhost:8000/v1
export TRIBUNE_OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
# the verifier can run on a stronger model:
export TRIBUNE_VERIFIER_PROVIDER=openai_compat
export TRIBUNE_VERIFIER_MODEL=Qwen/Qwen2.5-32B-Instruct
```

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
  runs the verifier across a quantization ladder (llama.cpp GGUF / vLLM formats,
  plus a free deterministic mock for CI) over a frozen, hash-pinned 50-case seed
  set and generates
  [Eval Note #1: *Does abstention calibration survive quantization?*](docs/eval_notes/eval_note_1_quant_sensitivity.md)

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
