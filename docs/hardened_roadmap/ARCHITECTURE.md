# Hardened Memory & Execution Architecture

Implements the P0/P1/P2 hardening roadmap: Protocol-Aware Retention,
cryptographic consolidation provenance, sliding intent-graph analysis,
Agentic MapReduce sweeps, and reasoning-budget/speculative-embedding controls.

## 1. Protocol-Aware Retention (`tribune/memory/timeline.py`, `retention.py`, `retrieval.py`)

Every timeline event is split at append time:

- **StateDelta** — compact, durable, structured state change (`event_id`,
  `parent_event_id`, `turn_id`, `session_id`, `timestamp`, `actor`,
  `tool_name`, `operation_type`, `entity_ids`, `file_changes`, `exit_code`,
  `command_status`, `result_summary`, `causal_refs`, `observation_digest`,
  `schema_version`). Kept in the active context.
- **EphemeralObservation** — noisy raw payload (`stdout`, `stderr`, raw JSON,
  sizes, content hash, cold URI). Visible during the active turn only.

At `end_turn()`, ephemeral payloads are flushed from the prompt window into a
content-addressed append-only cold store (`FileColdStorage`, JSONL, 0600 file
permissions; in-memory fallback for tests). Wrap with `EncryptedColdStorage`
(Fernet, `pip install tribune[security]`, key via `TRIBUNE_RETENTION_COLD_ENCRYPT_KEY`
or its `_FILE`/`_CMD` secret refs) for encryption at rest — requested
encryption with a missing key or package fails closed, never silently
plaintext. Active context keeps StateDeltas plus compact provenance
references (`observation_digest` + `cold_uri`).

Token accounting is pluggable (`tribune/memory/tokens.py`): the default char
heuristic matches repo convention; `TRIBUNE_TOKEN_COUNTER=tiktoken` (needs
`pip install tribune[tokenizers]`) switches to real tokenizer counts.

**Tool-loop mitigation** (`ToolLoopDetector`): exact repeats, no-new-delta
repeats, and A/B oscillation patterns emit structured warnings into the
timeline; optional pre-dispatch blocking via
`TRIBUNE_RETENTION_BLOCK_LOOPS=true`. Measured: ~91% context reduction on
noisy tool traffic; stable across simulated 100k-token sessions.

**Retrieval** (`RetentionAwareRetriever`): default associative search indexes
StateDelta nodes only. Raw observations are refused at index time (logged) and
retrievable only via explicit provenance lookup
(`fetch_cold_observation(digest)`) or privileged debug query. Ranking favors
verified provenance, recency, and causal relevance.

## 2. Consolidation schema IR + HMAC provenance

- `tribune/memory/consolidation_schema.py`: strictly typed
  `ConsolidationIR` (`schema_version`, `source_episodic_ids`, entities with
  `ACTIVE|DEPRECATED|CONTRADICTED` status, mutations with `confidence ∈ [0,1]`,
  declarative summary ≤500 chars, UTC `generated_at`). `ConsolidationSanitizer`
  rejects `SYSTEM:`/`OVERRIDE`/`IGNORE PREVIOUS`/`YOU MUST`-style markers,
  second-person imperatives, over-long fields (after NFKC normalization);
  rejections are logged and quarantined, never entering HDM centroids.
- `tribune/security/provenance.py` + `audit.py`: `HMAC-SHA256` over
  `canonical(source_ids) || canonical(payload) || timestamp`; keys resolve via
  `tribune/security/secrets.py` chain — direct env, `_FILE` ref (mode-checked),
  or `_CMD` ref (no shell, timeout) — for both the active key
  (`TRIBUNE_HMAC_SECRET*`) and rotations (`TRIBUNE_HMAC_SECRET_<KEYID>*`); append-only hash-chained audit log
  (`entry_digest = hash(prev || type || payload_digest || sig || ts)`).
- `tribune/memory/hdm.py::ProvenanceGatedMemory`: unsigned/tampered nodes are
  blocked, logged, quarantined; verification enforced on store, activation,
  bundling, and decay.

## 3. Sliding Intent Graph Analyzer (`tribune/security/secure_forge.py`)

Stateful per-session rolling window (default W=8):
`H_window = Σ ω_t · φ(ToolCall_t)`, `ω_t = exp(-λ·age)`, `φ` = VSA
bundle of normalized tool name + extracted entities (paths, symbols,
endpoints, packages, files) + params. Cosine similarity against 10
policy-derived exploit-topology exemplars (deserialization gadgets, credential
staging, secret harvesting, privesc prep, sandbox escape, exfiltration,
dependency confusion, destructive FS sequencing, recon→exploit, combined
benign-look subtasks). Scoring takes the max over the full window plus the
last-3 and last-1 suffixes, so focused sub-sequences are not diluted inside
longer mixed sessions. On `similarity ≥ tau_threat` (default 0.45): session
suspended, dispatch blocked (fail closed), trajectory logged, forensic
artifact preserved (`admin_review_artifact`), reinstatement only via logged
`admin_override`. Benign fixtures (grep, git diff, pytest, refactor, lint,
docs, dep inspection) + allowlist keep false positives low (0.0 on the local
fixture suite).

**Operating point (measured, `scripts/tune_intent_threshold.py`):** exploit
chains peak at 0.51–0.65 similarity while benign fixtures sit at 0.05–0.13,
so tau=0.45 catches all fixture chains with wide benign headroom. The tuner
sweeps 0.30–0.95 and recommends the most conservative feasible tau; re-run it
with production-derived fixtures (`--benign/--exploits/--threats`) before
changing prod tau. The library grows without code changes via
`TRIBUNE_INTENT_THREAT_LIBRARY` (JSON) and `add_exemplar_from_trajectory()`,
which distills a suspended session's top tokens into a new exemplar.

## 4. Agentic MapReduce Harness (`tribune/memory/episodic.py`)

Replaces recursive graph walks for large sweeps: **Plan** (deterministic
`ASTSelector` filtering + stable-hash disjoint sharding, no model inference),
**Map** (thread-pool isolated workers, bounded per-worker context, no shared
mutable state, explicit ok/failed/skipped per shard, deterministic ordering),
**Reduce** (bitwise majority-rule bundling over shard signatures, preserved
contradiction + coverage metadata, deterministic bundle). Coverage invariant:
`processed + failed + skipped == total`; gaps raise unless
`tolerate_missing_coverage`. Measured: coverage 1.0, ~linear scaling
(4× candidates ≈ 2.3× time), deterministic across runs.

## 5. Reasoning budgets + speculative drafting (`tribune/memory/hdm.py`)

- `ReasoningBudgetTracker`: caps trace tokens / vector counts / per-turn and
  per-session spend; evicts unverified low-utility stale traces first; prefers
  StateDelta-backed vectors; exposes exhaustion telemetry.
- `SpeculativeEmbeddingCache`: local-provider `embed()` hook, precompute of
  likely next queries, LRU-bounded cache; speculative vectors never enter
  signed memory and are discardable. Measured: ~99% per-query latency cut on
  cache hits; reasoning memory bounded by budget.

## Configuration

See `CONFIGURATION.md` and `.env.example`. All subsystems have safe defaults,
kill switches (`*_ENABLED=false`), and Pydantic settings in `tribune/config.py`.

## Verification

- `tests/test_hardened_roadmap.py` (10 tests: normal + adversarial + regression)
- `scripts/bench_hardened_roadmap.py` (measured evidence for every target metric)
- `scripts/backfill_retention.py` (dry-run legacy migration)
