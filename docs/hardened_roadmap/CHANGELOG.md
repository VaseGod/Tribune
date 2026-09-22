# Changelog — Hardened Roadmap

## Follow-up fixes (verification-report risks, now closed in code)

- `tribune/memory/tokens.py`: pluggable token counting (`char` default,
  `tiktoken` backend via `TRIBUNE_TOKEN_COUNTER`); `retention.estimate_tokens`
  delegates to it. New: `tokenizers` extra in `pyproject.toml`.
- `tribune/security/secrets.py`: env → file → command secret-resolution chain
  (no shell, timeout, file-mode warnings, never logs values); wired into
  `HMACKeyRing.from_env` for active and rotated keys.
- `tribune/memory/retention.py`: `FileColdStorage` hardened to `0600`;
  `EncryptedColdStorage` (Fernet, fails closed without `tribune[security]`);
  `build_cold_storage()` auto-wraps when a key is configured; `timeline.py`
  uses it. New: `security` extra in `pyproject.toml`.
- `tribune/security/secure_forge.py`: suffix-max scoring (full / last-3 /
  last-1 windows) against threat exemplars; file-backed library
  (`TRIBUNE_INTENT_THREAT_LIBRARY`, `load_threat_library`); incident learning
  via `register_exemplar` / `add_exemplar_from_trajectory`; default
  `tau_threat` recalibrated 0.75 → **0.45** from tuner evidence.
- `scripts/tune_intent_threshold.py`: tau sweep harness recommending the most
  conservative feasible operating point; accepts prod `--benign/--exploits/--threats`.
- `tribune/config.py`: `token_counter_backend`, `intent_threat_library` settings.
- 7 new tests (17 total in `test_hardened_roadmap.py`); full suite 592 passed.

## Added (P0 — Context Pipeline)

- `tribune/memory/retention.py`: `StateDelta`, `EphemeralObservation`,
  `RetentionPolicy`, `FileColdStorage`/`InMemoryColdStorage`,
  `ToolLoopDetector`, `RetentionMetrics`, token estimation.
- `tribune/memory/timeline.py`: retention-aware `MemoryEventsTimeline`
  (split on append, `begin_turn`/`end_turn` flush, `active_context()`,
  `check_tool_call` pre-dispatch gate, loop warnings, metrics).
- `tribune/memory/retrieval.py`: `RetentionAwareRetriever` (StateDelta-only
  default search, privileged raw access, provenance/recency/causal ranking).

## Added (P0 — Memory Integrity)

- `tribune/memory/consolidation_schema.py`: schema IR + adversarial sanitizer
  with quarantine.
- `tribune/memory/consolidation.py`: `SecureConsolidator` (validate → sign → envelope).
- `tribune/security/provenance.py`: HMAC-SHA256 keyring, signing/verification,
  hash-chained `ProvenanceAuditLog`.
- `tribune/security/audit.py`: `CONSOLIDATION_SIGNED/REJECTED`,
  `HMAC_VERIFICATION_FAILURE`, `UNSIGNED_VECTOR_BLOCKED`,
  `INTENT_GRAPH_ALERT`, `SESSION_SUSPENDED`, `TOOL_LOOP_DETECTED` events;
  `sign_consolidated_node` / `verify_consolidated_node`.
- `tribune/memory/hdm.py`: `ProvenanceGatedMemory` (fail-closed activation).

## Added (P1 — Security Sandbox)

- `tribune/security/secure_forge.py`: `SlidingIntentGraphAnalyzer` with
  `H_window` drift detection, 10 threat exemplars, benign fixture suite,
  suspension + forensic artifacts + logged admin override.

## Added (P1 — Search Harness)

- `tribune/memory/episodic.py`: `AgenticMapReduceHarness` (plan/map/reduce),
  `ASTSelector`, `MapReduceConfig`, coverage guarantees.

## Added (P2 — Hyperdimensional Store)

- `tribune/memory/hdm.py`: `ReasoningBudgetTracker`, `SpeculativeEmbeddingCache`,
  `majority_rule_bundle`.

## Added (config/tests/benchmarks/docs)

- `tribune/config.py`: 17 new hardened settings with safe defaults.
- `.env.example`: hardened section.
- `tests/test_hardened_roadmap.py`: 10 tests (normal/adversarial/regression).
- `scripts/bench_hardened_roadmap.py`, `scripts/backfill_retention.py`.
- `docs/hardened_roadmap/`: ARCHITECTURE, SECURITY, CONFIGURATION,
  MIGRATION, CHANGELOG, VERIFICATION_REPORT.

## Assumptions

- Token counts use the repo-standard ~4-chars-per-token estimator (consistent
  with `ContextPilot`); production prefill accounting should use real tokenizer
  counts — the harness exposes before/after hooks for that swap.
- Local HD vectors use seeded random projections (numpy, already a core dep);
  no new dependencies added.
- Exact production thresholds (tau, budgets) require production traffic to
  finalize; instrumentation for tuning ships with the code.
