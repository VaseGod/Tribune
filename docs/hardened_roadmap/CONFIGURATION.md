# Configuration & Operations — Hardened Roadmap

All settings live in `tribune/config.py` (`TribuneSettings`, `TRIBUNE_` prefix)
with `.env.example` examples. Safe defaults everywhere; every subsystem has a
kill switch.

## Retention (P0)

| Env | Default | Meaning |
|---|---|---|
| `TRIBUNE_RETENTION_ENABLED` | `true` | Master switch for StateDelta/Ephemeral split |
| `TRIBUNE_RETENTION_MAX_TOKENS` | `32000` | Active context token budget (oldest actives evicted) |
| `TRIBUNE_RETENTION_MAX_EVENTS` | `200` | Max active events in window |
| `TRIBUNE_RETENTION_EPHEMERAL_TTL_S` | `3600` | Ephemeral TTL (bookkeeping; flush happens at turn end) |
| `TRIBUNE_RETENTION_COLD_PATH` | `.tribune/cold_observations.jsonl` | Cold store backend path |
| `TRIBUNE_RETENTION_DEBUG_MODE` | `false` | Emergency debug: keep raw payloads in memory |
| `TRIBUNE_RETENTION_BLOCK_LOOPS` | `false` | Block (vs. warn) on repeated tool calls |
| `TRIBUNE_TOKEN_COUNTER` | `char` | `char` heuristic or `tiktoken` (needs `tribune[tokenizers]`) |
| `TRIBUNE_TOKEN_COUNTER_MODEL` | `cl100k_base` | Encoding for the tiktoken backend |
| `TRIBUNE_RETENTION_COLD_ENCRYPT_KEY` (+`_FILE`/`_CMD`) | (empty = plaintext 0600 file) | Fernet key for `EncryptedColdStorage` (needs `tribune[security]`) |

## Consolidation + provenance (P0)

| Env | Default | Meaning |
|---|---|---|
| `TRIBUNE_CONSOLIDATION_SCHEMA_ENFORCED` | `true` | Require schema IR on hardened path |
| `TRIBUNE_HMAC_KEY_ID` | `k1` | Active signing key id |
| `TRIBUNE_HMAC_SECRET` (+`_FILE`/`_CMD`) | (empty → ephemeral dev key) | **Set in production**; file refs are mode-checked, command refs run without shell |
| `TRIBUNE_HMAC_SECRET_<KEYID>` (+`_FILE`/`_CMD`) | — | Retired/rotated keys |

## Intent graph (P1)

| Env | Default | Meaning |
|---|---|---|
| `TRIBUNE_INTENT_GRAPH_ENABLED` | `true` | Kill switch |
| `TRIBUNE_INTENT_WINDOW` | `8` | Sliding window W |
| `TRIBUNE_INTENT_DECAY_LAMBDA` | `0.15` | Temporal decay λ |
| `TRIBUNE_INTENT_TAU_THREAT` | `0.45` | Suspension threshold (lower = more sensitive) |
| `TRIBUNE_INTENT_THREAT_LIBRARY` | (empty = 10 seed exemplars) | JSON file with extra exemplars grown from incidents |
| `TRIBUNE_INTENT_ALLOWLIST` | (empty) | Comma-separated benign tool prefixes |

Tuning: `scripts/tune_intent_threshold.py` sweeps tau and recommends the most
conservative feasible point (fixture evidence: tau 0.45 → 1.0 detection, 0.0
FP; exploit peaks 0.51–0.65 vs benign ≤0.13). Start at 0.45; re-run with
production-derived `--benign/--exploits` fixtures before changing prod tau —
recall-favoring is deliberate (FP costs an admin review, FN costs a breach).

## MapReduce (P1)

| Env | Default | Meaning |
|---|---|---|
| `TRIBUNE_MAPREDUCE_ENABLED` | `true` | Kill switch (fall back to legacy walks) |
| `TRIBUNE_MAPREDUCE_WORKERS` | `4` | Parallel workers |
| `TRIBUNE_MAPREDUCE_SHARD_SIZE` | `64` | Candidates per shard |

## Reasoning + speculative embeddings (P2)

| Env | Default | Meaning |
|---|---|---|
| `TRIBUNE_REASONING_BUDGET_TOKENS` | `4000` | Active reasoning trace token cap |
| `TRIBUNE_SPECULATIVE_EMBEDDING` | `true` | Enable drafting cache hooks |

## Operations

- **Metrics**: `retention_metrics()`, `sanitizer.stats()`,
  `ProvenanceAuditLog.stats()`, `ProvenanceGatedMemory.stats()`,
  `ReasoningBudgetTracker.telemetry()`, `SpeculativeEmbeddingCache.stats()`,
  `RetentionAwareRetriever.stats()`, MapReduce `coverage` bundles.
- **Benchmarks**: `.venv/bin/python scripts/bench_hardened_roadmap.py [--json out.json]`.
- **Backfill**: dry-run first —
  `.venv/bin/python scripts/backfill_retention.py --in legacy.jsonl`, then `--apply`.
- **Audit verification**: `get_provenance_log().verify_chain()`.
- **Cold storage access**: least privilege on `TRIBUNE_RETENTION_COLD_PATH`.
