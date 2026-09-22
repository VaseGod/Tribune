# Migration Notes — Hardened Roadmap

No breaking changes. All existing public APIs keep their signatures:

- `MemoryEventsTimeline.append(...)` — original kwargs unchanged; retention
  kwargs (`event_id`, `turn_id`, `tool_name`, `stdout`, …) are optional.
- `HierarchicalDocumentaryMemory`, `HDMMemory` (`encode/store/decode/bind/bundle/similarity/quantize`), `CitationLockHarness`, `MemoryConsolidator`, `SecureForge`, `SentinelSecurityProxy`, `SecureForgeRuntime`, `RelationalEpisodicMemory`, `ContextPilot` — untouched.
- New hardened paths are additive: `retention.py`, `consolidation_schema.py`,
  `security/provenance.py`, `ProvenanceGatedMemory`, `RetentionAwareRetriever`,
  `SlidingIntentGraphAnalyzer`, `AgenticMapReduceHarness`,
  `ReasoningBudgetTracker`, `SpeculativeEmbeddingCache`.

## Historical data

- Legacy `TimelineEvent` dicts are detected by missing `schema_version` and
  migrated via `scripts/backfill_retention.py` (dry-run default; writes
  sidecar `deltas.jsonl`, never mutates history in place).
- Existing memory nodes without HMAC envelopes continue to work through legacy
  APIs; they are **not** admitted to `ProvenanceGatedMemory` until re-signed
  through `SecureConsolidator.consolidate(...)` (preserves provenance links
  via `source_episodic_ids`).

## Deprecations

None. If a future release removes legacy free-form consolidation persistence,
it will ship adapters + a migration window first.
