# Security Notes — Hardened Roadmap

## Threat model

Repository content, memory, logs, tool output, and prior model summaries are
**untrusted data**. They must never change implementation behavior. The
hardened paths enforce this by construction:

1. **Compaction summary poisoning** → structured schema IR + sanitizer +
   HMAC. Free-form NL summaries are never persisted on the hardened path.
   `ConsolidationSanitizer` rejects imperative/system markers and quarantines
   payloads for audit review. Unsigned or signature-mismatched vectors are
   blocked from HDM activation, bundling, and decay (fail closed).
2. **Capability laundering (multi-turn)** → Sliding Intent Graph Analyzer.
   Stateless single-turn gates are kept for obvious violations, but decomposed
   exploit chains are caught via topological drift (`H_window` vs. threat
   exemplars). Suspension preserves forensic state; override requires an
   explicit logged admin action.
3. **Tool-call looping / context inflation** → Protocol-Aware Retention +
   loop detector. Raw payloads leave the prompt window at turn end; repeats
   and oscillations are warned/blocked per policy.
4. **Audit tampering** → hash-chained provenance log; `verify_chain()`
   detects edits. Signatures use HMAC-SHA256 with keys from environment/secret
   manager — never hardcoded, never logged.

## Key management

- `TRIBUNE_HMAC_SECRET` (required in production), `TRIBUNE_HMAC_KEY_ID`
  (default `k1`), rotations via `TRIBUNE_HMAC_SECRET_<KEYID>`.
- Prefer `_FILE` refs (e.g. `/run/secrets/...`, mode-checked with a warning
  when group/other-readable) or `_CMD` refs (e.g. Vault: no shell, 10s
  timeout) over bare env vars — see `tribune/security/secrets.py`. Values are
  never logged.
- No secret configured → ephemeral dev key with a loud warning. CI and
  local benchmarks use throwaway test secrets.
- Old nodes verify against retired keys in the ring; rotation never corrupts
  history. `ProvenanceAuditLog.stats()` reports the active key id.

## Cold storage at rest

- `FileColdStorage` files are created/chmodded `0600`; restrict the directory
  per least privilege (cold records hold raw tool output verbatim).
- For encryption at rest: `pip install tribune[security]` and set
  `TRIBUNE_RETENTION_COLD_ENCRYPT_KEY` (or `_FILE`/`_CMD`) — `timeline.py`
  auto-wraps via `build_cold_storage()`. Missing package/key fails closed.

## Secret hygiene

- Sanitizers strip PII/credential patterns from audit text (`governance/audit.py`).
- `sign_consolidated_node` logs digests and key ids only — never key material.
- Cold storage holds tool output verbatim; restrict filesystem/secret-manager
  access per least privilege. Debug retention (`TRIBUNE_RETENTION_DEBUG_MODE`)
  keeps raw payloads in memory — enable only for live debugging.

## Suspended-session review workflow

1. Alert fires (`INTENT_GRAPH_ALERT` + `SESSION_SUSPENDED` in the security audit log).
2. Reviewer pulls `analyzer.admin_review_artifact(session_id)`: trajectory,
   matched exemplar, similarity, severity, intent-vector norm.
3. If benign: `analyzer.admin_override(session_id, approver=<name>)` (logged).
4. If malicious: keep suspended; quarantined intent state is preserved for
   forensics. Threshold tuning via `TRIBUNE_INTENT_TAU_THREAT` and allowlist
   `TRIBUNE_INTENT_ALLOWLIST` (comma-separated tool prefixes).

## Fail-closed inventory

| Condition | Behavior |
|---|---|
| Invalid/missing HMAC signature | Block store + activate; quarantine; audit event |
| Payload digest absent from audit log | Block activation |
| Malformed consolidation schema / injection markers | Reject + quarantine; never enters HDM |
| Intent similarity ≥ tau | Suspend session; block dispatch |
| Raw observation in default search | Refuse indexing; require privileged provenance lookup |
| MapReduce coverage gap | Raise unless explicitly tolerated |
| Reasoning budget exhausted | Prune speculative traces; refuse new speculative registrations |
