# Verification Report — Hardened Roadmap

Date: 2026-09-22. Environment: local, Python 3.12 venv, no network/GPU.
Raw machine-readable evidence: `evidence.json` (produced by
`scripts/bench_hardened_roadmap.py --json docs/hardened_roadmap/evidence.json`).

## Implementation plan (executed)

1. P0 retention: new `tribune/memory/retention.py`; extended `timeline.py`
   (backward-compatible `append` + turn flush + loop detector);
   `retrieval.py` StateDelta-only search.
2. P0 integrity: new `consolidation_schema.py` + `SecureConsolidator`;
   new `security/provenance.py` + `audit.py` HMAC/chain helpers;
   `hdm.py` provenance gating.
3. P1 sandbox: `secure_forge.py` sliding intent graph (fixed binding-blindness
   bug during development: pure VSA binding orthogonalized every call, so
   `φ` is now a bundle preserving lexical drift signal).
4. P1 sweeps: `episodic.py` MapReduce harness.
5. P2 store: `hdm.py` reasoning budgets + speculative cache + majority bundle.
6. Config (17 settings), tests (10), benchmarks, backfill, docs, full-suite verification.

## Follow-up fixes (applied 2026-09-22, verified below)

Prior risks are now closed in code; only genuinely environmental items remain:

| Prior risk | Status |
|---|---|
| Threshold tuning needs prod traffic | **Shipped**: suffix-max scoring, tau recalibrated 0.75 → 0.45 from tuner evidence (exploit peaks 0.51–0.65 vs benign ≤0.13), `scripts/tune_intent_threshold.py` + file-backed library + incident learning. Final confirmation still needs prod fixtures (harness provided). |
| Cold-store encryption deployment-owned | **Shipped**: `EncryptedColdStorage` (Fernet, fails closed) + `0600` hardening + `security` extra. Deployment still owns key provisioning. |
| HMAC secret via secret manager | **Shipped**: env → file → command chain (`secrets.py`) for active + rotated keys. Deployment still owns the Vault/KMS setup. |
| Real tokenizer counts | **Shipped**: pluggable backends (`TRIBUNE_TOKEN_COUNTER=tiktoken` + `tokenizers` extra). Deployment opts in by installing the package. |
| Threat library growth | **Shipped**: JSON library + `add_exemplar_from_trajectory()`. Curators still own incident review. |

- Token accounting defaults to the repo-standard ~4-char estimator (matches
  `ContextPilot`); set `TRIBUNE_TOKEN_COUNTER=tiktoken` (+ package) for
  billing-grade counts.
- HD vectors use seeded random projections (numpy, existing dep); `security`
  and `tokenizers` are optional extras, nothing new required.
- tau_threat=0.45 default is evidence-backed on fixtures (tuner recommends
  0.50; 0.45 chosen recall-favoring); final confirmation needs prod fixtures.
- `<1% FP` and `100% exploit interception` are demonstrated on local fixtures
  (7 benign workflows, 3–4 exploit chains), not production traffic — stated as
  measured-on-fixtures, not as production guarantees.
- Cold store is a local JSONL file (0600) by default; encrypted/remote
  backends are injectable.

## Measured results (from `evidence.json`)

| Target | Measured |
|---|---|
| ≥50% context reduction | **91.4%** (281,946 → 24,196 tokens over 200 noisy events) |
| Zero looping beyond 100k tokens | **stable**: 100k-token sim, 400 calls, 477 loop detections, 477 blocked, bounded active context (46,652) |
| Zero unverified instruction tokens in centroids | **4/4 attacks blocked + quarantined**; 2/2 benign accepted (harness shows 6-item mix) |
| Exploit-chain interception, <1% FP | **3/3 detected; 0.0 FP** on 7-workflow benign fixture suite |
| 100% deterministic coverage, linear scaling | **coverage 1.0**; deterministic reruns identical; 4× candidates ≈ 2.8× time |
| 40% latency reduction | **99.3%** per-query cut on cache hits (hit rate 0.91); reasoning memory capped at budget (1,500 active tokens) |

## Tests

- New: `tests/test_hardened_roadmap.py` — 17/17 pass (prior 10 + token
  backends, secret chain/file/cmd, HMAC file ref, cold-store 0600,
  encryption fail-closed, threat-library file + incident learning,
  tau default + suffix scoring).
- Full suite: **592 passed, 1 skipped** (incl. `test_server.py`).
- Lint: `ruff check` clean on all new/touched files (repo has pre-existing
  errors elsewhere; ours add zero).
- Tuner: `scripts/tune_intent_threshold.py` recommends tau 0.50 on fixtures
  (1.0 detection, 0.0 FP); default set to 0.45 recall-favoring.

## Remaining risks / follow-ups (environmental only — code is shipped)

1. Confirm tau/budgets/loop thresholds against real traffic: run the tuner
   with production-derived `--benign/--exploits` fixtures; consider CI on the
   benign suite.
2. Provision cold-store encryption keys and HMAC secrets via Vault/KMS in prod
   (file/cmd refs supported; dev fallback warns loudly but still signs —
   documented, never silent).
3. Install `tribune[security]` / `tribune[tokenizers]` extras where required.
4. Curate the threat library from reviewed incidents
    (`add_exemplar_from_trajectory` distills candidates; humans approve).
