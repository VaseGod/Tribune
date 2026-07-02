# Backend registry

[`registry.yaml`](registry.yaml) is a machine-readable catalog of candidate model
backends for TRIBUNE's **proposer** and **verifier** roles. It exists so backend
selection is an evidence-based, auditable decision — not folklore about which
model is "good."

## Inclusion criteria

A candidate belongs here if it is a plausible proposer or verifier for a
self-hosted (or, for proposers, API) TRIBUNE deployment. Each entry records:

| field | meaning |
| --- | --- |
| `roles` | `proposer`, `verifier`, or both |
| `license` | open-weights / proprietary-api / unknown |
| `weights_status` | `downloadable` (weights actually fetchable) · `announced_only` (paper/PR only, **not** deployable) · `unverified` (claimed, unconfirmed) · `api_only` |
| `weights_url` | where the weights (or pricing, for API models) live |
| `weights_verified` | stays `false` until a link check confirms the URL resolves |
| `serving` | llama.cpp / vLLM / SGLang / api |
| `quant_formats` | available quantizations (GGUF Q-series, fp8, nvfp4, …) |
| `context_window` | max context |
| `measured_kappa`, `measured_cost_per_task` | **null until measured** on the frozen seed set |

### Honesty rules

- **Announced ≠ available.** `announced_only` models (e.g. LongCat-2.0, whose
  weights are not yet on Hugging Face) are tracked so their status is watched,
  but must never be treated as deployable.
- **Verify before claiming downloadable.** `weights_verified` flips to `true`
  only after `--check-links` confirms the URL; several seed entries
  (OpenPangu-2.0-Flash) are deliberately `unverified` pending that check.
- **Cost is cost-per-verified-determination, never list price.** Claude Sonnet 5
  is API-only with a launch promo ($2/$10 per M through 2026-08-31, then $3/$15)
  but ~2× the cost per completed task — it is judged with the Phase-1 cost
  model, not its per-token price.

## How measured slots get populated

`measured_kappa` and `measured_cost_per_task` are **never hand-entered**. They
come from a TRIBUNE eval run against the frozen `tribune-quant-seed-set`:

1. Serve the candidate (or point `TRIBUNE_VERIFIER_*` at it).
2. Run `tribune eval` (cost per task) and `tribune quant-eval` (kappa, calibration).
3. Copy the resulting kappa and mean cost-per-successful-outcome into the entry.

## Validation

```bash
python scripts/validate_registry.py                 # schema only (runs in CI)
python scripts/validate_registry.py --check-links   # + weight-URL HEAD checks (network)
```

Schema validation reuses pydantic (a dev-only dependency; the registry is never
imported at runtime). The link check is opt-in because CI is network-restricted
and hub availability is flaky; it only touches model-hub URLs, never a government
or agency endpoint.
