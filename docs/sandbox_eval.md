# Sandboxed eval environments

Stateful agent evals should be reproducible and isolated: same inputs, same
seeds, same result, and **no ability to reach a live system** — which for a
benefits tool is a hard safety requirement (never touch a real agency endpoint).
Phase 6 provides that for the appeals-workflow eval.

## One command

From a clean checkout:

```bash
make sandbox-appeals-eval
```

This builds the container ([sandbox/Dockerfile](../sandbox/Dockerfile)) and runs
the appeals eval end-to-end inside it with `--network=none`. It also runs
locally without Docker:

```bash
make appeals-eval          # same eval, in-process egress guard only
```

## Two layers of egress denial

1. **Container boundary.** The documented `docker run --network=none` gives the
   eval no network interface at all. When a remote model endpoint *is*
   configured, drop `--network=none` and rely on layer 2's allowlist.
2. **In-process guard.** [`tribune/eval/netguard.py`](../tribune/eval/netguard.py)
   patches the socket layer to **deny egress by default**; only the configured
   remote model endpoint (derived from settings) is allowlisted. Any other
   outbound TCP connection raises `NetworkEgressBlocked`, naming the host. This
   works in CI and on a laptop, not just in the container.

The appeals eval's default configuration is fully offline (local provider, local
rule store, structured ingest), so the allowlist is **empty** and egress is
total-deny. `run_appeals_eval` returns the list of blocked egress attempts; the
entrypoint exits non-zero if it is non-empty, so a fixtures-only run that tries
to phone home fails loudly.

## Reproducibility

- Base image and dependencies are pinned
  ([sandbox/requirements.lock](../sandbox/requirements.lock)); replace the
  Dockerfile's base-image digest placeholder with the digest your org pins.
- Seeds are deterministic (`TRIBUNE_SEED=7`, `PYTHONHASHSEED=0`), and a test
  asserts two runs produce identical outcomes.
- Everything is fixture-based: the appeals cases come from the synthetic
  generator; there are no real agency endpoints, mocked or otherwise.

## The acceptance test

`tests/test_netguard.py::test_appeals_eval_runs_end_to_end_with_zero_egress`
runs the appeals eval under the guard and asserts **zero** network egress
occurred, that the run actually covered the appeals program, and that it is
deterministic across runs.
