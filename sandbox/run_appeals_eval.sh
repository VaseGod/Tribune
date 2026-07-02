#!/usr/bin/env bash
# Entrypoint for the sandboxed appeals-workflow eval.
#
# Runs the appeals eval end-to-end with deterministic seeds and the in-process
# network-egress guard. Exits non-zero if any network egress was attempted, so a
# fixtures-only run that tries to reach out fails loudly.
set -euo pipefail

: "${TRIBUNE_SEED:=7}"
export TRIBUNE_SEED

echo "TRIBUNE sandboxed appeals eval — seed=${TRIBUNE_SEED}, egress deny-by-default"
exec python -m tribune.eval.appeals_eval "$@"
