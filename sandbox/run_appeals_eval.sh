#!/usr/bin/env bash
# Entrypoint for the sandboxed appeals-workflow eval runtime.
#
# Runs the appeals eval end-to-end with deterministic seeds, in-process
# network-egress guard, timeout trap, and artifact emission.
set -euo pipefail

: "${TRIBUNE_SEED:=7}"
: "${TRIBUNE_TIMEOUT_S:=60}"
: "${TRIBUNE_ARTIFACTS_DIR:=/app/artifacts}"

export TRIBUNE_SEED
export TRIBUNE_TIMEOUT_S
export TRIBUNE_ARTIFACTS_DIR

mkdir -p "${TRIBUNE_ARTIFACTS_DIR}"

echo "========================================================================="
echo "TRIBUNE Sandboxed Appeals Eval Runtime"
echo "Seed: ${TRIBUNE_SEED} | Egress: DENY-BY-DEFAULT | Artifacts: ${TRIBUNE_ARTIFACTS_DIR}"
echo "========================================================================="

# Trap signals for clean termination
trap 'echo "[SANDBOX-TRAP] SIGINT/SIGTERM received. Terminating safely."; exit 143' INT TERM

exec python -m tribune.eval.appeals_eval "$@"
