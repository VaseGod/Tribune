"""Cryptographic Trace Ledger Verification Utility & CLI.

Validates immutable hash-chain integrity of execution trace ledgers for regulatory
oversight bodies and compliance reviewers.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .tracing import GENESIS_HASH, HashedTraceEvent


def verify_trace_file(file_path: str) -> tuple[bool, str | None, dict[str, Any]]:
    """Verify cryptographic integrity of an exported ledger JSON file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"Failed to read trace file: {exc}", {}

    raw_events = data.get("events", [])
    if not raw_events:
        return False, "Trace file contains no events", data

    expected_prev = GENESIS_HASH
    for idx, raw_evt in enumerate(raw_events):
        evt = HashedTraceEvent(
            event_id=raw_evt.get("event_id", ""),
            parent_event_id=raw_evt.get("parent_event_id"),
            timestamp=raw_evt.get("timestamp", ""),
            task_id=raw_evt.get("task_id", ""),
            run_id=raw_evt.get("run_id", ""),
            step_type=raw_evt.get("step_type", ""),
            tool_name=raw_evt.get("tool_name"),
            tool_inputs_digest=raw_evt.get("tool_inputs_digest"),
            model_provider=raw_evt.get("model_provider"),
            model_id=raw_evt.get("model_id"),
            tier=raw_evt.get("tier"),
            token_usage=raw_evt.get("token_usage", {}),
            cost_estimate_usd=raw_evt.get("cost_estimate_usd", 0.0),
            verifier_result=raw_evt.get("verifier_result"),
            ast_verifier_rejection_details=raw_evt.get("ast_verifier_rejection_details"),
            security_policy_trigger_details=raw_evt.get("security_policy_trigger_details"),
            exception_details=raw_evt.get("exception_details"),
            outcome=raw_evt.get("outcome", "SUCCESS"),
            previous_hash=raw_evt.get("previous_hash", ""),
            current_hash=raw_evt.get("current_hash", ""),
        )

        # 1. Previous hash link assertion
        if evt.previous_hash != expected_prev:
            return (
                False,
                f"Broken chain link at index {idx} ({evt.event_id}): "
                f"previous_hash '{evt.previous_hash}' != expected '{expected_prev}'",
                data,
            )

        # 2. Recompute hash assertion
        recomputed = evt.compute_hash()
        if evt.current_hash != recomputed:
            return (
                False,
                f"Tampering detected at index {idx} ({evt.event_id}): "
                f"stored '{evt.current_hash}' != recomputed '{recomputed}'",
                data,
            )

        expected_prev = evt.current_hash

    return True, None, data


def main() -> int:
    parser = argparse.ArgumentParser(description="TRIBUNE Cryptographic Trace Verifier")
    parser.add_argument("trace_file", type=str, help="Path to exported trace ledger JSON file")
    args = parser.parse_args()

    print(f"=== Verifying Trace Ledger: {args.trace_file} ===")
    valid, error, data = verify_trace_file(args.trace_file)

    if valid:
        total = len(data.get("events", []))
        term_hash = data.get("terminal_hash", "")[:16]
        print(f"✅ VERIFICATION PASSED: All {total} events cryptographically sealed and untampered.")
        print(f"   Terminal Hash: {term_hash}...")
        return 0
    else:
        print(f"❌ VERIFICATION FAILED: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
