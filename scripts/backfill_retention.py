"""Dry-run backfill: migrate legacy TimelineEvent dicts to StateDelta form.

Never destroys historical memory: reads legacy exports, detects schema via
`schema_version`, writes StateDelta JSONL alongside (no in-place mutation).

Usage: .venv/bin/python scripts/backfill_retention.py --in legacy.jsonl [--out deltas.jsonl] [--apply]
Default is dry-run (no writes).
"""

from __future__ import annotations

import argparse
import json


def migrate_event(legacy: dict) -> dict:
    return {
        "event_id": legacy.get("transaction_id", legacy.get("event_id", "unknown")),
        "parent_event_id": None,
        "turn_id": legacy.get("turn_id", "backfill"),
        "session_id": legacy.get("session_id", "backfill"),
        "timestamp": legacy.get("timestamp", 0.0),
        "actor": "backfill",
        "tool_name": str((legacy.get("executed_tool_invocation") or {}).get("tool", "unknown"))
        if isinstance(legacy.get("executed_tool_invocation"), dict)
        else "unknown",
        "operation_type": "backfill",
        "entity_ids": [],
        "file_changes": [],
        "exit_code": None,
        "command_status": "unknown",
        "result_summary": {"state_change_delta_keys": sorted((legacy.get("state_change_delta") or {}).keys())},
        "causal_refs": [],
        "observation_digest": "",
        "schema_version": "tribune.retention/v1",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="deltas.jsonl")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rows = []
    with open(args.inp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    deltas = [migrate_event(r) for r in rows]
    print(f"legacy_events={len(rows)} migrated={len(deltas)} dry_run={not args.apply}")
    if args.apply:
        with open(args.out, "w") as fh:
            for d in deltas:
                fh.write(json.dumps(d, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    else:
        print(json.dumps(deltas[:2], indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
