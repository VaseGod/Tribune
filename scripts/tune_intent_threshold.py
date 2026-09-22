"""Tau auto-tuning harness for the Sliding Intent Graph Analyzer.

Sweeps tau over benign developer fixtures + synthetic exploit chains and
reports detection rate vs. false-positive rate per threshold, recommending the
highest-detection tau subject to FP <= 1%.

Bring your own fixtures for production tuning:
  --benign path/to/benign.json   ({workflow: [{tool, params}, ...]})
  --exploits path/to/chains.json ([[{tool, params}, ...], ...])
  --threats path/to/library.json (extra threat exemplars)

Usage: .venv/bin/python scripts/tune_intent_threshold.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tribune.security.secure_forge import (  # noqa: E402
    BENIGN_WORKFLOW_FIXTURES,
    DEFAULT_THREAT_EXEMPLARS,
    IntentGraphConfig,
    SlidingIntentGraphAnalyzer,
    load_threat_library,
)

DEFAULT_CHAINS = [
    [("read_file", {"cmd": "read config"}), ("list_dir", {"cmd": "list files"}),
     ("encode", {"cmd": "encode combine"}), ("send", {"cmd": "send combine upload external"})],
    [("run", {"cmd": "credential token collect stage"}), ("run", {"cmd": "secret harvest env"}),
     ("run", {"cmd": "exfiltrate upload curl external"})],
    [("run", {"cmd": "sudo chmod setuid escalate privilege"}),
     ("run", {"cmd": "sandbox escape mount nsenter breakout"})],
    [("run", {"cmd": "pickle loads gadget payload deserialize"}),
     ("run", {"cmd": "curl upload external exfiltrate"})],
]


def load_fixture_chains(path: str, benign: bool) -> list:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if benign:
        return [[(s.get("tool", "run"), s.get("params", {})) for s in steps]
                for steps in raw.values()]
    return [[(s.get("tool", "run"), s.get("params", {})) for s in chain] for chain in raw]


def chain_blocked(chain: list, tau: int | float, window: int, decay: float,
                  extra: dict, sid: str) -> bool:
    g = SlidingIntentGraphAnalyzer(
        IntentGraphConfig(window_size=window, decay_lambda=decay, tau_threat=float(tau)),
        threat_exemplars={**DEFAULT_THREAT_EXEMPLARS, **extra},
    )
    return any(not g.evaluate(sid, t, p)["allowed"] for t, p in chain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", default="")
    ap.add_argument("--exploits", default="")
    ap.add_argument("--threats", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--decay", type=float, default=0.15)
    args = ap.parse_args()

    extra = load_threat_library(args.threats) if args.threats else {}
    if args.benign:
        benign_chains = load_fixture_chains(args.benign, benign=True)
    else:
        benign_chains = [[(step, {"cmd": step}) for step in steps]
                         for steps in BENIGN_WORKFLOW_FIXTURES.values()]
    exploit_chains = load_fixture_chains(args.exploits, benign=False) if args.exploits else DEFAULT_CHAINS

    rows = []
    for i in range(14):  # tau 0.30..0.95
        tau = round(0.30 + 0.05 * i, 2)
        fp = sum(chain_blocked(c, tau, args.window, args.decay, extra, f"benign{i}")
                 for i, c in enumerate(benign_chains))
        det = sum(chain_blocked(c, tau, args.window, args.decay, extra, f"evil{i}")
                  for i, c in enumerate(exploit_chains))
        rows.append({
            "tau": tau,
            "detection_rate": round(det / max(1, len(exploit_chains)), 4),
            "fp_rate": round(fp / max(1, len(benign_chains)), 4),
        })
    feasible = [r for r in rows if r["fp_rate"] <= 0.01]
    # Most conservative feasible point: highest detection, then highest tau
    # (least sensitive while still catching everything in the fixture set).
    best = max(feasible, key=lambda r: (r["detection_rate"], r["tau"]), default=None)
    if best is None:
        best = min(rows, key=lambda r: (r["fp_rate"], -r["detection_rate"]))
    result = {"rows": rows, "recommended_tau": best["tau"],
              "at_recommended": best,
              "note": "Re-run with production-derived fixtures before changing prod tau."}
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
