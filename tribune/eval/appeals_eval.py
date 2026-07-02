"""Sandboxed appeals-workflow eval.

Runs the appeals program end-to-end through the full pipeline with deterministic
seeds, inside the network-egress guard so the run provably touches no network
(and therefore no live agency system). This is the eval the container in
``sandbox/`` executes from a clean checkout with a single command.

Fixture-based by construction: the appeals cases come from the synthetic
generator, and the default provider/store/ingest are the offline local
implementations, so the entire run is reproducible and free.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TribuneSettings, get_settings
from ..types import ProgramId
from . import netguard
from .harness import EvalHarness, EvalResult


@dataclass
class AppealsEvalOutcome:
    result: EvalResult
    blocked_egress: list[str]
    allowlist: list[tuple[str, int]]


def run_appeals_eval(
    settings: TribuneSettings | None = None, n: int = 24
) -> AppealsEvalOutcome:
    settings = settings or get_settings()
    allowlist = netguard.allowlist_from_settings(settings)
    harness = EvalHarness(settings)
    with netguard.deny_egress(allowlist=allowlist) as guard:
        result = harness.run(n_per_program=n, programs=[ProgramId.APPEALS])
        blocked = list(guard.blocked)
    return AppealsEvalOutcome(
        result=result, blocked_egress=blocked, allowlist=sorted(allowlist)
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sandboxed appeals-workflow eval")
    parser.add_argument("--n", type=int, default=24, help="appeals cases to run")
    args = parser.parse_args(argv)

    outcome = run_appeals_eval(n=args.n)
    print("=== TRIBUNE sandboxed appeals eval ===")
    print(f"  egress allowlist   : {outcome.allowlist or 'empty (fully offline)'}")
    print(f"  blocked egress     : {len(outcome.blocked_egress)} attempt(s)")
    for b in outcome.blocked_egress:
        print(f"      ! blocked {b}")
    print()
    print(outcome.result.report.render())
    print()
    print(outcome.result.cost_report.render())
    # A blocked egress attempt means code tried to reach the network during an
    # eval — a hard failure for a fixtures-only run.
    return 1 if outcome.blocked_egress else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
