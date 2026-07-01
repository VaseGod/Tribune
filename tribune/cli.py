"""TRIBUNE command-line interface.

    tribune demo      walk full synthetic cases (incl. abstentions) end-to-end
    tribune eval      run the evaluation harness and print the metrics table
    tribune canary    run the adversarial canary + rule-drift sentinel
    tribune disclose  print the plain-language explanation of the last demo run
    tribune web       serve the web UI (FastAPI + single-page app)

Everything runs offline with the default local providers. No API keys, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import get_settings
from .governance import disclosure
from .types import CaseRunResult

_BANNER = (
    "TRIBUNE — verification-first benefits navigator (informational only; not legal "
    "advice; never a binding determination; abstains and routes to a human when unsure)."
)


def _last_run_path() -> str:
    return os.path.join(get_settings().data_dir, "last_run.json")


def _persist_results(results: list[CaseRunResult]) -> None:
    path = _last_run_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.model_dump(mode="json") for r in results], fh, indent=2, default=str)


def cmd_demo(args: argparse.Namespace) -> int:
    from .casegen.synthetic import SyntheticCaseGenerator
    from .orchestration.pipeline import CasePipeline

    settings = get_settings()
    print(_BANNER)
    print(f"provider={settings.provider}  rule_store={settings.rule_store}  "
          f"ingest={settings.doc_ingest}  abstention_threshold={settings.abstention_threshold}\n")

    generator = SyntheticCaseGenerator(seed=settings.seed)
    cases = generator.generate_demo_set()
    pipeline = CasePipeline(settings)

    results: list[CaseRunResult] = []
    for case in cases:
        result = pipeline.run_case(case)
        results.append(result)
        print("#" * 78)
        print(disclosure.render(result))
        chain_ok = pipeline.audit.verify_chain(case.case_id)
        print(f"\n  [audit chain verified: {chain_ok}]\n")

    _persist_results(results)

    # Compact summary so the safety behaviors are visible at a glance.
    print("#" * 78)
    print("SUMMARY")
    for result in results:
        for o in result.outcomes:
            if o.abstained:
                tag = "ABSTAIN -> human"
            elif o.materials is not None:
                tag = f"{o.assessment.status.value} -> prepared (not submitted)"
            elif o.assessment is not None:
                tag = f"{o.assessment.status.value} -> recommend {o.assessment.recommended_action.value}"
            else:
                tag = "no assessment"
            replan = f" (replans={o.replans})" if o.replans else ""
            print(f"  {result.case_id:24s} {o.program.value:13s} {tag}{replan}")
    print(f"\nSaved last run to {_last_run_path()} (use `tribune disclose`).")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.harness import EvalHarness
    from .types import ProgramId

    settings = get_settings()
    print(_BANNER + "\n")
    harness = EvalHarness(settings)
    programs = [ProgramId(p) for p in args.programs] if args.programs else None
    result = harness.run(
        n_per_program=args.n, ambiguous_ratio=args.ambiguous_ratio, programs=programs
    )
    print(result.report.render_full())
    print()
    print(result.cost_report.render())
    return 0


def cmd_canary(args: argparse.Namespace) -> int:
    from .eval.canary import CanarySentinel

    settings = get_settings()
    report = CanarySentinel(settings).run(freeze=args.freeze)
    print(report.render())
    return 0 if report.ok else 1


def cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            'The web UI needs extra dependencies. Install them with:\n'
            '    pip install -e ".[web]"',
            file=sys.stderr,
        )
        return 1
    print(_BANNER)
    print(f"Serving TRIBUNE at http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    uvicorn.run("tribune.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_disclose(args: argparse.Namespace) -> int:
    path = _last_run_path()
    if not os.path.exists(path):
        print("No saved run found; running the demo first...\n")
        return cmd_demo(args)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    for d in payload:
        result = CaseRunResult.model_validate(d)
        print("#" * 78)
        print(disclosure.render(result))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tribune", description=_BANNER)
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="walk full synthetic cases end-to-end")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("eval", help="run the evaluation harness and print metrics")
    p_eval.add_argument("--n", type=int, default=24, help="cases per program")
    p_eval.add_argument("--ambiguous-ratio", dest="ambiguous_ratio", type=float, default=0.25)
    p_eval.add_argument(
        "--program",
        dest="programs",
        action="append",
        default=None,
        help="restrict to a program (repeatable), e.g. --program appeals",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_canary = sub.add_parser("canary", help="adversarial canary + rule-drift sentinel")
    p_canary.add_argument("--freeze", action="store_true", help="(re)write the drift baseline")
    p_canary.set_defaults(func=cmd_canary)

    p_disc = sub.add_parser("disclose", help="plain-language explanation of the last run")
    p_disc.set_defaults(func=cmd_disclose)

    p_web = sub.add_parser("web", help="serve the web UI (FastAPI + single-page app)")
    p_web.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 to expose on the network)")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
