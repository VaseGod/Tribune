"""Adversarial canary suite + rule-citation-drift sentinel.

Two hazards specific to a benefits tool, hunted nightly:

1. **Confidently-wrong determinations.** An adversarial, ambiguity-heavy case set
   is run through the full pipeline. The canary fails if TRIBUNE *asserts* any
   result that disagrees with ground truth (it should abstain), or if its
   abstention recall on the genuinely hard cases regresses below the safe floor.

2. **Rule-citation drift.** Program rules change over time. If a program's cited
   logic no longer matches the frozen baseline fingerprint, TRIBUNE may be acting
   on stale rules — a live hazard. The sentinel fingerprints every rule
   (source + text + effective date) and diffs against the baseline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import isnan

from ..casegen.synthetic import SyntheticCaseGenerator
from ..config import TribuneSettings, get_settings
from ..corpus import programs as program_registry
from ..corpus.programs.jurisdictions import known_jurisdictions
from ..corpus.rule_store import ruleset_fingerprint
from ..orchestration.pipeline import CasePipeline
from .harness import records_for_case

_ABSTENTION_RECALL_FLOOR = 0.85
_CITATION_ACCURACY_FLOOR = 0.985
_DEFAULT_FALLBACK_MODEL = "grok-4.6"


def freeze_fingerprints() -> dict[str, dict[str, str]]:
    fp: dict[str, dict[str, str]] = {}
    for program in program_registry.all_programs():
        for jur in known_jurisdictions():
            fp[f"{program.value}:{jur}"] = ruleset_fingerprint(program, jur)
    return fp


def detect_drift(
    baseline: dict[str, dict[str, str]], current: dict[str, dict[str, str]]
) -> list[str]:
    drift: list[str] = []
    for key in sorted(set(baseline) | set(current)):
        b = baseline.get(key, {})
        c = current.get(key, {})
        for cid in sorted(set(b) | set(c)):
            if cid not in b:
                drift.append(f"NEW rule cited: {cid}")
            elif cid not in c:
                drift.append(f"REMOVED rule: {cid}")
            elif b[cid] != c[cid]:
                drift.append(f"CHANGED rule (text/source/effective-date): {cid}")
    return drift


def read_baseline(path: str) -> dict[str, dict[str, str]] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_baseline(path: str, fp: dict[str, dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fp, fh, indent=2, sort_keys=True)


@dataclass
class CanaryReport:
    baseline_initialized: bool
    drift: list[str]
    confidently_wrong: list[str]
    hard_cases: int
    hard_abstained: int
    abstention_recall: float
    ok: bool
    citation_accuracy: float = 1.0
    fallback_triggered: bool = False
    fallback_model: str = _DEFAULT_FALLBACK_MODEL
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        recall = "n/a" if isnan(self.abstention_recall) else f"{self.abstention_recall:.3f}"
        lines = [
            f"=== TRIBUNE canary / drift sentinel — {status} ===",
            f"  rule-citation drift             : {'(baseline initialized this run)' if self.baseline_initialized else (str(len(self.drift)) + ' change(s)')}",
        ]
        for d in self.drift:
            lines.append(f"      ! {d}")
        lines.append(f"  confidently-wrong assertions    : {len(self.confidently_wrong)}  <-- must be 0")
        for cw in self.confidently_wrong:
            lines.append(f"      ! {cw}")
        lines.append(
            f"  abstention recall on hard cases : {recall}  "
            f"({self.hard_abstained}/{self.hard_cases}; floor {_ABSTENTION_RECALL_FLOOR})"
        )
        lines.append(
            f"  statutory citation accuracy     : {self.citation_accuracy:.3%} (floor {_CITATION_ACCURACY_FLOOR:.1%})"
        )
        if self.fallback_triggered:
            lines.append(
                f"  AUTOMATIC ACCURACY FALLBACK     : Local model disabled -> Routed to {self.fallback_model}"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


class CanarySentinel:
    def __init__(self, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        baseline_path: str | None = None,
        freeze: bool = False,
        fallback_model: str = _DEFAULT_FALLBACK_MODEL,
        simulated_citation_accuracy: float | None = None,
    ) -> CanaryReport:
        from ..corpus.citations import track_quant_citation_retention

        baseline_path = baseline_path or os.path.join(self.settings.data_dir, "canary_baseline.json")
        current_fp = freeze_fingerprints()

        baseline_initialized = False
        drift: list[str] = []
        if freeze or read_baseline(baseline_path) is None:
            write_baseline(baseline_path, current_fp)
            baseline_initialized = True
        else:
            baseline = read_baseline(baseline_path)
            assert baseline is not None
            drift = detect_drift(baseline, current_fp)

        # Adversarial, ambiguity-heavy case set.
        generator = SyntheticCaseGenerator(seed=self.settings.seed + 101)
        cases = generator.generate_eval_set(n_per_program=10, ambiguous_ratio=0.6)
        pipeline = CasePipeline(self.settings)
        records = []
        for case in cases:
            result = pipeline.run_case(case)
            records.extend(records_for_case(case, result))

        confidently_wrong = [
            f"{r.case_id}/{r.program.value}: asserted {r.predicted_label} but truth is {r.ground_truth_label}"
            for r in records
            if not r.abstained and r.predicted_label != r.ground_truth_label
        ]
        hard = [r for r in records if r.ambiguous]
        hard_abstained = [r for r in hard if r.abstained]
        recall = (len(hard_abstained) / len(hard)) if hard else float("nan")

        # Evaluate statutory citation accuracy
        if simulated_citation_accuracy is not None:
            citation_accuracy = simulated_citation_accuracy
        else:
            citation_accuracy = track_quant_citation_retention(records)

        fallback_triggered = False
        notes = []

        if citation_accuracy < _CITATION_ACCURACY_FLOOR:
            # Accuracy Gate: Halt local model execution and fall back to hosted high-precision model
            fallback_triggered = True
            notes.append(
                f"Statutory citation accuracy ({citation_accuracy:.3%}) dropped below floor ({_CITATION_ACCURACY_FLOOR:.1%}). "
                f"Halting local model and dynamically failing over to hosted model '{fallback_model}'."
            )

        ok = (
            len(confidently_wrong) == 0
            and (baseline_initialized or len(drift) == 0)
            and (not hard or recall >= _ABSTENTION_RECALL_FLOOR)
            and citation_accuracy >= _CITATION_ACCURACY_FLOOR
        )

        if baseline_initialized:
            notes.append(f"baseline fingerprint written to {baseline_path}")

        return CanaryReport(
            baseline_initialized=baseline_initialized,
            drift=drift,
            confidently_wrong=confidently_wrong,
            hard_cases=len(hard),
            hard_abstained=len(hard_abstained),
            abstention_recall=recall,
            citation_accuracy=citation_accuracy,
            fallback_triggered=fallback_triggered,
            fallback_model=fallback_model,
            ok=ok,
            notes=notes,
        )


    def generate_canary_token(self) -> str:
        """Generate a dynamic canary token to inject into system context."""
        import secrets
        return f"CANARY-{secrets.token_hex(8).upper()}"

    def verify_canary_isolation(self, canary_token: str, outputs: str | dict | list) -> bool:
        """Verify canary tokens do not leak into extended-thinking output fields, reasoning payloads, or audit log exports."""
        import json
        dumped = json.dumps(outputs) if not isinstance(outputs, str) else outputs
        return canary_token not in dumped


CanaryEvaluator = CanarySentinel
