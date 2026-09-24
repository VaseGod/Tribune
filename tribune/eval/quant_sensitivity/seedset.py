"""The frozen quant-sensitivity seed set.

~50 determination cases sampled deterministically from the synthetic corpus,
weighted toward Medicaid (state-dependent eligibility complexity) and SNAP. The
set is frozen by a manifest (case ids + a content hash over the full case
payloads) so ladder runs remain comparable over time: if the generator or the
rule corpus changes in a way that alters the cases, the hash changes and the
harness refuses to compare against stale results.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date

from ...casegen.synthetic import SyntheticCaseGenerator
from ...types import ProgramId, SyntheticCase

#: cases per program — weighted toward Medicaid and SNAP (50 total).
SEED_WEIGHTS: dict[ProgramId, int] = {
    ProgramId.MEDICAID: 15,
    ProgramId.SNAP: 15,
    ProgramId.UNEMPLOYMENT: 8,
    ProgramId.HOUSING: 6,
    ProgramId.APPEALS: 6,
}

_DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "data", "seed_manifest.json")
_SEED = 20260701  # frozen; changing it redefines the seed set


def build_seed_set(limit_per_program: dict[ProgramId, int] | None = None) -> list[SyntheticCase]:
    """Deterministically build the weighted seed set from the synthetic corpus.

    Generation goes through the public ``generate_eval_set`` (which produces a
    per-program mix of clear-eligible / clear-ineligible / ambiguous cases) and
    takes the first N per program according to the weights.
    """
    weights = limit_per_program or SEED_WEIGHTS
    n_max = max(weights.values())
    generator = SyntheticCaseGenerator(seed=_SEED)
    pool = generator.generate_eval_set(n_per_program=n_max, ambiguous_ratio=0.3)
    out: list[SyntheticCase] = []
    taken: dict[ProgramId, int] = {p: 0 for p in weights}
    for case in pool:
        program = case.target_programs[0]
        want = weights.get(program, 0)
        if taken.get(program, 0) < want:
            out.append(case)
            taken[program] = taken.get(program, 0) + 1
    return out


def seed_set_hash(cases: list[SyntheticCase]) -> str:
    """Content hash over the *stable* case payloads.

    Covers everything that defines a case (situation, documents, ground truth,
    targets) while excluding volatile fields like provenance ingest timestamps,
    so the same generated set always hashes identically.
    """
    payload = [
        {
            "case_id": c.case_id,
            "jurisdiction": c.jurisdiction,
            "language": c.language,
            "situation": c.situation.model_dump(mode="json"),
            "documents": [d.model_dump(mode="json") for d in c.documents],
            "ground_truth": {
                p.value: gt.model_dump(mode="json") for p, gt in sorted(c.ground_truth.items())
            },
            "target_programs": [p.value for p in c.target_programs],
        }
        for c in cases
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(cases: list[SyntheticCase], path: str | None = None) -> dict:
    manifest = {
        "name": "tribune-quant-seed-set",
        "version": 1,
        "frozen_on": date.today().isoformat(),
        "generator_seed": _SEED,
        "weights": {p.value: n for p, n in SEED_WEIGHTS.items()},
        "n_cases": len(cases),
        "case_ids": [c.case_id for c in cases],
        "content_hash": seed_set_hash(cases),
    }
    path = path or _DEFAULT_MANIFEST
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def load_manifest(path: str | None = None) -> dict | None:
    path = path or _DEFAULT_MANIFEST
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_frozen_seed_set(path: str | None = None, strict: bool = True) -> tuple[list[SyntheticCase], dict]:
    """Rebuild the seed set and check it against the frozen manifest.

    ``strict`` raises when the regenerated content no longer matches the frozen
    hash — runs against a drifted set must not be compared with older notes.
    """
    manifest = load_manifest(path)
    cases = build_seed_set()
    if manifest is None:
        raise FileNotFoundError(
            "seed manifest not found; freeze it first (tribune quant-eval --freeze-seed)"
        )
    current = seed_set_hash(cases)
    if manifest["content_hash"] != current:
        message = (
            "quant seed set drifted: manifest hash "
            f"{manifest['content_hash'][:12]}… != current {current[:12]}…; "
            "results are not comparable with prior runs. Re-freeze deliberately."
        )
        if strict:
            raise RuntimeError(message)
        manifest = dict(manifest, drift_warning=message)
    return cases, manifest


# --------------------------------------------------------------------------- #
# Machine-Readable Task Requirement Specifications for Proactive Memory
# --------------------------------------------------------------------------- #

from dataclasses import dataclass, field


@dataclass
class SeedTaskSpecification:
    """Machine-readable requirement specification for a seed evaluation task."""

    task_id: str
    prompt: str
    expected_artifacts: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    expected_command_success_criteria: list[str] = field(default_factory=list)
    expected_verifier_conditions: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    environment_setup: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    case_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "expected_artifacts": self.expected_artifacts,
            "expected_files": self.expected_files,
            "expected_command_success_criteria": self.expected_command_success_criteria,
            "expected_verifier_conditions": self.expected_verifier_conditions,
            "completion_criteria": self.completion_criteria,
            "forbidden_actions": self.forbidden_actions,
            "environment_setup": self.environment_setup,
            "required_capabilities": self.required_capabilities,
            "case_id": self.case_id,
        }


def get_case_task_specification(case: SyntheticCase) -> SeedTaskSpecification:
    """Derive structured machine-readable task requirements for a synthetic case."""
    program_str = case.target_programs[0].value if case.target_programs else "snap"
    task_id = f"task_{case.case_id}_{program_str}"
    
    expected_files = [f"/workspace/data/{d.doc_id}.txt" for d in case.documents]
    expected_artifacts = [f"/workspace/output/{case.case_id}_determination.json"]

    expected_verifier_conds = [
        f"verify_{program_str}_statutory_criteria",
        f"validate_income_calculation",
    ]
    completion_crit = [
        f"Determination matches ground truth for program '{program_str}'",
        "Statutory citations verified in statutory corpus",
    ]
    forbidden = [
        "Unauthenticated network egress outside Sentinel allowlist",
        "Privilege escalation (sudo/su)",
        "Reading protected system files",
    ]

    prompt = (
        f"Perform statutory eligibility determination for applicant case {case.case_id} "
        f"in jurisdiction {case.jurisdiction} for program {program_str}."
    )

    return SeedTaskSpecification(
        task_id=task_id,
        prompt=prompt,
        expected_artifacts=expected_artifacts,
        expected_files=expected_files,
        expected_command_success_criteria=[f"exit_code == 0 on {program_str} verifier"],
        expected_verifier_conditions=expected_verifier_conds,
        completion_criteria=completion_crit,
        forbidden_actions=forbidden,
        environment_setup=["mkdir -p /workspace/output /workspace/data"],
        required_capabilities=["shell_execution", "statutory_rule_eval"],
        case_id=case.case_id,
    )


def build_task_specifications(cases: list[SyntheticCase] | None = None) -> list[SeedTaskSpecification]:
    """Build structured task specifications for all cases in the seed set."""
    target_cases = cases if cases is not None else build_seed_set()
    return [get_case_task_specification(c) for c in target_cases]


def to_task_requirements(spec: SeedTaskSpecification) -> list[Any]:
    """Convert SeedTaskSpecification into a list of TaskRequirement instances for MemoryStateBank."""
    from ...memory.state_bank import TaskRequirement

    requirements: list[TaskRequirement] = []
    # 1. Expected artifacts
    for art in spec.expected_artifacts:
        filename = os.path.basename(art)
        requirements.append(
            TaskRequirement(
                requirement_id=f"art_{spec.task_id}_{filename}",
                description=f"Generate artifact: {filename}",
                satisfied=False,
                severity="HIGH",
                missing_artifacts=[art],
            )
        )
    # 2. Verifier conditions
    for cond in spec.expected_verifier_conditions:
        requirements.append(
            TaskRequirement(
                requirement_id=f"cond_{spec.task_id}_{cond}",
                description=f"Pass verifier condition: {cond}",
                satisfied=False,
                severity="HIGH",
                failed_checks=[cond],
            )
        )
    return requirements

