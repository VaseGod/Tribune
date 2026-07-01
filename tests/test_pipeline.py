"""End-to-end: the core loop exhibits prepare, replan, and abstention behaviors,
emits only cited assertions, and writes a verifiable audit chain."""

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.orchestration.pipeline import CasePipeline
from tribune.types import ProgramId


def _run_demo():
    gen = SyntheticCaseGenerator(seed=7)
    cases = gen.generate_demo_set()
    pipe = CasePipeline()
    results = [pipe.run_case(c) for c in cases]
    return cases, results, pipe


def test_demo_has_abstention_prepare_and_replan():
    cases, results, _ = _run_demo()
    outcomes = [o for r in results for o in r.outcomes]
    assert any(o.abstained for o in outcomes), "expected at least one abstention"
    assert any(o.materials is not None for o in outcomes), "expected at least one prepared packet"
    assert any(o.replans > 0 for o in outcomes), "expected at least one REPLAN"


def test_all_assertions_are_cited():
    _, results, _ = _run_demo()
    for r in results:
        for o in r.outcomes:
            if o.assessment is not None and o.assessment.is_assertion:
                assert o.assessment.citations, f"{o.program} asserted without citations"


def test_no_confidently_wrong_on_demo():
    cases, results, _ = _run_demo()
    label = {"likely_eligible": "eligible", "likely_ineligible": "ineligible"}
    for case, result in zip(cases, results, strict=True):
        for program in case.target_programs:
            o = result.outcome_for(program)
            if o is None or o.abstained or o.assessment is None:
                continue
            predicted = label.get(o.assessment.status.value)
            truth = case.ground_truth[program].label.value
            assert predicted == truth, f"confidently wrong on {case.case_id}/{program.value}"


def test_audit_chain_verifies():
    cases, results, pipe = _run_demo()
    for case in cases:
        assert pipe.audit.verify_chain(case.case_id)


def test_coverage_gap_abstains():
    gen = SyntheticCaseGenerator(seed=3)
    case = gen.build_case(
        "t-gap", "NX",
        dict(age=42, has_dependent_child=False, disabled=False, pregnant=False, monthly_income=850.0),
        [ProgramId.MEDICAID],
    )
    assert case.ground_truth[ProgramId.MEDICAID].ambiguous
    result = CasePipeline().run_case(case)
    assert result.outcome_for(ProgramId.MEDICAID).abstained
