# Contributing to TRIBUNE

Thank you for helping people get what they're legally entitled to. The most valuable contributions are **new program rule sets and jurisdictions**, **corrections to eligibility logic**, and **better synthetic cases** — especially the genuinely hard, ambiguous ones.

TRIBUNE's value is correctness and trust. A confidently-wrong eligibility result can cost someone food or shelter, so the bar for eligibility logic is deliberately high.

## Ground rules

- **Never weaken a safety property.** PRs that let an uncited eligibility claim be emitted, that auto-submit anything binding, that remove the human path, or that disable abstention will be rejected. These are enforced by tests in `tests/`.
- **No real PII, ever.** Only synthetic data in the repo, in issues, and in test fixtures.
- **Cite everything.** Every eligibility rule claim must point to a verifiable source.

## Adding a program rule set

No core code changes are required.

1. Create `tribune/corpus/programs/<program>.py` exporting:
   - `PROGRAM: ProgramId`
   - `RULESET: RuleSet` — a list of `Rule`s. Each `Rule` has:
     - a `criterion_id`, `description`, `required` flag,
     - the **citation** fields (`source`, `title`, `text`, `locator`, optional `effective_date`),
     - a pure **predicate** `(EvidenceView, JurisdictionProfile) -> CriterionOutcome` that returns `UNKNOWN` when the evidence needed to evaluate it is missing,
     - optional `margin_fn` (normalized distance-to-threshold, for calibration),
   - an optional `ambiguity_fn` on the `RuleSet` returning structural ambiguity signals from the evidence alone (e.g. a coverage gap, an unknown waitlist). These are the cases TRIBUNE must abstain on.
2. Register it in `tribune/corpus/programs/__init__.py`.
3. Create `tribune/casegen/programs/<program>.py` exporting `PROGRAM`, `RELEVANT_EVIDENCE`, and `ground_truth(situation, profile)` (use `generic_ground_truth` so labels are computed from the *same* predicates — do not duplicate eligibility logic). Register it in `tribune/casegen/programs/__init__.py`.
4. Add synthetic generation for the program in `tribune/casegen/synthetic.py` (`_tailor`) so the eval set includes clear-eligible, clear-ineligible, and ambiguous cases.
5. Add the program's document checklist to `tribune/agents/preparer.py`.

## Adding a jurisdiction

Add a `JurisdictionProfile` entry in `tribune/corpus/programs/jurisdictions.py`. Jurisdiction-specific thresholds (FPL, AMI, expansion status, appeal windows, etc.) live here so the same rule predicates serve every jurisdiction.

## The citation standard

- Cite the **controlling** authority (statute, regulation, or official policy), not a secondary summary.
- Include a `locator` (a URL or section pointer) so a human can read the original.
- Set `effective_date` when known — the canary's drift sentinel watches it.
- If a rule is genuinely state-specific and you are modeling an example, label it clearly (as the example unemployment code does).

## The review bar for eligibility logic

Eligibility logic should be reviewed by someone with domain knowledge (a navigator, benefits counselor, or legal-aid staffer) before it is trusted in a deployment. In review we look for:

- Correct, current thresholds and the correct controlling citation.
- Honest treatment of ambiguity — close calls and state-dependent edges should produce ambiguity signals so TRIBUNE abstains, not guesses.
- Honest treatment of the **eligibility ≠ access** gap (e.g. housing waitlists).
- A synthetic case set that actually exercises the ambiguous edges.

## Development

```bash
pip install -e ".[dev]"
ruff check tribune tests
pytest -q
tribune demo && tribune eval
```

All of the above must pass. New eligibility logic should come with tests and synthetic cases. Use the **"new program/jurisdiction"** issue template to propose work.
