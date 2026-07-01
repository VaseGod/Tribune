## What does this PR do?

## Type of change

- [ ] New program rule set / jurisdiction
- [ ] Correction to eligibility logic or a citation
- [ ] Synthetic cases / evaluation
- [ ] Core code (orchestration, agents, providers, governance, etc.)
- [ ] Docs

## Safety checklist (required)

- [ ] No real PII added anywhere (code, tests, fixtures, docs)
- [ ] Every new eligibility rule claim carries a verifiable citation
- [ ] No safety property weakened: uncited claims still impossible; nothing binding auto-submits; the human path remains; abstention still fires on uncertainty
- [ ] Ambiguous / state-dependent edges produce abstention rather than a guess
- [ ] `eligibility ≠ access` handled honestly where relevant (e.g. waitlists)

## Verification

- [ ] `ruff check tribune tests` passes
- [ ] `pytest -q` passes
- [ ] `tribune demo` and `tribune eval` run clean and exit 0

## Domain review

For eligibility-logic changes: who reviewed the rules/citations for correctness?
