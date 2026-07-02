# Multilingual parity audit (equity gate)

A quality gap by language is a fairness defect. If the Spanish-language path
abstains more, over-refuses more, or costs more per completed verification,
Spanish-speaking claimants get a worse service from the same tool. TRIBUNE
treats parity deltas beyond threshold as **equity bugs** — first-class defects
filed under a distinct tracker label — not footnotes.

## Running it

```bash
tribune parity            # full 50-case seed set, EN + ES twins; exits 1 on equity bugs
tribune parity --n 10     # smaller run
tribune parity --out docs/eval_notes/parity_report.md
```

The audit runs parallel EN/ES versions of the same seed cases (Phase-2 seed set)
through the full pipeline and compares per language: token counts, effective
cost, abstention rate, kappa vs. gold, over-refusal rate, and false-confidence
rate.

## The ES twin: translation is scaffolded, never silent

`tribune/casegen/i18n/` builds a Spanish twin of each case: the document *text*
is rendered in Spanish via a glossary while the structured fields keep canonical
keys, so ground truth and ingestion stay identical — the twin differs only in
language.

**Every glossary entry is machine-drafted and starts `human_reviewed=False`.**
Legal/eligibility terms (`legal_term=True`) are listed in every parity report as
blockers: a bilingual navigator, benefits counselor, or legal-aid staffer must
review them before any Spanish-facing deployment. Flipping `human_reviewed` is a
deliberate, attributable change — never bulk-flip it.

## Thresholds are config, breaches are bugs

Thresholds live in
[tribune/eval/parity_thresholds.json](../tribune/eval/parity_thresholds.json)
(override with `TRIBUNE_PARITY_THRESHOLDS_PATH`). Token inflation alone is
expected (tokenizers encode Spanish ~1.33× less efficiently) and gets a looser
threshold; the equity signal is quality and cost.

A breach produces a structured `EquityBug` with the label **`equity-bug`** and a
ready-to-file title, and `tribune parity` exits non-zero. File each one in the
tracker under that label.

## Expected offline result

With the default deterministic provider the pipeline is language-neutral, so
EN/ES metrics match exactly (a test enforces this — an offline language-driven
outcome difference would be a pipeline bug). Real deltas appear when a served
model backend is configured; run the audit per backend you evaluate.
