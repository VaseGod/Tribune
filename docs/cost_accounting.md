# Cost & turn-efficiency accounting

TRIBUNE's honest unit of account is **cost per completed verification**, never
per-token list price. A model that is cheaper per token but needs 40% more output
tokens and 3× the agentic turns is *more* expensive per completed task — and that
is the number a deploying clinic or legal-aid office actually pays.

## What is measured

Every `(case, program)` task accumulates a `TaskUsage`
([tribune/types.py](../tribune/types.py)) attached to its `ProgramOutcome`:

| field | meaning |
| --- | --- |
| `tokens_input` / `tokens_output` | per model call and per task total |
| `cache_read_tokens` / `cache_write_tokens` | prompt-cache traffic, when the backend reports it |
| `turns`, `proposer_turns`, `verifier_turns` | proposer/verifier round-trips (REPLANs add turns) |
| `tokenizer_ids` | which tokenizer produced the counts (per-language comparisons need this) |
| `estimated` | counts came from the deterministic offline estimator, not a serving backend |
| `cost_usd`, `cost_backend_id` | priced against the cost model at the accounting date |
| `language` | the case language ("en", "es", …) for per-language token accounting |

Recording is wired through `UsageRecorder`
([tribune/instrumentation/usage.py](../tribune/instrumentation/usage.py)): the
providers report per-call token usage (served backends return a `usage` block;
the offline provider uses a deterministic estimator flagged `estimated`), and the
state machine reports turns. When Weave tracing is enabled, each finished task
emits a `task_usage` event.

## Pricing is data, not code

Rates live in [tribune/eval/pricing.json](../tribune/eval/pricing.json)
(override with `TRIBUNE_PRICING_PATH`). Two kinds:

- **api** — $/M tokens for input / output, plus optional cache-read/cache-write
  rates,
- **self_hosted** — amortized `$ per GPU-hour ÷ measured throughput (tokens/s)`;
  replace the placeholder throughput with numbers measured on your hardware.

Every rate carries `effective_from` / `effective_until`, so **promotional prices
expire in the data**, not in code. Example: Claude Sonnet 5 at $2/$10 per M
through 2026-08-31, reverting to $3/$15 on 2026-09-01. Set
`TRIBUNE_PRICING_DATE=YYYY-MM-DD` to pin the accounting date for reproducible
reports (default: today).

Matching is first-pattern-wins over the provider's model name, so keep specific
patterns above catch-alls in the file.

## Accounting rules (invariant: abstention is a success)

Every task outcome is one of four **completed** types
(`tribune/eval/metrics.py::TaskOutcomeType`):

- `correct_determination` — asserted, and agrees with ground truth,
- `correct_abstention` — abstained on a genuinely ambiguous case. **This is a
  success outcome, reported at its actual (low) cost — never a failure, never
  infinite cost.** When the determinative fact lives with the claimant or a
  caseworker, routing to a human is correct behavior.
- `incorrect_determination` — asserted and wrong (the cardinal harm),
- `over_refusal` — abstained on a clear case (safe, but a missed opportunity).

`tribune eval` prints a cost report broken down by outcome type — mean/median
cost per task, turns, token totals — plus per-language token counts.
`mean cost per successful outcome` averages over correct determinations **and**
correct abstentions; that set is pinned in
`tribune/eval/costreport.py::SUCCESS_OUTCOMES` and tests enforce it.

## Notes

- The default offline provider is free (`local:rules` backend, $0), so the whole
  accounting path runs — and is tested — without any paid service.
- All new eval fields are additive; pre-existing eval outputs and tests are
  unchanged.
