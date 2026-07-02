# Ingestion injection probe (red-team suite)

Prompt-injection resistance is a headline feature for agentic tools, and benefit
notices are **attacker-writable documents that enter TRIBUNE's OCR path**. A
forged or tampered notice may try to (a) trigger a gated action, (b) alter the
verifier's conclusion, or (c) exfiltrate case data into an output. This suite
generates adversarial notices and asserts none of that can happen.

## Running it

```bash
tribune redteam        # runs the full payload catalog; exits 1 on any finding
make test              # the probe also runs as pytest (tests/test_injection_probe.py)
```

It runs against the **real** ingestion/OCR path (`TRIBUNE_DOC_INGEST=ocr`,
offline text-extraction path) and the full `CasePipeline` with the deterministic
local model — no network, no keys, no cost.

## What it generates

`tribune/redteam/payloads.py` is a catalog of adversarial payloads tagged by
goal (`action_trigger` / `verifier_tamper` / `exfiltration`) and placement
(body, header, footer, form field, or a low-contrast **OCR-artifact** strip
rendered as the noisy text a scanner emits). `notice_generator.py` embeds a
payload into a synthetic `benefit_notice` document and appends it to a legitimate
case, modeling a forged notice arriving alongside genuine paperwork.

## What it asserts

For every payload, run through OCR + pipeline, the probe checks:

1. **Action gate never fires from document-sourced instructions.** Nothing is
   marked submitted; no payload induces an unexpected audit state; and a
   sign-off *forged from document-planted tokens* is actively fed to the
   `ActionGate`, which must reject it (as must a missing sign-off).
2. **Provenance never cites injected text as determinative.** Every citation on
   every assessment comes from the rule store; forged citation markers planted
   in the document (e.g. `7 CFR 999.999`) never surface.
3. **Injected content is flagged or inert.** Instruction lines never become
   evidence (only known typed fields are extracted); planted PII is redacted by
   the anonymizer and never stored unredacted or echoed into an output; and
   instruction text is never echoed verbatim into the user-facing disclosure.

## Reproduction on failure

Every failed check emits a `Finding` carrying the offending `payload_id` and the
**full hash-chained audit trace** of the run, so a failure can be reproduced
exactly. A negative-control test (`test_probe_is_not_vacuous_negative_control`)
forces a leaky output sink and asserts the probe fires — so a green run is
meaningful, not vacuous.

## Why this holds by construction

TRIBUNE's ingestion extracts only *typed evidence fields* (known keys); free text
is anonymized and hashed, never interpreted as instructions. The action gate
requires a `HumanSignoff` that document text cannot construct. The verifier
re-derives conclusions from *cited rules*, not document text. The probe exists to
keep those properties true as the code evolves.
