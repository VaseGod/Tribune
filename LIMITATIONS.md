# Limitations & scope

Read this before deploying TRIBUNE anywhere near a real person's benefits.

## What TRIBUNE is not

- **Not legal advice.** TRIBUNE is an informational tool. It does not create an attorney–client relationship and is not a substitute for advice from a lawyer, accredited representative, or benefits counselor.
- **Not an official or binding determination.** Only the administering agency can determine eligibility. TRIBUNE surfaces what the governing rules say and what a situation suggests; the person, a navigator, or the agency decides.
- **Not a submission system.** TRIBUNE never auto-submits anything binding. It prepares materials; a human submits them.

## Known failure modes

- **Stale rules.** Benefit rules change frequently (income limits update annually; states change Medicaid and unemployment policy). The shipped rule sets and thresholds are **illustrative and modeled on 2024 federal figures** for two *example* jurisdictions. They are not the current, official rules of any real place. The canary's drift sentinel helps detect when cited logic diverges from a frozen baseline, but it cannot know the real-world rule changed — a human must keep the corpus current.
- **Retrieval recall.** Late-interaction retrieval can miss a governing rule; the verifier's coverage check is designed to catch this and trigger a replan, but coverage is only as complete as the rule set.
- **The model backend.** With the default deterministic `local_rules` provider, eligibility is exact rule application — there is no model "guessing," which is why offline evaluation shows near-perfect accuracy. A real served model introduces the very errors the false-confidence metric and verifier exist to catch; evaluate your specific model.
- **Synthetic ≠ real.** The synthetic generator produces internally consistent cases, but real applicants have messy, incomplete, contradictory documentation that no synthetic distribution fully captures.
- **English only**, single example jurisdiction set, no real agency API connectors, no UI. These are intentional v0 boundaries (see below).

## Eligibility is not access

For housing especially, qualifying for a program is **not** the same as receiving it. Vouchers and units are gated by waitlists that are frequently closed or years long. TRIBUNE represents waitlist status explicitly and abstains on the access question when status is unknown — but it cannot tell you when a closed waitlist will reopen.

## Privacy expectations

TRIBUNE is self-hostable so PII never has to leave the deploying organization. Ingested document text is scrubbed of obvious direct identifiers (SSN, email, phone, card numbers) before evidence is stored, but this is **not** a complete de-identification pipeline. Operators are responsible for their own data handling, retention, and consent practices. See [SECURITY.md](SECURITY.md).

## The human-in-the-loop requirement

TRIBUNE is built to *assist* a human, not replace one. Calibrated abstention, the always-offered human path, action-gating, and the inspectable audit log all assume a navigator/caseworker is in the loop for anything consequential. Do not deploy it as an autonomous decision-maker.

## Explicitly deferred (not implemented in v0)

A web/mobile UI, real agency API connectors, model fine-tuning, multilingual coverage beyond English, and hosted deployment. The code leaves clean extension points for these; they are not stubbed behind load-bearing logic.
