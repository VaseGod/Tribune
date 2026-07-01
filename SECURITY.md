# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities **privately**. Do not open a public issue for a security report.

- Use GitHub's **"Report a vulnerability"** (Security → Advisories) on this repository, or
- email the maintainers at the address listed in the repository's organization profile.

Include a description, reproduction steps, affected versions/commit, and impact. We aim to acknowledge within a few business days and to coordinate disclosure with you once a fix is available.

## PII-handling expectations

TRIBUNE handles information about real people's income, household, immigration status, health, and housing. Treat all of it as sensitive.

- **Self-host.** TRIBUNE is designed so PII never has to leave the deploying organization's infrastructure. Do not point the default configuration at a third-party model endpoint that you do not control if you are processing real PII.
- **Minimize and scrub.** Ingested document text is scrubbed of obvious direct identifiers (SSN, email, phone, payment-card numbers) before evidence is persisted (`tribune/corpus/provenance.py`). This is a safeguard, not a complete de-identification system — do not rely on it alone.
- **No PII in the repo.** Never commit real applicant data, screenshots containing PII, or logs containing PII. Issues and test fixtures must use synthetic data only. The synthetic generator (`tribune/casegen/`) exists for exactly this.
- **Access control.** Case memory is partitioned and access-controlled per case (`tribune/memory/`). Preserve this isolation in any storage backend you add.
- **Audit.** The hash-chained audit log (`tribune/governance/audit.py`) is there so access and reasoning can be inspected; protect it accordingly.

## Responsible-use note

A benefits tool that is confidently wrong can cause real harm. The safety properties in this project (citation enforcement, calibrated abstention, action-gating, the human path) are security-relevant. Changes that weaken them will be treated as security issues.
