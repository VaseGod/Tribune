"""Tests for explicit redaction disclaimers and failure-preserving masking."""

import pytest
from tribune.security.sanitization import RedactionDisclaimer, redact_sensitive_data


def test_pii_redaction_with_stable_markers():
    raw_text = (
        "Applicant John Doe has SSN 123-45-6789, email john.doe@example.com, "
        "and contact phone (555) 123-4567."
    )

    clean_text, disclaimer = redact_sensitive_data(raw_text)

    # 1. Stable markers present
    assert "[REDACTED_PII: SSN]" in clean_text
    assert "[REDACTED_PII: EMAIL]" in clean_text
    assert "[REDACTED_PII: PHONE]" in clean_text

    # 2. Raw PII absent
    assert "123-45-6789" not in clean_text
    assert "john.doe@example.com" not in clean_text
    assert "(555) 123-4567" not in clean_text

    # 3. Explicit disclaimer present
    assert disclaimer.has_redactions is True
    assert disclaimer.pii_count == 3
    assert "[REDACTION DISCLAIMER:" in clean_text


def test_business_records_redaction():
    raw_text = "Employer Corp EIN 12-3456789 reported gross_revenue: $1,250,000.00."

    clean_text, disclaimer = redact_sensitive_data(raw_text)

    assert "[REDACTED_BUSINESS_RECORD: EIN]" in clean_text
    assert "[REDACTED_BUSINESS_RECORD: FINANCIALS]" in clean_text
    assert "12-3456789" not in clean_text
    assert disclaimer.business_records_count == 2


def test_failure_evidence_strictly_preserved():
    raw_text_with_failure = (
        "Applicant SSN 111-22-3333.\n"
        "AssertionError: Expected SNAP eligibility but got denial.\n"
        "Traceback (most recent call last):\n"
        "  File 'pipeline.py', line 42, in run_case\n"
        "    assert result.eligible is True\n"
        "=== [VERIFIER_GATE_REJECTION] ===\n"
        "GATE STATUS: FAILED\n"
        "=================================\n"
    )

    clean_text, disclaimer = redact_sensitive_data(raw_text_with_failure, preserve_failure_evidence=True)

    # PII was masked
    assert "[REDACTED_PII: SSN]" in clean_text
    assert "111-22-3333" not in clean_text

    # Critical failure evidence MUST remain intact and visible!
    assert "AssertionError: Expected SNAP eligibility but got denial." in clean_text
    assert "Traceback (most recent call last):" in clean_text
    assert "[VERIFIER_GATE_REJECTION]" in clean_text
    assert len(disclaimer.preserved_failure_signatures) >= 2


def test_no_redaction_when_no_sensitive_data():
    raw_text = "Standard legal notice confirming filing of appeal on 2026-09-01."
    clean_text, disclaimer = redact_sensitive_data(raw_text)

    assert clean_text == raw_text
    assert disclaimer.has_redactions is False
    assert "[REDACTION DISCLAIMER" not in clean_text
