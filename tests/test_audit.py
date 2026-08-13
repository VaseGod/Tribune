"""Unit tests for AuditLogger deep trace sanitization."""

from tribune.governance.audit import AuditLog
from tribune.types import SMState


def test_audit_logger_pii_and_credential_sanitization():
    audit_log = AuditLog()

    case_id = "case_sanitization_test"
    sensitive_action = "Processing applicant SSN 123-45-6789 and email applicant@example.com"
    sensitive_payload = {
        "ssn": "123-45-6789",
        "email": "user@domain.org",
        "phone": "555-123-4567",
        "benefit_id": "BEN-987654",
        "api_key": "sk-1234567890abcdef1234567890abcdef",
        "bearer": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature",
        "notes": "Contact phone is 555-987-6543 for SSN 987-65-4321",
    }

    record = audit_log.append(
        case_id=case_id,
        state=SMState.ASSESS,
        agent="eligibility_proposer",
        action=sensitive_action,
        payload=sensitive_payload,
    )

    # Verify action is sanitized
    assert "123-45-6789" not in record.action
    assert "[REDACTED]" in record.action

    # Verify payload fields are sanitized
    assert record.payload["ssn"] == "[REDACTED]"
    assert record.payload["email"] == "[REDACTED]"
    assert record.payload["phone"] == "[REDACTED]"
    assert record.payload["benefit_id"] == "[REDACTED]"
    assert record.payload["api_key"] == "[REDACTED]"
    assert record.payload["bearer"] == "[REDACTED]"
    assert "987-65-4321" not in record.payload["notes"]
    assert "[REDACTED]" in record.payload["notes"]

    # Verify hash chain integrity
    assert audit_log.verify_chain(case_id) is True
