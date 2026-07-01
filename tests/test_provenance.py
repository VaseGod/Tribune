"""Anonymization scrubs direct identifiers at ingest."""

from tribune.corpus.provenance import anonymize, content_hash


def test_anonymize_scrubs_pii():
    text = "Name on file. SSN 123-45-6789, email a.b@example.com, phone 555-123-4567."
    clean, found = anonymize(text)
    assert "123-45-6789" not in clean
    assert "a.b@example.com" not in clean
    assert "555-123-4567" not in clean
    assert {"SSN", "EMAIL", "PHONE"}.issubset(set(found))


def test_anonymize_noop_when_clean():
    text = "household_size: 3\nmonthly_income: 1500"
    clean, found = anonymize(text)
    assert clean == text
    assert found == []


def test_content_hash_is_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
