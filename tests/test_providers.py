"""Unit tests for Provider Layer Payload Sanitization and HMAC Session State Binding."""

import pytest

from tribune.config import TribuneSettings
from tribune.providers.base import bind_session_state, verify_session_state
from tribune.providers.openai_compat import OpenAICompatProvider


def test_sanitize_reasoning_payload():
    settings = TribuneSettings(openai_api_key="test-key")
    provider = OpenAICompatProvider(model="test-model", settings=settings)

    dirty_payload = {
        "model": "test-model",
        "encrypted_content": "SGVsbG8gV29ybGQ=",
        "thought_signature": "sig_abc123",
        "raw_base64_blobs": "data:text/plain;base64,12345",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
                "encrypted_content": "secret",
                "thought_signature": "sig1",
            },
            {
                "role": "assistant",
                "content": "Response",
                "reasoning_content": "internal monologue",
            },
        ],
        "extra_body": {
            "thought": "opaque thought",
            "valid_param": 123,
        },
    }

    clean = provider.complete(dirty_payload)

    assert "encrypted_content" not in clean
    assert "thought_signature" not in clean
    assert "raw_base64_blobs" not in clean
    assert len(clean["messages"]) == 2
    assert "encrypted_content" not in clean["messages"][0]
    assert "thought_signature" not in clean["messages"][0]
    assert "reasoning_content" not in clean["messages"][1]
    assert clean["messages"][0]["content"] == "Hello"
    assert clean["messages"][1]["content"] == "Response"
    assert "thought" not in clean["extra_body"]
    assert clean["extra_body"]["valid_param"] == 123


def test_hmac_session_state_binding():
    key = "secret_session_key_123"
    state = {"session_id": "sess_456", "user_id": "usr_789", "role": "admin"}

    signed = bind_session_state(state, key)
    assert "_hmac_signature" in signed

    verified = verify_session_state(signed, key)
    assert verified == state

    # Test tampering
    tampered = dict(signed)
    tampered["role"] = "superadmin"

    with pytest.raises(ValueError, match="Session state HMAC signature verification failed"):
        verify_session_state(tampered, key)

    # Test replayed state without signature
    with pytest.raises(ValueError, match="Session state block replayed without a valid session key signature"):
        verify_session_state(state, key)
