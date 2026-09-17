"""Tests for the cryptographically hashed execution trace ledger."""

import json
import tempfile
import pytest
from tribune.instrumentation.tracing import (
    GENESIS_HASH,
    HashedTraceEvent,
    HashedTraceLedger,
)
from tribune.instrumentation.verify_traces import verify_trace_file


def test_hashed_trace_chain_validates():
    ledger = HashedTraceLedger(ledger_id="test_chain")

    # Step 1
    evt1 = ledger.record_step(
        task_id="task_001",
        run_id="run_001",
        step_type="INFERENCE",
        model_provider="openai_compatible",
        model_id="deepseek-v4.1-flash",
        tier="worker",
        token_usage={"prompt_tokens": 100, "completion_tokens": 20},
        cost_estimate_usd=0.0001,
        outcome="SUCCESS",
    )
    assert evt1.previous_hash == GENESIS_HASH
    assert len(evt1.current_hash) == 64

    # Step 2
    evt2 = ledger.record_step(
        task_id="task_001",
        run_id="run_001",
        step_type="TOOL",
        tool_name="search_statute",
        tool_inputs={"program": "SNAP"},
        outcome="SUCCESS",
    )
    assert evt2.previous_hash == evt1.current_hash

    # Step 3
    evt3 = ledger.record_step(
        task_id="task_001",
        run_id="run_001",
        step_type="VERIFIER",
        verifier_result={"is_valid": True, "status": "VALID"},
        outcome="SUCCESS",
    )
    assert evt3.previous_hash == evt2.current_hash

    # Verify integrity
    is_valid, error = ledger.verify_integrity()
    assert is_valid is True
    assert error is None


def test_tampered_trace_fails_cryptographic_verification():
    ledger = HashedTraceLedger(ledger_id="test_tamper")

    ledger.record_step(task_id="t1", run_id="r1", step_type="INFERENCE", outcome="SUCCESS")
    ledger.record_step(task_id="t1", run_id="r1", step_type="TOOL", outcome="SUCCESS")
    ledger.record_step(task_id="t1", run_id="r1", step_type="VERIFIER", outcome="SUCCESS")

    assert ledger.verify_integrity()[0] is True

    # Tamper with the middle event's outcome without updating hash
    ledger.events[1].outcome = "FAILED_TAMPERED"

    is_valid, error = ledger.verify_integrity()
    assert is_valid is False
    assert "tampering detected" in error.lower()


def test_broken_hash_link_fails_verification():
    ledger = HashedTraceLedger(ledger_id="test_link")

    ledger.record_step(task_id="t1", run_id="r1", step_type="INFERENCE")
    ledger.record_step(task_id="t1", run_id="r1", step_type="TOOL")

    # Manually corrupt previous_hash link
    ledger.events[1].previous_hash = "deadbeef" * 8

    is_valid, error = ledger.verify_integrity()
    assert is_valid is False
    assert "broken" in error.lower()


def test_verify_trace_file_utility():
    ledger = HashedTraceLedger(ledger_id="cli_test")
    ledger.record_step(task_id="t_cli", run_id="r_cli", step_type="INFERENCE", outcome="SUCCESS")
    ledger.record_step(task_id="t_cli", run_id="r_cli", step_type="VERIFIER", outcome="SUCCESS")

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
        ledger.export_json(f.name)
        f_path = f.name

    # Validate file
    valid, err, data = verify_trace_file(f_path)
    assert valid is True
    assert len(data["events"]) == 2

    # Now tamper with the file directly on disk
    with open(f_path, "r") as rf:
        file_data = json.load(rf)
    file_data["events"][0]["outcome"] = "ALTERED_ON_DISK"
    with open(f_path, "w") as wf:
        json.dump(file_data, wf)

    valid_tampered, err_tampered, _ = verify_trace_file(f_path)
    assert valid_tampered is False
    assert "tampering" in err_tampered.lower()
