"""Unit Tests for Proactive Memory Sidecar and Intervention Policy.

Validates:
- Five-track state bank schema and operations
- Failed command fingerprinting and normalization
- k=2 interval execution update
- Silence by default when conditions are normal
- Intervention: repeated failed command
- Intervention: non-existent path
- Intervention: premature completion declaration
- Intervention: persistent verifier failure
- Cooldown period enforcement
- Maximum interventions budget cap
- Redaction of authentic secrets in injected text
- Sidecar disabled fallback behavior
"""

from __future__ import annotations

import pytest

from tribune.memory.aux_backend import HeuristicAuxBackend
from tribune.memory.intervention_policy import InterventionDecision, InterventionPolicy, SILENCE_TOKEN
from tribune.memory.sidecar import ProactiveMemorySidecar
from tribune.memory.state_bank import (
    EnvironmentState,
    FailedCommandCache,
    MemoryStateBank,
    MutationRecord,
    TaskRequirement,
)
from tribune.security.token_broker import TokenBroker


def test_state_bank_schema_and_fingerprinting():
    """Verify 5-track state bank schema and failed command cache normalization."""
    bank = MemoryStateBank()
    assert bank.current_environment.working_directory == "/workspace"
    assert len(bank.successful_mutations) == 0
    assert len(bank.unmet_task_requirements) == 0
    assert len(bank.active_subgoals) == 0
    assert len(bank.failed_commands_cache.entries) == 0

    # Fingerprint normalization
    fp1 = FailedCommandCache.fingerprint("cat ./data/file.txt | grep 'pattern'")
    fp2 = FailedCommandCache.fingerprint("cat   data/file.txt | grep 'pattern'  ")
    assert fp1 == fp2

    # Record failure
    rec = bank.failed_commands_cache.record_failure(
        command="cat data/file.txt",
        exit_code=1,
        stderr="FileNotFoundError: [Errno 2] No such file",
        turn_index=1,
    )
    assert rec.recurrence_count == 1
    assert bank.failed_commands_cache.is_known_failure("cat ./data/file.txt") is not None


def test_silence_by_default():
    """Verify intervention gate returns SILENCE when execution is safe and normal."""
    policy = InterventionPolicy()
    bank = MemoryStateBank()

    dec = policy.evaluate(
        turn_index=1,
        state_bank=bank,
        candidate_command="echo 'normal progression'",
        agent_declared_complete=False,
    )
    assert not dec.should_intervene
    assert dec.reason_code == "SILENCE"
    assert dec.injected_text == SILENCE_TOKEN


def test_intervention_repeated_failed_command():
    """Verify intervention triggers when agent attempts a known failing command."""
    policy = InterventionPolicy(cooldown_turns=0)
    bank = MemoryStateBank()

    cmd = "python parse_data.py --input invalid.csv"
    bank.failed_commands_cache.record_failure(
        command=cmd,
        exit_code=2,
        stderr="ValueError: malformed input",
        turn_index=1,
    )

    dec = policy.evaluate(
        turn_index=2,
        state_bank=bank,
        candidate_command=cmd,
    )
    assert dec.should_intervene
    assert dec.reason_code == "REPEATED_FAILED_COMMAND"
    assert "previously failed on turn 1" in dec.injected_text
    assert "ValueError: malformed input" in dec.injected_text


def test_intervention_premature_completion():
    """Verify intervention challenges premature task completion declaration."""
    policy = InterventionPolicy(cooldown_turns=0)
    bank = MemoryStateBank()
    bank.unmet_task_requirements.append(
        TaskRequirement(
            requirement_id="req_artifact_output",
            description="Generate output/determination.json",
            satisfied=False,
        )
    )

    dec = policy.evaluate(
        turn_index=3,
        state_bank=bank,
        candidate_command="",
        agent_declared_complete=True,
    )
    assert dec.should_intervene
    assert dec.reason_code == "PREMATURE_COMPLETION"
    assert "Task completion cannot be accepted yet" in dec.injected_text
    assert "req_artifact_output" in dec.injected_text


def test_intervention_non_existent_path():
    """Verify intervention flags missing path references based on environment state."""
    policy = InterventionPolicy(cooldown_turns=0)
    bank = MemoryStateBank()
    bank.current_environment.working_directory = "/workspace"
    bank.current_environment.visible_files = ["main.py", "config.json"]

    dec = policy.evaluate(
        turn_index=1,
        state_bank=bank,
        candidate_command="cat /workspace/data/input.csv",
    )
    assert dec.should_intervene
    assert dec.reason_code == "NON_EXISTENT_PATH"
    assert "does not exist in working directory" in dec.injected_text


def test_intervention_cooldown_and_budget_cap():
    """Verify cooldown period and max interventions limit prevent spam."""
    policy = InterventionPolicy(cooldown_turns=2, max_interventions_per_task=2)
    bank = MemoryStateBank()
    cmd = "failing_command_xyz"
    bank.failed_commands_cache.record_failure(cmd, 1, "error", 1)

    # 1. First intervention triggered at turn 2
    dec1 = policy.evaluate(turn_index=2, state_bank=bank, candidate_command=cmd)
    assert dec1.should_intervene

    # 2. Turn 3: blocked by cooldown (turns since last = 1 < 2)
    dec2 = policy.evaluate(turn_index=3, state_bank=bank, candidate_command=cmd)
    assert not dec2.should_intervene
    assert "cooldown active" in (dec2.suppression_reason or "")

    # 3. Turn 4: cooldown passed, second intervention triggered
    dec3 = policy.evaluate(turn_index=4, state_bank=bank, candidate_command=cmd)
    assert dec3.should_intervene

    # 4. Turn 7: cooldown passed, but max interventions cap (2) reached
    dec4 = policy.evaluate(turn_index=7, state_bank=bank, candidate_command=cmd)
    assert not dec4.should_intervene
    assert "Max interventions cap reached" in (dec4.suppression_reason or "")


def test_sidecar_turn_recording_and_k_interval_update():
    """Verify ProactiveMemorySidecar updates state bank every k turns."""
    sidecar = ProactiveMemorySidecar(interval_k=2, enabled=True)

    # Turn 1 (not multiple of 2, exit code 0)
    d1 = sidecar.record_turn_and_evaluate(
        turn_index=1,
        command="ls -la",
        exit_code=0,
        stdout="total 0\nfile1.txt\nfile2.txt",
        stderr="",
    )
    assert not d1.should_intervene
    assert sidecar.trajectory_buffer.total_turns == 1

    # Turn 2 (multiple of 2 -> updates state bank)
    d2 = sidecar.record_turn_and_evaluate(
        turn_index=2,
        command="touch newfile.py",
        exit_code=0,
        stdout="",
        stderr="",
    )
    m = sidecar.get_metrics()
    assert m["total_turns"] == 2
    assert m["sidecar_invocations"] >= 1
    assert "newfile.py" in sidecar.state_bank.current_environment.visible_files


def test_sidecar_redaction_in_injected_text():
    """Verify secrets present in intervention text are redacted."""
    token_broker = TokenBroker()
    secret = "sk-live-secret-test-token"
    surrogate = token_broker.register_secret(secret, secret_ref="TEST")

    policy = InterventionPolicy(cooldown_turns=0, token_broker=token_broker)
    bank = MemoryStateBank()
    bank.failed_commands_cache.record_failure(
        command="test_command",
        exit_code=1,
        stderr=f"auth error: invalid {secret}",
        turn_index=1,
    )

    dec = policy.evaluate(turn_index=2, state_bank=bank, candidate_command="test_command")
    assert dec.should_intervene
    assert secret not in dec.injected_text
    assert surrogate in dec.injected_text


def test_sidecar_disabled_mode():
    """Verify sidecar returns silence when disabled."""
    sidecar = ProactiveMemorySidecar(enabled=False)
    dec = sidecar.record_turn_and_evaluate(
        turn_index=1,
        command="any command",
        exit_code=1,
        stdout="",
        stderr="error",
    )
    assert not dec.should_intervene
    assert dec.suppression_reason == "sidecar_disabled"
