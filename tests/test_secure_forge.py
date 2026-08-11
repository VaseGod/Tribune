"""Unit tests for SecureForge system prompt wrapping and static analysis validation."""

from __future__ import annotations

import unittest

from tribune.providers.llm_client import LLMCompletionRequest, LLMCompletionResponse, LocalRulesLLMAdapter
from tribune.security.secure_forge import SECURE_FORGE_SYSTEM_PROMPT, SecureForge
from tribune.security.static_analysis import SecurityFinding, StaticAnalysisValidator


class MockLLMProvider:
    def __init__(self, fixed_response: str) -> None:
        self.name = "mock_provider"
        self.fixed_response = fixed_response

    def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            content=self.fixed_response,
            model="mock-model",
            provider_name=self.name,
            input_tokens=10,
            output_tokens=10,
        )


class TestSecureForgeAndStaticAnalysis(unittest.TestCase):
    def test_secure_forge_prompt_wrapping(self) -> None:
        sf = SecureForge(base_system_prompt="Base app prompt")
        wrapped = sf.wrap_prompt("Write a database query function", task_context="Module: db.py")

        self.assertIn(SECURE_FORGE_SYSTEM_PROMPT, wrapped["system"])
        self.assertIn("Base app prompt", wrapped["system"])
        self.assertIn("[CONTEXT GRAPH & REPO ARCHITECTURE]", wrapped["user"])
        self.assertIn("Write a database query function", wrapped["user"])

    def test_ast_security_scanner_flags_eval(self) -> None:
        insecure_code = """
def run_user_code(user_input):
    result = eval(user_input)
    return result
"""
        validator = StaticAnalysisValidator()
        report = validator.analyze_code(insecure_code)
        self.assertFalse(report.passed)
        self.assertTrue(any(f.rule_id == "AST-EVAL-EXEC-INSECURE" for f in report.findings))

    def test_ast_security_scanner_flags_sql_injection(self) -> None:
        insecure_sql = """
def search_user(cursor, username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
"""
        validator = StaticAnalysisValidator()
        report = validator.analyze_code(insecure_sql)
        self.assertFalse(report.passed)
        self.assertTrue(any(f.rule_id == "AST-SQL-INJECTION-STRING-FORMAT" for f in report.findings))

    def test_ast_security_scanner_flags_subprocess_shell_true(self) -> None:
        insecure_sub = """
import subprocess
def run_command(cmd):
    subprocess.run(cmd, shell=True)
"""
        validator = StaticAnalysisValidator()
        report = validator.analyze_code(insecure_sub)
        self.assertFalse(report.passed)
        self.assertTrue(any(f.rule_id == "AST-SUBPROCESS-SHELL-TRUE" for f in report.findings))

    def test_ast_security_scanner_passes_safe_code(self) -> None:
        safe_code = """
import sqlite3
def get_user_safe(cursor, user_id: int):
    cursor.execute("SELECT id, name FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
"""
        validator = StaticAnalysisValidator()
        report = validator.analyze_code(safe_code)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.findings), 0)

    def test_validate_and_autocorrect_retry_loop(self) -> None:
        insecure_code = "eval('1 + 1')"
        safe_refactored_code = "```python\nimport ast\nast.literal_eval('1 + 1')\n```"
        mock_provider = MockLLMProvider(safe_refactored_code)

        validator = StaticAnalysisValidator()
        code_res, passed, findings = validator.validate_and_autocorrect(
            insecure_code, mock_provider, max_retries=1
        )
        self.assertTrue(passed)
        self.assertIn("literal_eval", code_res)


if __name__ == "__main__":
    unittest.main()
