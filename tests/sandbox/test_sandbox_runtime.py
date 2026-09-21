"""Unit and integration tests for ShellSandboxExecutor, modes, timeouts, and token reduction."""

from __future__ import annotations

import tempfile
import unittest

from sandbox.container_policy import SandboxPolicy
from sandbox.sandbox_runtime import (
    RuntimeMode,
    ShellSandboxExecutor,
)


class TestShellSandboxRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.policy = SandboxPolicy(
            allow_shell=True,
            allow_network=False,
            max_execution_seconds=5,
            max_output_bytes=2048,
        )
        self.executor = ShellSandboxExecutor(
            policy=self.policy,
            mode=RuntimeMode.SHELL_FIRST,
            workspace_dir=self.tmp_dir.name,
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_shell_first_execution_success(self) -> None:
        """Verify standard shell execution succeeds with expected output and artifacts."""
        cmd = "echo 'LINE1' > output.txt && cat output.txt"
        res = self.executor.execute_shell(cmd)

        self.assertEqual(res.exit_code, 0)
        self.assertIn("LINE1", res.stdout)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.isolation_mode, "shell_first")
        self.assertTrue(any("output.txt" in a for a in res.collected_artifacts))
        self.assertGreater(res.metadata.get("tokens_consumed", 0), 0)

    def test_shell_policy_denial(self) -> None:
        """Dangerous shell command triggers policy denial before process launch."""
        cmd = "sudo rm -rf /etc"
        res = self.executor.execute_shell(cmd)

        self.assertEqual(res.exit_code, 126)
        self.assertIn("security deny pattern", res.stderr)
        self.assertEqual(res.isolation_mode, "policy_denial")

    def test_shell_timeout_kill_switch(self) -> None:
        """Process exceeding execution timeout is cleanly killed with -9."""
        cmd = "python3 -c 'import time; time.sleep(10)'"
        res = self.executor.execute_shell(cmd, timeout_s=1.0)

        self.assertTrue(res.timed_out)
        self.assertTrue(res.killed)
        self.assertEqual(res.exit_code, -9)

    def test_catalog_only_mode(self) -> None:
        """When in CATALOG_ONLY mode, shell execution is strictly blocked."""
        catalog_executor = ShellSandboxExecutor(
            policy=self.policy,
            mode=RuntimeMode.CATALOG_ONLY,
            workspace_dir=self.tmp_dir.name,
        )
        res = catalog_executor.execute_shell("echo 'hello'")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("CATALOG_ONLY", res.stderr)

    def test_catalog_tool_execution(self) -> None:
        """Typed programmatic tool executes and returns audited structured output."""
        res = self.executor.execute_catalog_tool(
            "query_statutory_table",
            {"program": "snap", "household_size": 2, "year": 2026},
        )
        self.assertEqual(res.exit_code, 0)
        self.assertIn("gross_monthly_income_limit", res.stdout)
        self.assertIn("audited", res.stdout)

    def test_token_comparison_stats(self) -> None:
        """Shell-first execution achieves token reduction vs verbose catalog tool calls."""
        # 1. Shell command: compact (~15 tokens)
        self.executor.execute_shell("echo '1450'")
        # 2. Catalog tool: verbose schema (~250-300 tokens)
        self.executor.execute_catalog_tool(
            "query_statutory_table",
            {"program": "snap", "household_size": 1},
        )

        stats = self.executor.get_token_comparison_stats()
        self.assertGreater(stats["total_catalog_tokens"], stats["total_shell_tokens"])
        self.assertGreater(stats["token_reduction_ratio"], 0.50)


if __name__ == "__main__":
    unittest.main()
