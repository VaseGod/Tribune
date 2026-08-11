"""Static Analysis & Validation Hooks for Security Inspection and Auto-Correction.

Passes generated code through static analysis (Semgrep CLI if available, with robust
built-in AST security scanner fallback). If vulnerabilities or CWE violations are found,
feeds findings into an automated retry loop to auto-correct code.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.llm_client import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    rule_id: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    message: str
    line_number: int = 0
    cwe: str | None = None


@dataclass
class StaticAnalysisReport:
    passed: bool
    findings: list[SecurityFinding] = field(default_factory=list)
    raw_output: str = ""

    def format_retry_prompt(self) -> str:
        if not self.findings:
            return ""
        lines = ["Static security analysis identified the following vulnerabilities in your generated code:"]
        for f in self.findings:
            cwe_str = f" [{f.cwe}]" if f.cwe else ""
            lines.append(f"- Line {f.line_number}: [{f.severity}] {f.rule_id}{cwe_str} - {f.message}")
        lines.append("\nPlease refactor the code to eliminate ALL security findings while preserving functionality.")
        return "\n".join(lines)


class ASTSecurityScanner(ast.NodeVisitor):
    """Fallback built-in AST security scanner checking for dangerous Python patterns."""

    def __init__(self) -> None:
        self.findings: list[SecurityFinding] = []
        self._formatted_vars: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, (ast.JoinedStr, ast.BinOp)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._formatted_vars.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for eval() / exec()
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                self.findings.append(
                    SecurityFinding(
                        rule_id="AST-EVAL-EXEC-INSECURE",
                        severity="HIGH",
                        message=f"Use of `{node.func.id}()` leads to arbitrary code execution / insecure deserialization.",
                        line_number=node.lineno,
                        cwe="CWE-95",
                    )
                )

        # Check for pickle.loads()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "loads" and isinstance(node.func.value, ast.Name) and node.func.value.id == "pickle":
                self.findings.append(
                    SecurityFinding(
                        rule_id="AST-PICKLE-DESERIALIZATION",
                        severity="HIGH",
                        message="Use of `pickle.loads()` allows arbitrary object execution.",
                        line_number=node.lineno,
                        cwe="CWE-502",
                    )
                )

        # Check subprocess shell=True
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.findings.append(
                        SecurityFinding(
                            rule_id="AST-SUBPROCESS-SHELL-TRUE",
                            severity="HIGH",
                            message="`subprocess` with `shell=True` creates Command Injection vulnerability.",
                            line_number=node.lineno,
                            cwe="CWE-78",
                        )
                    )

        # Check os.system()
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
            self.findings.append(
                SecurityFinding(
                    rule_id="AST-OS-SYSTEM-COMMAND-INJECTION",
                    severity="HIGH",
                    message="`os.system()` is vulnerable to Command Injection.",
                    line_number=node.lineno,
                    cwe="CWE-78",
                )
            )

        # Check SQL string formatting in execute calls: cursor.execute(f"...") or cursor.execute(formatted_var)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("execute", "executemany"):
            if node.args:
                first_arg = node.args[0]
                is_unformatted = isinstance(first_arg, (ast.JoinedStr, ast.BinOp))
                is_var_formatted = isinstance(first_arg, ast.Name) and first_arg.id in self._formatted_vars
                if is_unformatted or is_var_formatted:
                    self.findings.append(
                        SecurityFinding(
                            rule_id="AST-SQL-INJECTION-STRING-FORMAT",
                            severity="HIGH",
                            message="SQL query constructed using string formatting/f-strings; leads to SQL Injection.",
                            line_number=node.lineno,
                            cwe="CWE-89",
                        )
                    )

        self.generic_visit(node)


class StaticAnalysisValidator:
    """Runs static security checks via Semgrep (if installed) or AST security scanner."""

    def __init__(self, semgrep_bin: str = "semgrep", timeout_s: float = 30.0) -> None:
        self.semgrep_bin = semgrep_bin
        self.timeout_s = timeout_s

    def analyze_code(self, code: str) -> StaticAnalysisReport:
        # First try AST parsing
        try:
            tree = ast.parse(code)
            scanner = ASTSecurityScanner()
            scanner.visit(tree)
            ast_findings = scanner.findings
        except SyntaxError as err:
            return StaticAnalysisReport(
                passed=False,
                findings=[
                    SecurityFinding(
                        rule_id="AST-SYNTAX-ERROR",
                        severity="HIGH",
                        message=f"Code syntax error: {err}",
                        line_number=err.lineno or 1,
                    )
                ],
                raw_output=str(err),
            )

        # If Semgrep is available on system, run semgrep CLI check
        semgrep_path = shutil.which(self.semgrep_bin)
        if semgrep_path:
            semgrep_findings = self._run_semgrep(code, semgrep_path)
            all_findings = ast_findings + semgrep_findings
        else:
            all_findings = ast_findings

        passed = len(all_findings) == 0
        return StaticAnalysisReport(passed=passed, findings=all_findings)

    def _run_semgrep(self, code: str, semgrep_path: str) -> list[SecurityFinding]:
        findings: list[SecurityFinding] = []
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            cmd = [
                semgrep_path,
                "--config", "p/python",
                "--json",
                "--quiet",
                tmp_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            if res.returncode in (0, 1) and res.stdout:
                data = json.loads(res.stdout)
                for item in data.get("results", []):
                    extra = item.get("extra", {})
                    metadata = extra.get("metadata", {})
                    cwes = metadata.get("cwe", [])
                    cwe_val = cwes[0] if isinstance(cwes, list) and cwes else str(cwes)
                    findings.append(
                        SecurityFinding(
                            rule_id=item.get("check_id", "SEMGREP-SECURITY-CHECK"),
                            severity=extra.get("severity", "MEDIUM").upper(),
                            message=extra.get("message", "Potential security issue detected."),
                            line_number=item.get("start", {}).get("line", 1),
                            cwe=cwe_val if cwe_val else None,
                        )
                    )
        except Exception as exc:
            logger.debug(f"Semgrep execution skipped or failed: {exc}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return findings

    def validate_and_autocorrect(
        self,
        code: str,
        llm_provider: LLMProvider,
        max_retries: int = 2,
    ) -> tuple[str, bool, list[SecurityFinding]]:
        """Validate code and automatically retry refactoring with LLM if findings are found."""
        from ..providers.llm_client import LLMCompletionRequest
        from .secure_forge import SECURE_FORGE_SYSTEM_PROMPT

        current_code = code
        last_report = self.analyze_code(current_code)

        retries = 0
        while not last_report.passed and retries < max_retries:
            retries += 1
            logger.warning(f"Static analysis failed on attempt {retries}. Auto-correcting with LLM.")
            retry_prompt = (
                f"{last_report.format_retry_prompt()}\n\n"
                f"CURRENT CODE:\n```python\n{current_code}\n```\n\n"
                "Return ONLY the fixed, corrected Python code inside a ```python ``` block without explanations."
            )
            req = LLMCompletionRequest(
                system_prompt=SECURE_FORGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0.0,
            )
            resp = llm_provider.complete(req)
            new_content = resp.content.strip()

            # Extract code block if wrapped in markdown
            if "```python" in new_content:
                new_content = new_content.split("```python")[1].split("```")[0].strip()
            elif "```" in new_content:
                new_content = new_content.split("```")[1].split("```")[0].strip()

            current_code = new_content
            last_report = self.analyze_code(current_code)

        return current_code, last_report.passed, last_report.findings
