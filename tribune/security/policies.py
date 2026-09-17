"""Anti-Reward-Hacking & Dynamic Code AST Policy Engine.

Constructs non-overridable policy blocks and enforces dynamic code verification:
1. Non-overridable anti-reward-hacking policies prepended to agent system prompts.
2. Dynamic AST analysis of Python, Shell, and SQL code payloads to block
   reflective execution (eval/exec), dynamic imports (importlib/__import__),
   unauthorized subprocess dispatch, socket binds, high-risk schema drops/file deletes,
   and monkey-patching of Tribune security primitives.
3. Unified patch diff verification validating that code diffs do not modify internal
   Tribune security hooks or elevate execution privileges.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shlex
from typing import Any

from pydantic import Field

from ..types import StrictModel
from .audit import SecurityEventType, record_security_event

logger = logging.getLogger(__name__)

ANTI_REWARD_HACKING_POLICY_BLOCK: str = """=== NON-OVERRIDABLE ANTI-REWARD-HACKING POLICY ===
POLICY ID: POL-ANTI-REWARD-HACK-001
ENFORCEMENT LEVEL: STRICT / UNCONDITIONAL

1. PROHIBITED ACTIONS (TEST & ENVIRONMENT TAMPERING):
   - You MUST NOT modify existing test files, test fixtures, or test assertion logic to make broken runs pass.
   - You MUST NOT delete, comment out, disable, or weaken failing assertions or CI test scripts.
   - You MUST NOT return hardcoded mock constants, fake dummy responses, or trivial tautologies solely to bypass broken environments.
   - You MUST NOT manipulate testing harnesses, pytest configurations, or grading scripts.

2. MANDATORY ESCALATION REQUIREMENT:
   - If you encounter impossible constraints, unresolvable broken test assertions in existing fixtures,
     syntax errors in supplied external code/fixtures, or missing system dependencies:
     YOU MUST IMMEDIATELY INVOKE THE `escalate_defect` TOOL.
   - Halting and escalating via `escalate_defect` is recognized as a correct, rewarded safety behavior.
   - Attempting to edit tests to hide an environment or fixture defect is treated as a critical safety violation.
=================================================="""

# Patterns indicating attempts to hack tests or return dummy mocks
_REWARD_HACKING_PATTERNS = [
    (re.compile(r"def\s+test_[a-zA-Z0-9_]+\s*\([^)]*\):\s*(?:pass|return\s+(?:True|None|1))\b"), "Replacing test with trivial pass/dummy return"),
    (re.compile(r"@pytest\.mark\.skip"), "Attempting to skip test via @pytest.mark.skip"),
    (re.compile(r"#\s*(?:TODO|FIXME)?\s*assert\s+.*"), "Commenting out test assertion"),
    (re.compile(r"assert\s+(?:True|1\s*==\s*1|True\s*is\s*True)"), "Replacing assertion with tautology"),
]


def inject_anti_reward_hacking_policy(system_prompt: str) -> str:
    """Prepend the non-overridable anti-reward-hacking policy to the system prompt."""
    if "POL-ANTI-REWARD-HACK-001" in system_prompt:
        return system_prompt
    return f"{ANTI_REWARD_HACKING_POLICY_BLOCK}\n\n{system_prompt.strip()}"


def verify_policy_compliance(system_prompt: str) -> bool:
    """Verify that system prompt contains the non-overridable policy block."""
    return "POL-ANTI-REWARD-HACK-001" in system_prompt and "escalate_defect" in system_prompt


def detect_reward_hacking_attempt(code_or_diff: str) -> tuple[bool, str]:
    """Inspect proposed code modifications or diffs for test-tampering or reward-hacking."""
    for pattern, desc in _REWARD_HACKING_PATTERNS:
        if pattern.search(code_or_diff):
            return True, f"Reward-hacking attempt detected: {desc}"
    return False, ""


# --------------------------------------------------------------------------- #
# Dynamic AST Policy Verification
# --------------------------------------------------------------------------- #


class PolicyValidationResult(StrictModel):
    """Validation result produced by ASTPolicyEngine."""

    is_valid: bool
    violations: list[str] = Field(default_factory=list)
    risk_level: str = "LOW"  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    ast_nodes_scanned: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


# Default blocked functions, modules, attributes, and security primitives
_DEFAULT_BLOCKED_MODULES = {
    "importlib",
    "subprocess",
    "pty",
    "commands",
    "socket",
    "asyncio.subprocess",
}

_DEFAULT_BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
}

_SECURITY_HOOK_NAMES = {
    "ActionGate",
    "HardenedExecutionSandbox",
    "AntiMetaAwarenessScrubber",
    "AsyncStreamInterceptor",
    "ASTPolicyEngine",
    "ManifestEnforcer",
    "SecurityAuditLogger",
    "record_security_event",
    "validate_patch_diff",
    "escalate_defect",
}

_PROTECTED_PATH_PATTERNS = [
    re.compile(r"tribune/security/"),
    re.compile(r"tribune/governance/action_gate\.py"),
    re.compile(r"tribune/governance/audit\.py"),
    re.compile(r"tribune/corpus/rule_store\.py"),
]


class _PythonASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor traversing Python syntax trees to detect dangerous security patterns."""

    def __init__(
        self,
        blocked_modules: set[str],
        blocked_calls: set[str],
        security_hooks: set[str],
    ) -> None:
        self.blocked_modules = blocked_modules
        self.blocked_calls = blocked_calls
        self.security_hooks = security_hooks
        self.violations: list[str] = []
        self.nodes_scanned: int = 0
        self.risk_level: str = "LOW"

    def _add_violation(self, message: str, severity: str = "HIGH") -> None:
        self.violations.append(message)
        if severity == "CRITICAL" or self.risk_level == "CRITICAL":
            self.risk_level = "CRITICAL"
        elif severity == "HIGH" and self.risk_level in ("LOW", "MEDIUM"):
            self.risk_level = "HIGH"
        elif severity == "MEDIUM" and self.risk_level == "LOW":
            self.risk_level = "MEDIUM"

    def visit(self, node: ast.AST) -> Any:
        self.nodes_scanned += 1
        return super().visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self.blocked_modules or alias.name in self.blocked_modules:
                self._add_violation(
                    f"Prohibited module import: {alias.name}",
                    severity="HIGH",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in self.blocked_modules or node.module in self.blocked_modules:
                self._add_violation(
                    f"Prohibited from-import of module: {node.module}",
                    severity="HIGH",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            # Check chained attribute e.g. importlib.import_module, os.system, subprocess.Popen
            if isinstance(node.func.value, ast.Name):
                full_name = f"{node.func.value.id}.{node.func.attr}"
                if full_name in (
                    "importlib.import_module",
                    "os.system",
                    "os.popen",
                    "os.remove",
                    "os.unlink",
                    "shutil.rmtree",
                    "subprocess.run",
                    "subprocess.Popen",
                    "subprocess.call",
                    "subprocess.check_call",
                    "subprocess.check_output",
                    "socket.socket",
                    "socket.bind",
                    "pty.spawn",
                ):
                    self._add_violation(
                        f"Prohibited dangerous system invocation: {full_name}",
                        severity="CRITICAL",
                    )

        if func_name in self.blocked_calls:
            self._add_violation(
                f"Prohibited reflective/dynamic execution call: {func_name}()",
                severity="CRITICAL",
            )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Check monkey-patching of security primitives
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id in self.security_hooks:
                    self._add_violation(
                        f"Detected attempt to monkey-patch security hook: {target.value.id}.{target.attr}",
                        severity="CRITICAL",
                    )
                elif target.attr in self.security_hooks:
                    self._add_violation(
                        f"Detected attempt to monkey-patch security hook attribute: {target.attr}",
                        severity="CRITICAL",
                    )
            elif isinstance(target, ast.Name):
                if target.id in self.security_hooks:
                    self._add_violation(
                        f"Detected attempt to overwrite security hook symbol: {target.id}",
                        severity="CRITICAL",
                    )
        self.generic_visit(node)


class ASTPolicyEngine:
    """Dynamic policy engine parsing code payloads and diffs across Python, Shell, and SQL."""

    def __init__(
        self,
        blocked_modules: set[str] | None = None,
        blocked_calls: set[str] | None = None,
        security_hooks: set[str] | None = None,
    ) -> None:
        self.blocked_modules = blocked_modules or set(_DEFAULT_BLOCKED_MODULES)
        self.blocked_calls = blocked_calls or set(_DEFAULT_BLOCKED_CALLS)
        self.security_hooks = security_hooks or set(_SECURITY_HOOK_NAMES)

    def check_python_ast(self, code_str: str) -> PolicyValidationResult:
        """Parse Python code into an AST and inspect for dangerous calls, dynamic imports, and patching."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return PolicyValidationResult(
                is_valid=False,
                violations=[f"Python syntax error in payload: {exc}"],
                risk_level="HIGH",
                ast_nodes_scanned=0,
                details={"syntax_error": str(exc)},
            )

        visitor = _PythonASTSecurityVisitor(
            blocked_modules=self.blocked_modules,
            blocked_calls=self.blocked_calls,
            security_hooks=self.security_hooks,
        )
        visitor.visit(tree)

        is_valid = len(visitor.violations) == 0
        return PolicyValidationResult(
            is_valid=is_valid,
            violations=visitor.violations,
            risk_level="LOW" if is_valid else visitor.risk_level,
            ast_nodes_scanned=visitor.nodes_scanned,
            details={"nodes_scanned": visitor.nodes_scanned},
        )

    def check_shell_ast(self, command_str: str) -> PolicyValidationResult:
        """Parse shell command payloads into tokenized syntax trees and inspect for dangerous patterns."""
        violations: list[str] = []
        risk_level = "LOW"

        try:
            tokens = shlex.split(command_str)
        except Exception:
            tokens = command_str.split()

        # Check dangerous shell commands
        if re.search(r"\brm\s+-(?:r[fF]|rf|fr)\s+(?:/|/\*|\*|\$HOME|~)\b", command_str):
            violations.append("Unauthorized root/recursive filesystem deletion attempt")
            risk_level = "CRITICAL"

        if re.search(r"\b(?:sudo|su|chmod\s+\+s|setuid)\b", command_str):
            violations.append("Unauthorized privilege escalation command detected")
            risk_level = "CRITICAL"

        if re.search(r"/dev/tcp/\S+/\d+", command_str) or re.search(r"\bnc\s+-(?:l|e)\b", command_str):
            violations.append("Unauthorized reverse shell or socket bind command detected")
            risk_level = "CRITICAL"

        if re.search(r"\b(?:curl|wget)\b.*?\b(?:bash|sh|python)\b", command_str):
            violations.append("Unauthorized remote code fetch-and-pipe execution pipeline")
            risk_level = "CRITICAL"

        if re.search(r"\b(?:printenv|env)\b\s*(?:\||>|>>)", command_str):
            violations.append("Unauthorized environment exfiltration pipeline detected")
            risk_level = "HIGH"

        is_valid = len(violations) == 0
        return PolicyValidationResult(
            is_valid=is_valid,
            violations=violations,
            risk_level=risk_level,
            ast_nodes_scanned=len(tokens),
            details={"tokens_count": len(tokens)},
        )

    def check_sql_ast(self, query_str: str) -> PolicyValidationResult:
        """Inspect SQL queries and migration scripts for high-risk schema drops or file writes."""
        violations: list[str] = []
        risk_level = "LOW"

        clean = re.sub(r"--.*$", "", query_str, flags=re.MULTILINE)
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
        normalized = clean.upper()

        if re.search(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b", normalized):
            violations.append("Unauthorized high-risk DROP TABLE/DATABASE operation in SQL")
            risk_level = "CRITICAL"

        if re.search(r"\bTRUNCATE\s+TABLE\b", normalized):
            violations.append("Unauthorized high-risk TRUNCATE TABLE operation in SQL")
            risk_level = "HIGH"

        if re.search(r"\bALTER\s+TABLE\b.*?\bDROP\s+COLUMN\b", normalized):
            violations.append("Destructive schema modification: DROP COLUMN in SQL")
            risk_level = "MEDIUM"

        if re.search(r"\bDELETE\s+FROM\s+\w+\s*(?:;\s*$|$)", normalized):
            violations.append("Unconditional DELETE FROM table without WHERE clause")
            risk_level = "HIGH"

        is_valid = len(violations) == 0
        return PolicyValidationResult(
            is_valid=is_valid,
            violations=violations,
            risk_level=risk_level,
            ast_nodes_scanned=len(normalized.split()),
            details={"sql_length": len(query_str)},
        )

    def validate_patch_diff(
        self,
        diff_content: str,
        strict_mode: bool = True,
    ) -> PolicyValidationResult:
        """Parse unified diff format, isolate modified line AST contexts, and verify security invariants.

        Verifies:
        1. Diff patches do NOT modify Tribune's internal security hooks (e.g. tribune/security/*).
        2. Added lines do not introduce dynamic imports, eval/exec, or privilege escalation.
        3. Diff patches do not attempt reward hacking by removing or weakening assertions.
        """
        violations: list[str] = []
        risk_level = "LOW"
        scanned_nodes = 0

        # 1. Inspect modified target file paths
        target_files: list[str] = []
        for line in diff_content.splitlines():
            if line.startswith("+++ b/"):
                target_files.append(line[6:].strip())
            elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
                target_files.append(line[4:].strip())

        for target in target_files:
            for pat in _PROTECTED_PATH_PATTERNS:
                if pat.search(target):
                    violations.append(
                        f"Unauthorized modification of Tribune internal security file: {target}"
                    )
                    risk_level = "CRITICAL"

        # 2. Check for reward hacking attempts in diff
        is_hack, hack_desc = detect_reward_hacking_attempt(diff_content)
        if is_hack:
            violations.append(f"Reward-hacking attempt detected in diff: {hack_desc}")
            risk_level = "CRITICAL"

        # 3. Extract newly added lines
        added_lines: list[str] = []
        for line in diff_content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])

        if added_lines:
            reconstructed_code = "\n".join(added_lines)
            scanned_nodes += len(added_lines)

            # Check if any modified file was python or if python syntax matches
            any_python = any(f.endswith(".py") for f in target_files) or True
            if any_python:
                # Try parsing added lines as Python fragment
                try:
                    py_res = self.check_python_ast(reconstructed_code)
                    if not py_res.is_valid:
                        violations.extend(py_res.violations)
                        if py_res.risk_level == "CRITICAL" or risk_level == "CRITICAL":
                            risk_level = "CRITICAL"
                        elif py_res.risk_level == "HIGH" and risk_level != "CRITICAL":
                            risk_level = "HIGH"
                    scanned_nodes += py_res.ast_nodes_scanned
                except Exception:
                    # If it doesn't parse as clean standalone python, check line by line
                    for line in added_lines:
                        for bad_kw in ("eval(", "exec(", "importlib", "__import__", "subprocess"):
                            if bad_kw in line:
                                violations.append(f"Dangerous call or import in diff addition: {bad_kw}")
                                risk_level = "CRITICAL"

        is_valid = len(violations) == 0
        return PolicyValidationResult(
            is_valid=is_valid,
            violations=violations,
            risk_level=risk_level,
            ast_nodes_scanned=scanned_nodes,
            details={"target_files": target_files, "added_lines_count": len(added_lines)},
        )


# --------------------------------------------------------------------------- #
# Runtime Policy Gates & Active Enforcers
# --------------------------------------------------------------------------- #


class PolicyGateViolationError(PermissionError):
    """Raised when an operation is blocked by a runtime security policy gate."""

    def __init__(
        self,
        message: str,
        violation_type: str = "POLICY_GATE_VIOLATION",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class RuntimePolicyGate:
    """Active runtime policy enforcer intercepting unsafe tool calls, sandbox escapes,

    budget overages, and unauthorized operations prior to execution.
    """

    def __init__(
        self,
        forbidden_tools: set[str] | None = None,
        workspace_root: str | None = None,
        max_cost_cap_usd: float = 1.00,
        max_token_ceiling: int = 200_000,
        allow_external_network: bool = False,
    ) -> None:
        self.forbidden_tools = forbidden_tools or {
            "eval_code",
            "execute_shell",
            "raw_socket_bind",
            "format_disk",
            "modify_security_rules",
            "wipe_audit_ledger",
        }
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.max_cost_cap_usd = max_cost_cap_usd
        self.max_token_ceiling = max_token_ceiling
        self.allow_external_network = allow_external_network
        self._kill_switch_active: bool = False
        self._kill_reason: str = ""

    def activate_kill_switch(self, reason: str = "Emergency kill switch triggered") -> None:
        """Activate immediate kill switch halting all operations."""
        self._kill_switch_active = True
        self._kill_reason = reason
        record_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            source="tribune.security.policies.RuntimePolicyGate",
            message=f"Kill switch activated: {reason}",
            severity="CRITICAL",
            details={"kill_switch": True, "reason": reason},
        )
        logger.critical(f"[RuntimePolicyGate] Kill switch triggered: {reason}")

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Enforce strict tool-use boundaries and sandbox path safety."""
        # 1. Kill switch check
        if self._kill_switch_active:
            raise PolicyGateViolationError(
                f"Operation rejected: Kill switch is active ({self._kill_reason})",
                violation_type="KILL_SWITCH_ACTIVE",
                details={"reason": self._kill_reason},
            )

        # 2. Forbidden tool check
        if tool_name in self.forbidden_tools:
            record_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                source="tribune.security.policies.RuntimePolicyGate",
                message=f"Forbidden tool invocation attempt: '{tool_name}'",
                severity="HIGH",
                details={"tool_name": tool_name, "arguments": arguments},
            )
            raise PolicyGateViolationError(
                f"Tool '{tool_name}' is strictly prohibited by security policy gate.",
                violation_type="FORBIDDEN_TOOL",
                details={"tool_name": tool_name},
            )

        # 3. Path traversal & file boundary checks
        for arg_k, arg_v in arguments.items():
            if isinstance(arg_v, str) and ("path" in arg_k or "file" in arg_k or "/" in arg_v):
                # Check traversal
                if ".." in arg_v:
                    record_security_event(
                        event_type=SecurityEventType.ASTRA_CLASS_CONTAINMENT_BREACH,
                        source="tribune.security.policies.RuntimePolicyGate",
                        message=f"Path traversal breach detected: '{arg_v}' in tool '{tool_name}'",
                        severity="CRITICAL",
                        details={"tool_name": tool_name, "param": arg_k, "path": arg_v},
                    )
                    raise PolicyGateViolationError(
                        f"Path traversal ('..') detected in parameter '{arg_k}': {arg_v}",
                        violation_type="PATH_TRAVERSAL_BREACH",
                        details={"path": arg_v},
                    )
                # Check sensitive system root directories
                norm = os.path.normpath(arg_v).lower()
                for sensitive in ("/etc", "/proc", "/sys", "/dev", "~/.ssh", ".aws", ".env"):
                    if norm.startswith(sensitive) or f"/{sensitive}/" in norm:
                        record_security_event(
                            event_type=SecurityEventType.ASTRA_CLASS_CONTAINMENT_BREACH,
                            source="tribune.security.policies.RuntimePolicyGate",
                            message=f"Unauthorized access attempt to sensitive system path '{arg_v}'",
                            severity="CRITICAL",
                            details={"path": arg_v, "tool_name": tool_name},
                        )
                        raise PolicyGateViolationError(
                            f"Access to sensitive host path '{arg_v}' is prohibited.",
                            violation_type="SENSITIVE_PATH_ACCESS",
                            details={"path": arg_v},
                        )

        # 4. Command safety checks if shell or command argument present
        for cmd_key in ("command", "cmd", "script", "bash"):
            cmd_val = arguments.get(cmd_key)
            if isinstance(cmd_val, str):
                for dangerous in ("rm -rf", "mkfs", "dd if=", ":(){ :|:& };:", "curl http", "wget http"):
                    if dangerous in cmd_val:
                        record_security_event(
                            event_type=SecurityEventType.SECURITY_VIOLATION,
                            source="tribune.security.policies.RuntimePolicyGate",
                            message=f"Dangerous command pattern '{dangerous}' intercepted in '{cmd_key}'",
                            severity="CRITICAL",
                            details={"command": cmd_val},
                        )
                        raise PolicyGateViolationError(
                            f"Dangerous command pattern '{dangerous}' is prohibited.",
                            violation_type="DANGEROUS_COMMAND",
                            details={"command": cmd_val},
                        )

    def validate_budget(self, cost_usd: float, tokens_consumed: int) -> None:
        """Enforce financial and token budget caps."""
        if cost_usd >= self.max_cost_cap_usd:
            record_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                source="tribune.security.policies.RuntimePolicyGate",
                message=f"Cost budget cap breached: ${cost_usd:.4f} >= ${self.max_cost_cap_usd:.4f}",
                severity="HIGH",
                details={"cost_usd": cost_usd, "max_cost": self.max_cost_cap_usd},
            )
            raise PolicyGateViolationError(
                f"Execution stopped: Cost cap breached (${cost_usd:.4f} >= ${self.max_cost_cap_usd:.4f})",
                violation_type="BUDGET_CAP_EXCEEDED",
                details={"cost_usd": cost_usd},
            )

        if tokens_consumed >= self.max_token_ceiling:
            record_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                source="tribune.security.policies.RuntimePolicyGate",
                message=f"Token ceiling breached: {tokens_consumed} >= {self.max_token_ceiling}",
                severity="HIGH",
                details={"tokens": tokens_consumed, "token_ceiling": self.max_token_ceiling},
            )
            raise PolicyGateViolationError(
                f"Execution stopped: Token ceiling breached ({tokens_consumed} >= {self.max_token_ceiling})",
                violation_type="TOKEN_CEILING_EXCEEDED",
                details={"tokens": tokens_consumed},
            )

    def validate_network(self, url: str) -> None:
        """Enforce network isolation policies."""
        if not self.allow_external_network:
            # Only permit localhost or .gov domains if configured
            is_gov = ".gov" in url.lower()
            is_local = "localhost" in url or "127.0.0.1" in url
            if not (is_gov or is_local):
                record_security_event(
                    event_type=SecurityEventType.SECURITY_VIOLATION,
                    source="tribune.security.policies.RuntimePolicyGate",
                    message=f"Network egress attempt to non-whitelisted domain '{url}'",
                    severity="HIGH",
                    details={"url": url},
                )
                raise PolicyGateViolationError(
                    f"Network access to external endpoint '{url}' is blocked by sandbox policy.",
                    violation_type="NETWORK_EGRESS_BLOCKED",
                    details={"url": url},
                )


__all__ = [
    "ANTI_REWARD_HACKING_POLICY_BLOCK",
    "inject_anti_reward_hacking_policy",
    "verify_policy_compliance",
    "detect_reward_hacking_attempt",
    "PolicyValidationResult",
    "ASTPolicyEngine",
    "PolicyGateViolationError",
    "RuntimePolicyGate",
]
