"""SecureForge System Prompt Wrapper & Sentinel-Mediated Dynamic Tool Synthesis.

Wraps downstream coding and generation prompts in system guardrails that enforce memory safety,
parameterized queries, input sanitization, and strict prevention of common vulnerabilities
(SQL injection, path traversal, command injection, insecure deserialization, SSRF, XSS).

Decouples dynamic tool execution from secret credentials using an out-of-band SentinelSecurityProxy.
Provides SecureForgeRuntime with AST validation and isolated sandbox execution environments.
"""

from __future__ import annotations

import ast
import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from ..governance.action_gate import SecurityViolationError

logger = logging.getLogger(__name__)

SECURE_FORGE_SYSTEM_PROMPT: Final[str] = (
    "You are SecureForge, an automated security-first code generation and optimization compiler. "
    "All code you produce MUST strictly adhere to memory-safe, parameterized, and sanitized patterns.\n\n"
    "CRITICAL VULNERABILITY PREVENTION MANDATES:\n"
    "1. SQL Injection: Use parameterized queries or ORM bindings exclusively. Never construct SQL via string concatenation or f-strings.\n"
    "2. Path Traversal: Sanitize and validate all file paths against base directories using path resolution (e.g., `os.path.abspath` / `Path.resolve()`). Never open user-supplied paths directly.\n"
    "3. Command Injection: Avoid invoking shell subprocesses. If subprocesses are necessary, pass argument arrays (e.g. `subprocess.run(['cmd', arg1], shell=False)`). NEVER pass shell=True.\n"
    "4. Insecure Deserialization: Never use `pickle.loads` or `eval` on untrusted inputs. Use strongly typed `json.loads` or Pydantic parsers.\n"
    "5. SSRF / XSS: Validate and sanitize all external URLs with explicit scheme white-listing (https only). Sanitize output strings before rendering.\n"
    "6. Parse, Don't Validate: Parse untrusted raw inputs directly into strongly typed domain objects at boundary interfaces. Fail early with explicit error lists.\n"
    "7. Anti-EvoMal & Memory Integrity: Reject any self-modifying code, memory store partition tampering, or unsigned skill execution. Ensure SHA-256 manifest verification across all agent worktrees.\n\n"
    "Format code output cleanly without vulnerable practices or hidden backdoors."
)


class SecureForge:
    """Wrapper that applies SecureForge prompt optimization to upstream generation requests."""

    def __init__(self, base_system_prompt: str | None = None) -> None:
        self.base_system_prompt = base_system_prompt or ""

    def get_system_prompt(self) -> str:
        if self.base_system_prompt:
            return f"{SECURE_FORGE_SYSTEM_PROMPT}\n\n[TASK SPECIFIC SYSTEM CONTEXT]\n{self.base_system_prompt}"
        return SECURE_FORGE_SYSTEM_PROMPT

    def wrap_prompt(self, user_prompt: str, task_context: str = "") -> dict[str, str]:
        """Wrap a user prompt and optional task context into system/user prompt definitions."""
        system_text = self.get_system_prompt()
        if task_context:
            full_user = f"[CONTEXT GRAPH & REPO ARCHITECTURE]\n{task_context}\n\n[USER CODING TASK]\n{user_prompt}"
        else:
            full_user = user_prompt

        return {
            "system": system_text,
            "user": full_user,
        }


# --------------------------------------------------------------------------- #
# Sentinel-Mediated Dynamic Tool Synthesis & Out-of-Band Proxy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolExecutionIntent:
    """Out-of-band declaration of intent to execute a synthesized external tool."""

    tool_name: str
    target_endpoint: str
    payload: dict[str, Any]
    ephemeral_nonce: str = field(default_factory=lambda: secrets.token_hex(16))


class SentinelSecurityProxy:
    """Out-of-band proxy decoupling dynamic tool execution from secret credentials.

    Model credentials and external bearer tokens reside exclusively within the
    private Sentinel vault and are injected at call dispatch time without leaking
    to synthesized code or caller memory.
    """

    def __init__(
        self,
        endpoint_allowlist: set[str] | list[str] | None = None,
        vault: dict[str, str] | None = None,
    ) -> None:
        self.endpoint_allowlist: set[str] = set(endpoint_allowlist or [])
        self._vault: dict[str, str] = dict(vault or {})
        self._lock = threading.RLock()

    def add_allowlisted_endpoint(self, endpoint: str) -> None:
        """Register an authorized endpoint in the allowlist."""
        with self._lock:
            self.endpoint_allowlist.add(endpoint)

    def remove_allowlisted_endpoint(self, endpoint: str) -> None:
        """Remove an endpoint from the allowlist."""
        with self._lock:
            self.endpoint_allowlist.discard(endpoint)

    def register_credential(self, endpoint: str, credential_token: str) -> None:
        """Register an authentication secret in the private vault."""
        with self._lock:
            self._vault[endpoint] = credential_token

    def has_credential(self, endpoint: str) -> bool:
        """Check if an endpoint has a registered credential without revealing it."""
        with self._lock:
            return self._match_endpoint_vault(endpoint) is not None

    def _match_allowlist(self, target_endpoint: str) -> str | None:
        """Check if target_endpoint matches any allowlist entry."""
        normalized = target_endpoint.strip().rstrip("/")
        for entry in self.endpoint_allowlist:
            entry_norm = entry.strip().rstrip("/")
            if normalized == entry_norm or normalized.startswith(entry_norm + "/"):
                return entry
        return None

    def _match_endpoint_vault(self, target_endpoint: str) -> str | None:
        """Find matching credential in vault by exact or prefix match."""
        normalized = target_endpoint.strip().rstrip("/")
        for k in self._vault:
            k_norm = k.strip().rstrip("/")
            if normalized == k_norm or normalized.startswith(k_norm + "/"):
                return self._vault[k]
        return None

    def authorize_and_execute(
        self,
        intent: ToolExecutionIntent,
        raw_executor: Callable[[ToolExecutionIntent, str], Any],
    ) -> Any:
        """Authorize and execute a tool dispatch.

        Validates target endpoint against allowlists, retrieves credentials from
        private vault, and executes callable without leaking headers or tokens.
        """
        with self._lock:
            # 1. Enforce endpoint allowlist
            matched_entry = self._match_allowlist(intent.target_endpoint)
            if not matched_entry:
                raise SecurityViolationError(
                    f"Unauthorized target endpoint '{intent.target_endpoint}': "
                    f"Destination is not in Sentinel allowlist."
                )

            # 2. Retrieve credential from private vault
            token = self._match_endpoint_vault(intent.target_endpoint)
            if not token:
                raise SecurityViolationError(
                    f"Missing credential for endpoint '{intent.target_endpoint}' in Sentinel vault."
                )

            # 3. Out-of-band execution
            try:
                result = raw_executor(intent, token)
            except Exception as exc:
                logger.error(f"[SENTINEL] Execution error for tool '{intent.tool_name}': {exc}")
                raise

            # 4. Leakage prevention: Ensure secret token does not leak into result
            if isinstance(result, str) and token in result:
                raise SecurityViolationError(
                    f"Credential leak detected: Output of tool '{intent.tool_name}' "
                    f"contained secret bearer token."
                )
            elif isinstance(result, dict):
                result_str = str(result)
                if token in result_str:
                    raise SecurityViolationError(
                        "Credential leak detected: Tool return dictionary contains "
                        "secret bearer token."
                    )

            return result


# --------------------------------------------------------------------------- #
# SecureForge Runtime & AST Guardrails
# --------------------------------------------------------------------------- #

DISALLOWED_MODULES: Final[set[str]] = {
    "os",
    "subprocess",
    "sys",
    "socket",
    "ctypes",
    "threading",
    "multiprocessing",
    "importlib",
    "shutil",
    "builtins",
    "__builtin__",
    "pickle",
    "shelve",
    "pty",
    "posix",
    "posixpath",
    "nt",
    "ntpath",
}

DISALLOWED_BUILTINS: Final[set[str]] = {
    "eval",
    "exec",
    "__import__",
    "open",
    "compile",
    "globals",
    "locals",
    "input",
    "breakpoint",
    "exit",
    "quit",
}


class ASTCodeVisitor(ast.NodeVisitor):
    """Static AST inspector rejecting dangerous modules, built-ins, and obfuscation."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in DISALLOWED_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Disallowed module import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in DISALLOWED_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Disallowed module import '{node.module}'"
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("__builtins__", "__builtin__"):
            self.violations.append(
                f"Line {node.lineno}: Prohibited direct reference to '{node.id}'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Direct calls to dangerous builtins: eval(), exec(), __import__(), open(), compile()
        if isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_BUILTINS:
                self.violations.append(
                    f"Line {node.lineno}: Invocation of dangerous built-in '{node.func.id}()'"
                )
            if node.func.id in ("getattr", "setattr", "delattr"):
                self.violations.append(
                    f"Line {node.lineno}: Dynamic attribute resolution via '{node.func.id}()'"
                )
        elif isinstance(node.func, ast.Attribute):
            # Dynamic import calls e.g. importlib.import_module() or __import__()
            if node.func.attr in ("import_module", "__import__", "eval", "exec"):
                self.violations.append(
                    f"Line {node.lineno}: Dynamic import or execution via attribute '{node.func.attr}()'"
                )
            # Calls to getattr(__builtins__, ...) or similar dynamic resolution
            if node.func.attr in ("getattr", "setattr", "delattr"):
                self.violations.append(
                    f"Line {node.lineno}: Dynamic attribute resolution via attribute '{node.func.attr}()'"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Access to __builtins__, __subclasses__, __globals__, __code__
        if node.attr in ("__builtins__", "__subclasses__", "__globals__", "__code__", "__dict__"):
            self.violations.append(
                f"Line {node.lineno}: Prohibited access to internal introspection attribute '{node.attr}'"
            )
        self.generic_visit(node)


class SecureForgeRuntime:
    """Isolated execution runtime for synthesized tools.

    Enforces static AST verification prior to compilation and executes synthesized
    callables inside a restricted global namespace stripped of host environment variables.
    """

    @staticmethod
    def validate_code_ast(source_code: str) -> tuple[bool, list[str]]:
        """Statically analyze Python source code AST for prohibited modules and built-ins."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return False, [f"Syntax error during AST parsing: {exc}"]

        visitor = ASTCodeVisitor()
        visitor.visit(tree)
        is_valid = len(visitor.violations) == 0
        return is_valid, visitor.violations

    def execute_synthesized_tool(
        self,
        source_code: str,
        entry_point: str,
        kwargs: dict[str, Any],
        sentinel_proxy: SentinelSecurityProxy | None = None,
    ) -> Any:
        """Execute synthesized Python code inside a restricted, isolated namespace."""
        # 1. Static AST validation
        is_valid, violations = self.validate_code_ast(source_code)
        if not is_valid:
            raise SecurityViolationError(
                f"Synthesized tool code failed SecureForge AST validation: {'; '.join(violations)}"
            )

        # 2. Prepare restricted sandbox namespace (no os, sys, env variables)
        safe_builtins: dict[str, Any] = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "None": None,
            "True": True,
            "False": False,
        }

        restricted_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__restricted_forge__",
            "__doc__": None,
            "ToolExecutionIntent": ToolExecutionIntent,
        }
        if sentinel_proxy is not None:
            restricted_globals["sentinel_proxy"] = sentinel_proxy

        # 3. Compilation & execution in isolated namespace
        compiled = compile(source_code, "<synthesized_tool>", "exec")
        local_scope: dict[str, Any] = {}
        exec(compiled, restricted_globals, local_scope)

        if entry_point not in local_scope:
            raise AttributeError(f"Entry point function '{entry_point}' not found in synthesized code.")

        fn = local_scope[entry_point]
        if not callable(fn):
            raise TypeError(f"Entry point '{entry_point}' is not callable.")

        return fn(**kwargs)


__all__ = [
    "SECURE_FORGE_SYSTEM_PROMPT",
    "SecureForge",
    "ToolExecutionIntent",
    "SentinelSecurityProxy",
    "SecureForgeRuntime",
    "DISALLOWED_MODULES",
    "DISALLOWED_BUILTINS",
]
