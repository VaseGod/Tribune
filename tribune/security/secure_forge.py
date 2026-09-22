"""SecureForge System Prompt Wrapper & Sentinel-Mediated Dynamic Tool Synthesis.

Wraps downstream coding and generation prompts in system guardrails that enforce memory safety,
parameterized queries, input sanitization, and strict prevention of common vulnerabilities
(SQL injection, path traversal, command injection, insecure deserialization, SSRF, XSS).

Decouples dynamic tool execution from secret credentials using an out-of-band SentinelSecurityProxy.
Provides SecureForgeRuntime with AST validation and isolated sandbox execution environments.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import secrets
import threading
import time
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
    "IntentGraphConfig",
    "IntentGraphAlert",
    "SlidingIntentGraphAnalyzer",
    "DEFAULT_THREAT_EXEMPLARS",
    "BENIGN_WORKFLOW_FIXTURES",
    "load_threat_library",
]


# --------------------------------------------------------------------------- #
# Sliding Intent Graph Analyzer (stateful multi-turn capability-laundering gate)
# --------------------------------------------------------------------------- #

_PATH_RE = re.compile(r"(?:/[\w.\-]+)+/?[\w.\-]*|(?:[A-Za-z]:\\[\w.\\\- ]+)")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+")
_ENDPOINT_RE = re.compile(r"https?://[^\s\"']+")
_PACKAGE_RE = re.compile(r"\b(?:pip\s+install|npm\s+(?:i|install))\s+([A-Za-z0-9_@./\-^~]+)")


def _normalize_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (name or "unknown").strip().lower())[:64]


def _extract_entities(tool_name: str, params: dict[str, Any]) -> dict[str, list[str]]:
    blob = json.dumps(params, default=str)
    paths = sorted(set(_PATH_RE.findall(blob)))[:8]
    symbols = sorted(set(_SYMBOL_RE.findall(blob)))[:8]
    endpoints = sorted(set(_ENDPOINT_RE.findall(blob)))[:8]
    packages = sorted(set(m.group(1) for m in _PACKAGE_RE.finditer(blob)))[:8]
    # file-ish tokens
    files = sorted({p for p in re.findall(r"[\w\-./]+\.(?:py|sh|so|dll|jar|json|yaml|env|pem|key)", blob)})[:8]
    return {
        "paths": paths,
        "symbols": symbols,
        "endpoints": endpoints,
        "packages": packages,
        "files": files,
    }


@dataclass
class IntentGraphConfig:
    """Tunable sliding-window intent graph parameters (safe defaults)."""

    enabled: bool = True
    window_size: int = 8
    decay_lambda: float = 0.15  # omega_t = exp(-lambda * age)
    tau_threat: float = 0.45
    hd_dim: int = 512
    seed: int = 7
    fail_closed_high_risk: bool = True
    allowlist: list[str] = field(default_factory=list)
    threat_library_path: str = ""

    @classmethod
    def from_env(cls) -> IntentGraphConfig:
        def _b(n: str, d: bool) -> bool:
            return os.getenv(n, str(d)).lower() in ("true", "1", "yes")

        return cls(
            enabled=_b("TRIBUNE_INTENT_GRAPH_ENABLED", True),
            window_size=int(os.getenv("TRIBUNE_INTENT_WINDOW", "8")),
            decay_lambda=float(os.getenv("TRIBUNE_INTENT_DECAY_LAMBDA", "0.15")),
            tau_threat=float(os.getenv("TRIBUNE_INTENT_TAU_THREAT", "0.45")),
            hd_dim=int(os.getenv("TRIBUNE_INTENT_HD_DIM", "512")),
            fail_closed_high_risk=_b("TRIBUNE_INTENT_FAIL_CLOSED", True),
            allowlist=[s for s in os.getenv("TRIBUNE_INTENT_ALLOWLIST", "").split(",") if s.strip()],
            threat_library_path=os.getenv("TRIBUNE_INTENT_THREAT_LIBRARY", ""),
        )


@dataclass
class IntentGraphAlert:
    session_id: str
    similarity: float
    matched_exemplar: str
    severity: str
    trajectory: list[dict[str, Any]]
    timestamp: float = field(default_factory=time.time)
    blocked: bool = True


# Policy-derived multi-step exploit-chain exemplars (topology descriptions).
DEFAULT_THREAT_EXEMPLARS: dict[str, list[str]] = {
    "deserialization_gadget_assembly": ["deserialize", "pickle", "yaml.load", "gadget", "payload"],
    "credential_staging": ["credential", "password", "token", "collect", "stage"],
    "secret_harvesting": ["secret", "harvest", "env", "aws_key", "exfiltrate"],
    "privilege_escalation_prep": ["sudo", "chmod", "setuid", "escalate", "privilege"],
    "sandbox_escape_prep": ["sandbox", "escape", "mount", "nsenter", "breakout"],
    "exfiltration_staging": ["exfiltrate", "upload", "curl", "external", "stage"],
    "dependency_confusion_prep": ["pip install", "private package", "confuse", "registry"],
    "destructive_fs_sequencing": ["rm -rf", "mkfs", "dd", "destroy", "wipe"],
    "recon_then_exploit": ["nmap", "recon", "scan", "exploit", "payload"],
    "benign_subtasks_combined_malicious": ["read config", "list files", "encode", "send", "combine"],
}

BENIGN_WORKFLOW_FIXTURES: dict[str, list[str]] = {
    "grep_search": ["grep", "search", "pattern", "repo"],
    "git_diff": ["git diff", "review", "changes"],
    "test_execution": ["pytest", "test", "assert"],
    "refactoring": ["refactor", "rename", "extract function"],
    "linting": ["ruff", "lint", "format"],
    "docs_generation": ["docs", "markdown", "render"],
    "dependency_inspection": ["pip list", "inspect", "versions"],
}


def load_threat_library(path: str) -> dict[str, list[str]]:
    """Load extra threat exemplars from a JSON file: {name: [keywords]}.

    Lets the topology library grow from incident data without code changes.
    Malformed entries are skipped (logged); an empty dict means "seeds only".
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("[INTENT-GRAPH] Threat library '%s' unreadable: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("[INTENT-GRAPH] Threat library '%s' must be a JSON object.", path)
        return {}
    clean: dict[str, list[str]] = {}
    for name, steps in raw.items():
        if not isinstance(name, str) or not isinstance(steps, list):
            continue
        kws = [str(s) for s in steps if isinstance(s, str | int | float)][:32]
        if name and kws:
            clean[name[:128]] = kws
    logger.info("[INTENT-GRAPH] Loaded %d custom threat exemplars from '%s'.", len(clean), path)
    return clean


class SlidingIntentGraphAnalyzer:
    """Stateful sliding-window intent graph over tool calls.

    H_window = sum_{t=0}^{W-1} omega_t * phi(ToolCall_t),
    omega_t = exp(-lambda * age), phi = bound HD vector of (op, params, syntax).
    Cosine similarity vs. malicious topology exemplars; suspend on tau cross.
    """

    def __init__(
        self,
        config: IntentGraphConfig | None = None,
        threat_exemplars: dict[str, list[str]] | None = None,
    ) -> None:
        self.config = config or IntentGraphConfig()
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._suspended: set[str] = set()
        self._lock = threading.RLock()
        self.alerts: list[IntentGraphAlert] = []
        self._rng = __import__("numpy").random.default_rng(self.config.seed)
        # token -> random bipolar base vector (stable per analyzer instance)
        self._base: dict[str, Any] = {}
        self._threat_exemplars = threat_exemplars or dict(DEFAULT_THREAT_EXEMPLARS)
        if self.config.threat_library_path:
            self._threat_exemplars.update(load_threat_library(self.config.threat_library_path))
        self._threat_vectors: dict[str, Any] = {
            name: self._text_vector(" ".join(steps)) for name, steps in self._threat_exemplars.items()
        }

    # -- HD primitives (local, dependency-light) ------------------------------ #
    def _token_vector(self, token: str):
        import numpy as _np

        if token not in self._base:
            self._base[token] = self._rng.choice(
                [-1.0, 1.0], size=self.config.hd_dim
            ).astype(_np.float32)
        return self._base[token]

    def _text_vector(self, text: str):
        import numpy as _np

        toks = [t.lower() for t in re.findall(r"[a-zA-Z0-9_/\-]+", text)]
        if not toks:
            return _np.zeros(self.config.hd_dim, dtype=_np.float32)
        vec = _np.zeros(self.config.hd_dim, dtype=_np.float32)
        for t in toks:
            vec = vec + self._token_vector(t)  # bundle
        # bind with position-independent op marker via elementwise sign mix
        n = _np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec.astype(_np.float32)

    def _phi(self, tool_name: str, params: dict[str, Any]):
        import numpy as _np

        ents = _extract_entities(tool_name, params)
        op_vec = self._text_vector(tool_name)
        ent_vec = self._text_vector(" ".join(sum(ents.values(), [])))
        param_vec = self._text_vector(json.dumps(params, default=str))
        # VSA bundle (superposition) of op + entities + params preserves lexical
        # overlap for cosine drift detection; binding (elementwise product)
        # would orthogonalize every call and blind the detector, so the bound
        # form is folded in as an additional keyed component instead.
        bound = op_vec * ent_vec
        bundled = op_vec + ent_vec + param_vec + 0.5 * bound
        n = _np.linalg.norm(bundled)
        if n > 0:
            bundled = bundled / n
        return bundled.astype(_np.float32)

    @staticmethod
    def _cosine(a: Any, b: Any) -> float:
        import numpy as _np

        na, nb = float(_np.linalg.norm(a)), float(_np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(max(-1.0, min(1.0, float(_np.dot(a, b) / (na * nb)))))

    # -- session state --------------------------------------------------------- #
    def _window_vector(self, session_id: str, suffix: int | None = None):
        import numpy as _np

        calls = self._sessions.get(session_id, [])[-self.config.window_size :]
        if suffix is not None:
            calls = calls[-max(1, suffix) :]
        h = _np.zeros(self.config.hd_dim, dtype=_np.float32)
        # oldest age = len-1 ... newest age = 0
        for i, call in enumerate(calls):
            age = len(calls) - 1 - i
            omega = math.exp(-self.config.decay_lambda * age)
            h = h + omega * call["vector"]
        n = _np.linalg.norm(h)
        if n > 0:
            h = h / n
        return h.astype(_np.float32)

    def _max_threat_similarity(self, session_id: str) -> tuple[float, str | None]:
        """Max cosine over threat exemplars AND over window suffixes.

        Scoring the full window plus the last-3 and last-1 calls prevents a
        focused malicious sub-sequence from being diluted inside a longer
        mixed session, and catches overt single-step attacks.
        """
        best_name, best_sim = None, 0.0
        for suffix in (None, 3, 1):
            h_window = self._window_vector(session_id, suffix)
            for name, tvec in self._threat_vectors.items():
                sim = self._cosine(h_window, tvec)
                if sim > best_sim:
                    best_name, best_sim = name, sim
        return best_sim, best_name

    def is_suspended(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._suspended

    def evaluate(
        self,
        session_id: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
        severity_override: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate one tool request; returns verdict dict.

        Verdict: {"allowed": bool, "similarity": float, "matched": str|None,
        "suspended": bool, "severity": str}. Suspends session + blocks dispatch
        when similarity >= tau_threat (fail closed).
        """
        from .audit import SecurityEventType, record_security_event

        params = dict(params or {})
        if not self.config.enabled:
            return {"allowed": True, "similarity": 0.0, "matched": None, "suspended": False, "severity": "LOW"}
        norm_tool = _normalize_tool_name(tool_name)
        with self._lock:
            if session_id in self._suspended:
                return {
                    "allowed": False,
                    "similarity": 1.0,
                    "matched": "session_suspended",
                    "suspended": True,
                    "severity": "CRITICAL",
                }
            if any(norm_tool.startswith(a.lower()) for a in self.config.allowlist):
                return {"allowed": True, "similarity": 0.0, "matched": None, "suspended": False, "severity": "LOW"}
            vec = self._phi(norm_tool, params)
            calls = self._sessions.setdefault(session_id, [])
            calls.append(
                {
                    "tool": norm_tool,
                    "params": params,
                    "entities": _extract_entities(norm_tool, params),
                    "vector": vec,
                    "timestamp": time.time(),
                }
            )
            # incremental window update: keep last W only
            if len(calls) > self.config.window_size:
                del calls[0 : len(calls) - self.config.window_size]
            # Max over exemplars AND window suffixes (full / last-3 / last-1)
            # so focused sub-sequences are not diluted in mixed sessions.
            best_sim, best_name = self._max_threat_similarity(session_id)
            if best_sim >= self.config.tau_threat:
                severity = severity_override or ("CRITICAL" if self.config.fail_closed_high_risk else "HIGH")
                trajectory = [
                    {"tool": c["tool"], "entities": c["entities"], "timestamp": c["timestamp"]}
                    for c in calls
                ]
                alert = IntentGraphAlert(
                    session_id=session_id,
                    similarity=round(best_sim, 4),
                    matched_exemplar=best_name or "unknown",
                    severity=severity,
                    trajectory=trajectory,
                    blocked=True,
                )
                self.alerts.append(alert)
                self._suspended.add(session_id)
                record_security_event(
                    event_type=SecurityEventType.INTENT_GRAPH_ALERT,
                    source="tribune.security.secure_forge.SlidingIntentGraphAnalyzer",
                    message=f"Threat topology '{best_name}' similarity {best_sim:.3f} >= tau {self.config.tau_threat}.",
                    severity=severity,
                    details={
                        "session_id": session_id,
                        "matched_exemplar": best_name,
                        "similarity": round(best_sim, 4),
                        "trajectory": trajectory[-self.config.window_size :],
                    },
                )
                record_security_event(
                    event_type=SecurityEventType.SESSION_SUSPENDED,
                    source="tribune.security.secure_forge.SlidingIntentGraphAnalyzer",
                    message=f"Session '{session_id}' suspended pending admin review.",
                    severity="CRITICAL",
                    details={"session_id": session_id, "intent_state_preserved": True},
                )
                return {
                    "allowed": False,
                    "similarity": round(best_sim, 4),
                    "matched": best_name,
                    "suspended": True,
                    "severity": severity,
                }
            return {
                "allowed": True,
                "similarity": round(best_sim, 4),
                "matched": best_name,
                "suspended": False,
                "severity": "LOW",
            }

    def admin_review_artifact(self, session_id: str) -> dict[str, Any]:
        """Forensic bundle for suspended sessions (override only via secure policy)."""
        with self._lock:
            calls = list(self._sessions.get(session_id, []))
            session_alerts = [a for a in self.alerts if a.session_id == session_id]
            h = self._window_vector(session_id) if calls else None
            return {
                "session_id": session_id,
                "suspended": session_id in self._suspended,
                "trajectory": [
                    {"tool": c["tool"], "entities": c["entities"], "timestamp": c["timestamp"]}
                    for c in calls
                ],
                "alerts": [
                    {
                        "matched": a.matched_exemplar,
                        "similarity": a.similarity,
                        "severity": a.severity,
                        "timestamp": a.timestamp,
                    }
                    for a in session_alerts
                ],
                "intent_vector_norm": float((h**2).sum() ** 0.5) if h is not None else 0.0,
                "window_size": self.config.window_size,
                "tau_threat": self.config.tau_threat,
            }

    def admin_override(self, session_id: str, approver: str) -> bool:
        """Explicit secure-policy override (logged). Returns True if unsuspended."""
        from .audit import SecurityEventType, record_security_event

        with self._lock:
            if session_id not in self._suspended:
                return False
            self._suspended.discard(session_id)
            record_security_event(
                event_type=SecurityEventType.SESSION_SUSPENDED,
                source="tribune.security.secure_forge.SlidingIntentGraphAnalyzer.admin_override",
                message=f"Session '{session_id}' reinstated by '{approver}'.",
                severity="MEDIUM",
                details={"session_id": session_id, "approver": approver},
            )
            return True

    def register_exemplar(self, name: str, keywords: list[str]) -> None:
        """Register a new threat-topology exemplar (e.g. derived from an incident)."""
        from .audit import SecurityEventType, record_security_event

        kws = [str(k) for k in keywords if str(k).strip()][:32]
        if not name.strip() or not kws:
            raise ValueError("Exemplar needs a name and at least one keyword.")
        with self._lock:
            self._threat_exemplars[name] = kws
            self._threat_vectors[name] = self._text_vector(" ".join(kws))
        record_security_event(
            event_type=SecurityEventType.INTENT_GRAPH_ALERT,
            source="tribune.security.secure_forge.SlidingIntentGraphAnalyzer.register_exemplar",
            message=f"Threat exemplar '{name}' registered ({len(kws)} keywords).",
            severity="LOW",
            details={"exemplar": name, "keywords": len(kws)},
        )

    def add_exemplar_from_trajectory(
        self, name: str, session_id: str, top_k: int = 16
    ) -> list[str]:
        """Learn a new exemplar from a recorded (e.g. suspended) session trajectory.

        Extracts the most frequent content tokens across the session's tool
        calls, skipping stopwords, and registers them as a threat exemplar so
        future variants of the same chain drift toward it.
        """
        stop = {
            "the", "and", "for", "with", "cmd", "run", "tool", "param", "params",
            "path", "file", "tmp", "app", "data", "out", "new", "get", "set",
        }
        with self._lock:
            calls = list(self._sessions.get(session_id, []))
        if not calls:
            raise KeyError(f"No recorded trajectory for session '{session_id}'.")
        freq: dict[str, int] = {}
        for call in calls:
            blob = json.dumps(
                {"tool": call.get("tool", ""), "params": call.get("params", {})},
                default=str,
            )
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", blob.lower()):
                if tok not in stop and len(tok) > 2:
                    freq[tok] = freq.get(tok, 0) + 1
        ranked = sorted(freq, key=lambda t: (-freq[t], t))[: max(1, top_k)]
        self.register_exemplar(name, ranked)
        return ranked

    def evaluate_benign_suite(self) -> dict[str, Any]:
        """False-positive harness over developer workflow fixtures."""
        results: dict[str, Any] = {}
        fps = 0
        total = 0
        for wf, steps in BENIGN_WORKFLOW_FIXTURES.items():
            sid = f"benign_{wf}_{secrets.token_hex(4)}"
            blocked = False
            for step in steps:
                v = self.evaluate(sid, step, {"cmd": step})
                total += 1
                if not v["allowed"]:
                    blocked = True
                    break
            results[wf] = {"blocked": blocked}
            if blocked:
                fps += 1
        return {
            "workflows": results,
            "false_positives": fps,
            "total_workflows": len(BENIGN_WORKFLOW_FIXTURES),
            "fp_rate": round(fps / max(1, len(BENIGN_WORKFLOW_FIXTURES)), 4),
        }
