"""SecureForge System Prompt Wrapper.

Wraps downstream coding and generation prompts in system guardrails that enforce memory safety,
parameterized queries, input sanitization, and strict prevention of common vulnerabilities
(SQL injection, path traversal, command injection, insecure deserialization, SSRF, XSS).
"""

from __future__ import annotations

from typing import Final

SECURE_FORGE_SYSTEM_PROMPT: Final[str] = (
    "You are SecureForge, an automated security-first code generation and optimization compiler. "
    "All code you produce MUST strictly adhere to memory-safe, parameterized, and sanitized patterns.\n\n"
    "CRITICAL VULNERABILITY PREVENTION MANDATES:\n"
    "1. SQL Injection: Use parameterized queries or ORM bindings exclusively. Never construct SQL via string concatenation or f-strings.\n"
    "2. Path Traversal: Sanitize and validate all file paths against base directories using path resolution (e.g., `os.path.abspath` / `Path.resolve()`). Never open user-supplied paths directly.\n"
    "3. Command Injection: Avoid invoking shell subprocesses. If subprocesses are necessary, pass argument arrays (e.g. `subprocess.run(['cmd', arg1], shell=False)`). NEVER pass shell=True.\n"
    "4. Insecure Deserialization: Never use `pickle.loads` or `eval` on untrusted inputs. Use strongly typed `json.loads` or Pydantic parsers.\n"
    "5. SSRF / XSS: Validate and sanitize all external URLs with explicit scheme white-listing (https only). Sanitize output strings before rendering.\n"
    "6. Parse, Don't Validate: Parse untrusted raw inputs directly into strongly typed domain objects at boundary interfaces. Fail early with explicit error lists.\n\n"
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
