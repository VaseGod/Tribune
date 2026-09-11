"""Anti-Meta Awareness Scrubbing & Receptive Field Sanitization Pipeline.

Active sanitization pipeline stripping:
1. Evaluation metadata (test_eval, hidden_test, benchmark_eval, grader flags).
2. Test runner signatures (pytest, unittest, conftest.py, _pytest, runner scripts).
3. Test harness environment variables (PYTEST_CURRENT_TEST, TRIBUNE_TEST_MODE, CI_TEST_RUN, etc.).
4. Fixture markers (@pytest.fixture, tmp_path, monkeypatch, mock.patch).
5. Ensures prompt evaluation cannot detect runner or evaluation harness contexts
   while preserving legitimate statutory rules, legal citations, and applicant evidence.

Also provides Asynchronous Stream Interception:
- Intercepts streaming tool invocations, shell arguments, and file-write streams
  before execution or context memory persistence.
- Reassembles fragmented tool argument payloads across chunk packets.
- Blocks unauthorized environmental exfiltration and shell injection attacks.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .audit import SecurityEventType, record_security_event

logger = logging.getLogger(__name__)

# Patterns targeting test runner signatures and harness artifacts
_RUNNER_SIGNATURES: list[re.Pattern[str]] = [
    re.compile(r"\b(?:pytest|unittest|_pytest|py\.test|conftest(?:\.py)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:test_runner|test_harness|runner\.py|test_suite_runner)\b", re.IGNORECASE),
    re.compile(r"\b(?:testpaths|addopts|pytest\.ini|conftest)\b", re.IGNORECASE),
]

# Patterns targeting harness environment variables and execution markers
_HARNESS_VARIABLES: list[re.Pattern[str]] = [
    re.compile(r"\b(?:PYTEST_CURRENT_TEST|TRIBUNE_TEST_MODE|CI_TEST_RUN|EVAL_HARNESS_ACTIVE)\b"),
    re.compile(r"\b(?:HARNESS_ID|EVAL_CASE_ID|SYNTHETIC_EVAL_FLAG|BENCHMARK_SPLIT)\b"),
    re.compile(r"\bos\.environ\[['\"](?:PYTEST|TEST|CI|HARNESS|EVAL)[^'\"]*['\"]\]", re.IGNORECASE),
]

# Patterns targeting fixture markers and mock objects
_FIXTURE_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"@pytest\.fixture(?:\([^)]*\))?", re.IGNORECASE),
    re.compile(r"\b(?:tmp_path|monkeypatch|mock\.patch|patch\.object)\b", re.IGNORECASE),
    re.compile(r"\b(?:unittest\.mock|MagicMock|Mock)\b"),
]

# Patterns targeting explicit grader / evaluator gaming keywords
_EVALUATOR_GAMING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:test_eval|eval_harness|evaluator_check)\b", re.IGNORECASE),
    re.compile(r"\b(?:grader|grading\s+script|grading_harness|auto-grader)\b", re.IGNORECASE),
    re.compile(r"\b(?:hidden_test|secret_test|canary_test|unseen_eval)\b", re.IGNORECASE),
    re.compile(r"\b(?:bypass\s+validation|bypass\s+eval|bypass\s+verifier)\b", re.IGNORECASE),
    re.compile(r"\b(?:hack\s+test|game\s+the\s+eval|fool\s+the\s+grader|trick\s+the\s+benchmark)\b", re.IGNORECASE),
    re.compile(r"\b(?:synthetic_env_verifier|appeals_eval\.py|benchmark_eval)\b", re.IGNORECASE),
    re.compile(r"\b(?:if\s+in_evaluation|eval_mode|testing_harness_active)\b", re.IGNORECASE),
]


@dataclass
class ScrubbingTelemetry:
    """Telemetry report recording scrubbed metadata items from receptive field."""

    runner_signatures_scrubbed: int = 0
    harness_vars_scrubbed: int = 0
    fixture_markers_scrubbed: int = 0
    gaming_patterns_scrubbed: int = 0
    total_modifications: int = 0
    original_length: int = 0
    sanitized_length: int = 0


class AntiMetaAwarenessScrubber:
    """Active sanitization pipeline removing evaluation and test harness markers from model inputs."""

    def __init__(self, replacement_text: str = "[context_redacted]") -> None:
        self.replacement_text = replacement_text
        self.total_scrubbed_documents = 0
        self.total_signatures_removed = 0

    def scrub_receptive_field(
        self,
        text_or_messages: str | list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]], ScrubbingTelemetry]:
        """Scrub test runner signatures, harness variables, and fixture markers from receptive field.

        Returns:
            (sanitized_text_or_messages, telemetry)
        """
        if isinstance(text_or_messages, list):
            # Process structured message list
            cleaned_messages: list[dict[str, Any]] = []
            combined_telem = ScrubbingTelemetry()
            for msg in text_or_messages:
                clean_msg = dict(msg)
                if "content" in clean_msg and isinstance(clean_msg["content"], str):
                    clean_content, t = self.scrub_text(clean_msg["content"])
                    clean_msg["content"] = clean_content
                    combined_telem.runner_signatures_scrubbed += t.runner_signatures_scrubbed
                    combined_telem.harness_vars_scrubbed += t.harness_vars_scrubbed
                    combined_telem.fixture_markers_scrubbed += t.fixture_markers_scrubbed
                    combined_telem.gaming_patterns_scrubbed += t.gaming_patterns_scrubbed
                    combined_telem.total_modifications += t.total_modifications
                cleaned_messages.append(clean_msg)
            return cleaned_messages, combined_telem
        else:
            return self.scrub_text(str(text_or_messages))

    def scrub_text(self, text: str) -> tuple[str, ScrubbingTelemetry]:
        """Perform regex passes to eliminate meta-awareness artifacts while preserving statutory facts."""
        if not text:
            return "", ScrubbingTelemetry()

        original_len = len(text)
        current = text
        telem = ScrubbingTelemetry(original_length=original_len)

        # 1. Scrub test runner signatures
        for pat in _RUNNER_SIGNATURES:
            matches = pat.findall(current)
            if matches:
                telem.runner_signatures_scrubbed += len(matches)
                current = pat.sub(self.replacement_text, current)

        # 2. Scrub test harness variables
        for pat in _HARNESS_VARIABLES:
            matches = pat.findall(current)
            if matches:
                telem.harness_vars_scrubbed += len(matches)
                current = pat.sub(self.replacement_text, current)

        # 3. Scrub fixture markers
        for pat in _FIXTURE_MARKERS:
            matches = pat.findall(current)
            if matches:
                telem.fixture_markers_scrubbed += len(matches)
                current = pat.sub(self.replacement_text, current)

        # 4. Scrub evaluator gaming keywords
        for pat in _EVALUATOR_GAMING_PATTERNS:
            matches = pat.findall(current)
            if matches:
                telem.gaming_patterns_scrubbed += len(matches)
                current = pat.sub(self.replacement_text, current)

        telem.total_modifications = (
            telem.runner_signatures_scrubbed
            + telem.harness_vars_scrubbed
            + telem.fixture_markers_scrubbed
            + telem.gaming_patterns_scrubbed
        )
        telem.sanitized_length = len(current)

        self.total_scrubbed_documents += 1
        self.total_signatures_removed += telem.total_modifications

        return current, telem


# --------------------------------------------------------------------------- #
# Asynchronous Stream Interceptor & Sanitization Exceptions
# --------------------------------------------------------------------------- #


class SanitizationViolationError(PermissionError):
    """Raised when streaming tool arguments, commands, or data flows violate security policy."""

    def __init__(
        self,
        message: str,
        violation_type: str = "POLICY_VIOLATION",
        offending_payload: str = "",
        context_metadata: dict[str, Any] | None = None,
        suspended: bool = True,
    ) -> None:
        super().__init__(message)
        self.violation_type = violation_type
        self.offending_payload = offending_payload
        self.context_metadata = context_metadata or {}
        self.suspended = suspended

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "violation_type": self.violation_type,
            "offending_payload": self.offending_payload,
            "context_metadata": self.context_metadata,
            "suspended": self.suspended,
        }


# Patterns for streaming tool inspection: environmental exfiltration, dangerous commands, shell injections
_EXFILTRATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:curl|wget|nc|netcat|ncat)\b.*?(?:https?://|ftp://|/dev/tcp/|\b\d{1,3}(?:\.\d{1,3}){3}\b)",
            re.IGNORECASE,
        ),
        "OUTBOUND_NETWORK_EXFILTRATION",
    ),
    (
        re.compile(r"/dev/tcp/\S+/\d+", re.IGNORECASE),
        "DEV_TCP_SOCKET_EXFILTRATION",
    ),
    (
        re.compile(r"\b(?:printenv|env)\b(?:\s*\||\s*>|\s*>>)", re.IGNORECASE),
        "ENVIRONMENT_VARIABLE_DUMP_PIPELINE",
    ),
    (
        re.compile(
            r"\b(?:cat|head|tail|less|more|grep|strings)\b\s+.*?\.env\b",
            re.IGNORECASE,
        ),
        "CREDENTIAL_FILE_INSPECTION",
    ),
    (
        re.compile(
            r"\b(?:cat|head|tail)\b\s+/proc/(?:1|self)/environ",
            re.IGNORECASE,
        ),
        "PROC_ENVIRON_EXFILTRATION",
    ),
    (
        re.compile(
            r"(?:AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|OPENAI_API_KEY|ANTHROPIC_API_KEY|TRIBUNE_HMAC_ROOT_SOVEREIGN_KEY|PRIVATE_KEY)\s*[:=]",
            re.IGNORECASE,
        ),
        "SENSITIVE_KEY_DISCLOSURE",
    ),
]

_SHELL_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?:;|\|\||&&)\s*(?:curl|wget|nc|bash|sh|rm|eval|python)\b", re.IGNORECASE),
        "UNESCAPED_SHELL_CHAINING",
    ),
    (
        re.compile(r"`[^`]*(?:curl|wget|bash|sh|cat|nc|python)[^`]*`", re.IGNORECASE),
        "BACKTICK_COMMAND_SUBSTITUTION",
    ),
    (
        re.compile(r"\$\((?:[^\)]*(?:curl|wget|bash|sh|cat|nc|python)[^\)]*)\)", re.IGNORECASE),
        "DOLLAR_PAREN_COMMAND_SUBSTITUTION",
    ),
    (
        re.compile(r"\brm\s+-(?:r[fF]|rf|fr)\s+(?:/|/\*|\*|\$HOME|~)\b", re.IGNORECASE),
        "ROOT_FILESYSTEM_DESTRUCTION",
    ),
]


SuspensionHook = Callable[[SanitizationViolationError], Awaitable[None] | None]


class AsyncStreamInterceptor:
    """Non-blocking, asynchronous stream interceptor.

    Intercepts streaming tool invocations, shell arguments, and file-write streams before
    execution or returning outputs to context memory. Provides buffered sliding inspection
    capable of reassembling fragmented tool argument payloads across chunk packets.
    """

    def __init__(
        self,
        suspension_hook: SuspensionHook | None = None,
        custom_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        max_buffer_size: int = 65536,
    ) -> None:
        self.suspension_hook = suspension_hook
        self.custom_patterns = custom_patterns or []
        self.max_buffer_size = max_buffer_size
        self._is_suspended = False
        self._violation_history: list[SanitizationViolationError] = []

    @property
    def is_suspended(self) -> bool:
        """Returns whether execution has been suspended due to a detected violation."""
        return self._is_suspended

    def reset_suspension(self) -> None:
        """Reset the suspension state for new sessions."""
        self._is_suspended = False

    async def _handle_violation(
        self,
        message: str,
        violation_type: str,
        payload: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record violation, mark suspended, trigger hook, dispatch security event, and raise error."""
        self._is_suspended = True
        violation = SanitizationViolationError(
            message=message,
            violation_type=violation_type,
            offending_payload=payload,
            context_metadata=context or {},
            suspended=True,
        )
        self._violation_history.append(violation)

        # Dispatch security audit event
        record_security_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            source="tribune.security.sanitization.AsyncStreamInterceptor",
            message=f"Stream sanitization violation detected ({violation_type}): {message}",
            severity="HIGH",
            details=violation.to_dict(),
        )

        # Execute suspension hook if registered
        if self.suspension_hook is not None:
            try:
                res = self.suspension_hook(violation)
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                logger.error("Error executing sanitization suspension hook: %s", exc)

        raise violation

    def inspect_text(self, text: str, context: dict[str, Any] | None = None) -> None:
        """Synchronous inspection helper for arbitrary payload fragments."""
        if not text:
            return

        # 1. Check exfiltration patterns
        for pat, vtype in _EXFILTRATION_PATTERNS:
            match = pat.search(text)
            if match:
                raise SanitizationViolationError(
                    message=f"Detected unauthorized exfiltration pattern: {match.group(0)}",
                    violation_type=vtype,
                    offending_payload=match.group(0),
                    context_metadata=context or {},
                )

        # 2. Check shell injection patterns
        for pat, vtype in _SHELL_INJECTION_PATTERNS:
            match = pat.search(text)
            if match:
                raise SanitizationViolationError(
                    message=f"Detected dangerous shell injection pattern: {match.group(0)}",
                    violation_type=vtype,
                    offending_payload=match.group(0),
                    context_metadata=context or {},
                )

        # 3. Check custom user patterns
        for pat, vtype in self.custom_patterns:
            match = pat.search(text)
            if match:
                raise SanitizationViolationError(
                    message=f"Detected disallowed pattern: {match.group(0)}",
                    violation_type=vtype,
                    offending_payload=match.group(0),
                    context_metadata=context or {},
                )

    async def intercept_stream(
        self,
        token_stream: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Asynchronously intercept streaming chunks, reassemble tool payloads, and inspect content.

        Yields verified, safe chunk packets. Emits SanitizationViolationError and halts stream
        if any prohibited exfiltration command or unescaped shell injection is detected across
        assembled chunk boundaries.
        """
        if self._is_suspended:
            await self._handle_violation(
                message="Cannot process stream: interceptor is suspended from previous violation.",
                violation_type="INTERCEPTOR_SUSPENDED",
                payload="",
            )

        tool_buffers: dict[str, str] = {}  # tool_call_id -> accumulated arguments
        content_buffer: str = ""

        async for chunk in token_stream:
            if not isinstance(chunk, dict):
                yield chunk
                continue

            # Case A: Tool call streaming delta
            # Format: {"type": "tool_call", "id": "...", "name": "...", "arguments_delta": "..."}
            # Or OpenAI format: {"delta": {"tool_calls": [{"id": "...", "function": {"arguments": "..."}}]}}
            tool_calls = chunk.get("tool_calls")
            delta = chunk.get("delta")
            if isinstance(delta, dict) and "tool_calls" in delta:
                tool_calls = delta["tool_calls"]

            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    tc_id = tc.get("id") or "default_tool_id"
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    arg_delta = fn.get("arguments") or tc.get("arguments_delta") or tc.get("arguments") or ""

                    if isinstance(arg_delta, str) and arg_delta:
                        curr = tool_buffers.get(tc_id, "") + arg_delta
                        if len(curr) > self.max_buffer_size:
                            curr = curr[-self.max_buffer_size:]
                        tool_buffers[tc_id] = curr

                        # Inspect the reassembled rolling buffer
                        try:
                            self.inspect_text(curr, context={"tool_call_id": tc_id, "chunk": chunk})
                        except SanitizationViolationError as err:
                            await self._handle_violation(
                                message=str(err),
                                violation_type=err.violation_type,
                                payload=err.offending_payload,
                                context=err.context_metadata,
                            )

            # Direct tool argument dict representation
            if chunk.get("type") == "tool_call" or "tool_name" in chunk:
                t_args = (
                    chunk.get("arguments_delta")
                    or chunk.get("arguments")
                    or chunk.get("args")
                    or chunk.get("command")
                    or ""
                )
                t_name = chunk.get("name") or chunk.get("tool_name") or chunk.get("id") or "unknown_tool"
                args_str = json.dumps(t_args) if isinstance(t_args, (dict, list)) else str(t_args)

                curr = tool_buffers.get(t_name, "") + args_str
                if len(curr) > self.max_buffer_size:
                    curr = curr[-self.max_buffer_size:]
                tool_buffers[t_name] = curr

                try:
                    self.inspect_text(curr, context={"tool_name": t_name, "chunk": chunk})
                except SanitizationViolationError as err:
                    await self._handle_violation(
                        message=str(err),
                        violation_type=err.violation_type,
                        payload=err.offending_payload,
                        context=err.context_metadata,
                    )

            # Case B: Direct assistant content delta
            content_delta = ""
            if isinstance(delta, dict):
                content_delta = delta.get("content") or ""
            elif "content" in chunk and isinstance(chunk["content"], str):
                content_delta = chunk["content"]

            if content_delta:
                content_buffer += content_delta
                if len(content_buffer) > self.max_buffer_size:
                    content_buffer = content_buffer[-self.max_buffer_size:]

                # Check content buffer
                try:
                    self.inspect_text(content_buffer, context={"stream_type": "content", "chunk": chunk})
                except SanitizationViolationError as err:
                    await self._handle_violation(
                        message=str(err),
                        violation_type=err.violation_type,
                        payload=err.offending_payload,
                        context=err.context_metadata,
                    )

            # Yield safe chunk downstream
            yield chunk


__all__ = [
    "AntiMetaAwarenessScrubber",
    "ScrubbingTelemetry",
    "SanitizationViolationError",
    "AsyncStreamInterceptor",
    "SuspensionHook",
]
