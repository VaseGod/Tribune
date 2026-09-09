"""Anti-Meta Awareness Scrubbing & Receptive Field Sanitization Pipeline.

Active sanitization pipeline stripping:
1. Evaluation metadata (test_eval, hidden_test, benchmark_eval, grader flags).
2. Test runner signatures (pytest, unittest, conftest.py, _pytest, runner scripts).
3. Test harness environment variables (PYTEST_CURRENT_TEST, TRIBUNE_TEST_MODE, CI_TEST_RUN, etc.).
4. Fixture markers (@pytest.fixture, tmp_path, monkeypatch, mock.patch).
5. Ensures prompt evaluation cannot detect runner or evaluation harness contexts
   while preserving legitimate statutory rules, legal citations, and applicant evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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


__all__ = [
    "AntiMetaAwarenessScrubber",
    "ScrubbingTelemetry",
]
