"""Context Window Compaction & Structured Feedback Injection.

Implements deterministic context management that:
1. Preserves invariant system prompts, statutory anchors, and verifier failure records.
2. Truncates or prunes ephemeral intermediate tool calls and intermediate thoughts.
3. Injects structured verifier gate rejection feedback directly into context.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextCompactor:
    """Manages context window size while preserving critical evidentiary markers."""

    def __init__(
        self,
        max_context_tokens: int = 16_000,
        intermediate_thought_ceiling_chars: int = 2048,
        preserve_recent_turns: int = 4,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.intermediate_thought_ceiling_chars = intermediate_thought_ceiling_chars
        self.preserve_recent_turns = preserve_recent_turns

    def estimate_tokens(self, text: str) -> int:
        """Heuristic offline token estimate (~4 chars per token)."""
        return max(1, (len(text) + 3) // 4)

    def compact(
        self,
        messages: list[dict[str, Any]],
        mandatory_anchors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compact conversation messages if context token limit is exceeded.

        Guarantees:
        - System instructions (role='system') are NEVER removed.
        - Verifier feedback or defect injection messages are NEVER removed.
        - Recent turns are prioritized.
        - Intermediate verbose thoughts in older messages are trimmed with explicit markers.
        """
        if not messages:
            return []

        anchors = mandatory_anchors or []
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.max_context_tokens:
            return list(messages)

        logger.info(
            f"[ContextCompactor] Compacting context ({estimated_tokens} tokens > {self.max_context_tokens} limit)"
        )

        compacted: list[dict[str, Any]] = []
        # Separate system messages and conversation turns
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation_msgs = [m for m in messages if m.get("role") != "system"]

        compacted.extend(system_msgs)

        # Truncate intermediate scratchpad thoughts in older messages
        cutoff_idx = max(0, len(conversation_msgs) - self.preserve_recent_turns)

        for idx, msg in enumerate(conversation_msgs):
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            # If message contains verifier failure or mandatory anchor, preserve it completely
            is_critical = (
                "[VERIFIER_GATE_REJECTION]" in content
                or "[DEFECT_LOCATED]" in content
                or any(anchor in content for anchor in anchors)
            )

            if is_critical or idx >= cutoff_idx:
                compacted.append(msg)
            else:
                # Older turn: compress intermediate verbosity
                if len(content) > self.intermediate_thought_ceiling_chars:
                    head = content[: self.intermediate_thought_ceiling_chars // 2]
                    tail = content[-self.intermediate_thought_ceiling_chars // 4 :]
                    pruned_content = (
                        f"{head}\n... [TRUNCATED {len(content) - len(head) - len(tail)} chars of intermediate reasoning] ...\n{tail}"
                    )
                    compacted.append({"role": role, "content": pruned_content})
                else:
                    compacted.append(msg)

        return compacted

    def inject_verifier_feedback(
        self,
        messages: list[dict[str, Any]],
        verifier_feedback: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format structured verifier gate rejection into a context feedback message."""
        status = verifier_feedback.get("status", "FAILED")
        reason = verifier_feedback.get("failure_reason", "Validation failure")
        remediation = verifier_feedback.get("remediation_hint", "Correct the defective citation or fact.")
        failed_citation = verifier_feedback.get("citation", "")

        feedback_content = (
            f"=== [VERIFIER_GATE_REJECTION] ===\n"
            f"GATE STATUS: {status}\n"
            f"DEFECTIVE ITEM: {failed_citation}\n"
            f"FAILURE REASON: {reason}\n"
            f"REQUIRED ACTION: {remediation}\n"
            f"================================="
        )

        new_messages = list(messages)
        new_messages.append({"role": "user", "content": feedback_content})
        return new_messages
