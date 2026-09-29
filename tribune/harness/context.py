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
        tool_output_ceiling_chars: int = 4000,
        enable_staged_elision: bool = True,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.intermediate_thought_ceiling_chars = intermediate_thought_ceiling_chars
        self.preserve_recent_turns = preserve_recent_turns
        self.tool_output_ceiling_chars = tool_output_ceiling_chars
        self.enable_staged_elision = enable_staged_elision

    def estimate_tokens(self, text: str) -> int:
        """Heuristic offline token estimate (~4 chars per token)."""
        return max(1, (len(text) + 3) // 4)

    def compact(
        self,
        messages: list[dict[str, Any]],
        mandatory_anchors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compact conversation messages via staged elision and token management.

        Guarantees:
        - System instructions (role='system') are NEVER removed.
        - Verifier feedback or defect injection messages are NEVER removed.
        - Recent turns (last preserve_recent_turns) are prioritized and preserved intact.
        - Stage 1 (Deterministic Tool Elision): Bulky middle tool outputs (older than
          preserve_recent_turns) exceeding tool_output_ceiling_chars are deterministically
          replaced with concise static stubs, omitting complex recoverable storage hooks.
        - Stage 2 (Thought Compaction): If total tokens still exceed max_context_tokens,
          intermediate verbose thoughts in older messages are trimmed with explicit markers.
        """
        if not messages:
            return []

        anchors = mandatory_anchors or []

        # Separate system messages and conversation turns
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conversation_msgs = [m for m in messages if m.get("role") != "system"]

        cutoff_idx = max(0, len(conversation_msgs) - self.preserve_recent_turns)

        # ------------------------------------------------------------------- #
        # Stage 1: Deterministic Tool Output Elision for Middle Turns
        # ------------------------------------------------------------------- #
        staged_msgs: list[dict[str, Any]] = []

        for idx, msg in enumerate(conversation_msgs):
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            # Invariant check: verifier gate rejections, defect locations, or anchors
            is_critical = (
                "[VERIFIER_GATE_REJECTION]" in content
                or "[DEFECT_LOCATED]" in content
                or any(anchor in content for anchor in anchors)
            )

            # Check if this is an older turn tool output exceeding the ceiling
            is_tool = (role == "tool" or "tool_call_id" in msg)
            if (
                self.enable_staged_elision
                and is_tool
                and not is_critical
                and idx < cutoff_idx
                and len(content) > self.tool_output_ceiling_chars
            ):
                tool_name = msg.get("name", "unknown_tool")
                tool_call_id = msg.get("tool_call_id", "")
                orig_len = len(content)
                est_tokens = self.estimate_tokens(content)

                stub = (
                    f"[TOOL OUTPUT ELIDED: name={tool_name}"
                    + (f", tool_call_id={tool_call_id}" if tool_call_id else "")
                    + f", original_length={orig_len} chars (~{est_tokens} tokens)]"
                )

                elided_msg = dict(msg)
                elided_msg["content"] = stub
                staged_msgs.append(elided_msg)
                logger.debug(
                    f"[ContextCompactor] Elided older tool output (turn={idx}, tool={tool_name}, len={orig_len} -> {len(stub)})"
                )
            else:
                staged_msgs.append(msg)

        # Calculate estimated tokens across candidate messages
        all_candidate_msgs = system_msgs + staged_msgs
        total_chars = sum(len(str(m.get("content", ""))) for m in all_candidate_msgs)
        estimated_tokens = total_chars // 4

        # If estimated tokens are within budget, return candidate messages
        if estimated_tokens <= self.max_context_tokens:
            return all_candidate_msgs

        # ------------------------------------------------------------------- #
        # Stage 2: Compaction of Intermediate Reasoning Thoughts
        # ------------------------------------------------------------------- #
        logger.info(
            f"[ContextCompactor] Compacting context ({estimated_tokens} tokens > {self.max_context_tokens} limit)"
        )

        compacted: list[dict[str, Any]] = []
        compacted.extend(system_msgs)

        for idx, msg in enumerate(staged_msgs):
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            is_critical = (
                "[VERIFIER_GATE_REJECTION]" in content
                or "[DEFECT_LOCATED]" in content
                or any(anchor in content for anchor in anchors)
            )

            if is_critical or idx >= cutoff_idx:
                compacted.append(msg)
            else:
                # Older turn: compress intermediate verbosity if still above ceiling
                if len(content) > self.intermediate_thought_ceiling_chars:
                    head = content[: self.intermediate_thought_ceiling_chars // 2]
                    tail = content[-self.intermediate_thought_ceiling_chars // 4 :]
                    pruned_content = (
                        f"{head}\n... [TRUNCATED {len(content) - len(head) - len(tail)} chars of intermediate reasoning] ...\n{tail}"
                    )
                    pruned_msg = dict(msg)
                    pruned_msg["content"] = pruned_content
                    compacted.append(pruned_msg)
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
