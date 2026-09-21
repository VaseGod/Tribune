"""Background Asynchronous Task Pool for Spoken Dialogue Tool Dispatch.

Enables spoken dialogues to trigger database queries, verification rules, or OCR analysis
in the background without blocking or stuttering real-time audio playback and streaming synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Outcome of an asynchronous background tool call."""

    call_id: str
    tool_name: str
    success: bool
    result: Any
    latency_ms: float
    error: str | None = None


class BackgroundToolExecutor:
    """Asynchronous background worker pool for spoken dialogue tool execution."""

    def __init__(self, max_concurrency: int = 5) -> None:
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}
        self.completed_results: dict[str, ToolExecutionResult] = {}
        self.total_dispatched = 0

    async def dispatch_tool_call(
        self,
        tool_name: str,
        tool_fn: Callable[..., Coroutine[Any, Any, Any] | Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Dispatch a tool call to run asynchronously in the background.

        Returns immediately with a call_id without blocking the audio loop.
        """
        call_id = f"tool_call_{uuid.uuid4().hex[:8]}"
        self.total_dispatched += 1

        async def _runner() -> None:
            start_t = time.perf_counter()
            async with self.semaphore:
                try:
                    if asyncio.iscoroutinefunction(tool_fn):
                        res = await tool_fn(*args, **kwargs)
                    else:
                        loop = asyncio.get_event_loop()
                        res = await loop.run_in_executor(None, lambda: tool_fn(*args, **kwargs))

                    dur_ms = (time.perf_counter() - start_t) * 1000.0
                    out = ToolExecutionResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        success=True,
                        result=res,
                        latency_ms=dur_ms,
                    )
                except Exception as exc:
                    dur_ms = (time.perf_counter() - start_t) * 1000.0
                    logger.error(f"[BackgroundToolExecutor] Tool {tool_name} failed: {exc}")
                    out = ToolExecutionResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        success=False,
                        result=None,
                        latency_ms=dur_ms,
                        error=str(exc),
                    )

                self.completed_results[call_id] = out
                self.active_tasks.pop(call_id, None)

        task = asyncio.create_task(_runner())
        self.active_tasks[call_id] = task
        logger.info(f"[BackgroundToolExecutor] Dispatched {tool_name} (call_id: {call_id}) to background worker")
        return call_id

    def get_result(self, call_id: str) -> ToolExecutionResult | None:
        """Poll or retrieve result of completed background tool call."""
        return self.completed_results.get(call_id)

    async def wait_for_call(self, call_id: str, timeout: float = 5.0) -> ToolExecutionResult | None:
        """Optionally wait for a specific background tool call to complete."""
        task = self.active_tasks.get(call_id)
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except TimeoutError:
                return None
        return self.completed_results.get(call_id)

    def stats(self) -> dict[str, Any]:
        return {
            "total_dispatched": self.total_dispatched,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_results),
        }


__all__ = [
    "ToolExecutionResult",
    "BackgroundToolExecutor",
]
