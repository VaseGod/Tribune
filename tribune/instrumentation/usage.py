"""Usage metering: tokens, turns, and cost per (case, program) task.

The pipeline owns one :class:`UsageRecorder` and hands it to the providers (which
record per-call token usage) and the state machine (which records proposer /
verifier round-trips). The pipeline scopes it: ``start_case`` -> ``start_task`` ->
... -> ``finish_task``, which prices the accumulated usage against the cost model
and returns a :class:`~tribune.types.TaskUsage` that is attached to the program
outcome.

Everything here is additive observability. Recording failures must never affect a
case outcome, and a missing recorder (``None``) is always a valid configuration —
agents and providers degrade to exactly the previous behavior.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from ..eval.costmodel import CostModel
from ..types import ModelCallUsage, ProgramId, TaskUsage
from . import tracing

#: tokenizer_id used when tokens are estimated deterministically (offline provider
#: or a serving backend that did not return a usage block).
ESTIMATOR_TOKENIZER_ID = "ws-estimator-v1"


def estimate_tokens(text: str) -> int:
    """Deterministic offline token estimate (~4 characters per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class UsageRecorder:
    def __init__(self, cost_model: CostModel | None = None, pricing_date: date | None = None) -> None:
        self.cost_model = cost_model
        self.pricing_date = pricing_date
        self._case_id = ""
        self._language = "en"
        self._current: TaskUsage | None = None
        self.history: list[TaskUsage] = []

    # -- scoping (driven by the pipeline) ------------------------------------ #

    def start_case(self, case_id: str, language: str = "en") -> None:
        self._case_id = case_id
        self._language = language or "en"
        self._current = None

    def start_task(self, program: ProgramId | str) -> None:
        prog_val = program.value if isinstance(program, ProgramId) else str(program)
        self._current = TaskUsage(
            case_id=self._case_id, program=prog_val, language=self._language
        )

    def finish_task(self) -> TaskUsage | None:
        """Price and return the current task's usage; the task scope ends here."""
        task = self._current
        self._current = None
        if task is None:
            return None
        task.cache_hit_ratio = self.calculate_cache_hit_ratio(task)
        if self.cost_model is not None:
            on = self.pricing_date or date.today()
            task.cost_usd, task.cost_backend_id = self.cost_model.cost_of_task(task, on)
        tracing.log(
            "task_usage",
            case_id=task.case_id,
            program=task.program,
            language=task.language,
            turns=task.turns,
            tokens_input=task.tokens_input,
            tokens_output=task.tokens_output,
            cost_usd=task.cost_usd,
            cache_hit_ratio=task.cache_hit_ratio,
            early_exit_step=task.early_exit_step,
            tokens_saved_estimate=task.tokens_saved_estimate,
        )
        self.history.append(task)
        return task

    # -- recording (driven by agents and providers) -------------------------- #

    def record_turn(self, role: str) -> None:
        if self._current is None:
            self.start_task("simulation")
        assert self._current is not None
        self._current.turns += 1
        if role == "proposer":
            self._current.proposer_turns += 1
        elif role == "verifier":
            self._current.verifier_turns += 1

    def record_call(
        self,
        role: str,
        model: str,
        tokenizer_id: str,
        tokens_input: int,
        tokens_output: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        estimated: bool = False,
        active_experts: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        if self._current is None:
            self.start_task("general")
        assert self._current is not None
        call = ModelCallUsage(
            role=role,
            model=model,
            tokenizer_id=tokenizer_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            estimated=estimated,
            active_experts=active_experts,
            timestamp=timestamp
            or datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        )
        cur = self._current
        cur.calls.append(call)
        cur.tokens_input += tokens_input
        cur.tokens_output += tokens_output
        cur.cache_read_tokens += cache_read_tokens
        cur.cache_write_tokens += cache_write_tokens
        if active_experts is not None:
            cur.active_experts = active_experts
        if tokenizer_id not in cur.tokenizer_ids:
            cur.tokenizer_ids.append(tokenizer_id)
        cur.estimated = cur.estimated or estimated
        cur.cache_hit_ratio = self.calculate_cache_hit_ratio(cur)

    def record_crc_early_exit(
        self,
        early_exit_step: int,
        max_steps: int,
        tokens_per_step_estimate: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a Conformal Risk Control (CRC) inline early-termination event."""
        if self._current is None:
            self.start_task("simulation_crc")
        assert self._current is not None

        saved = max(0, (max_steps - early_exit_step) * tokens_per_step_estimate)
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")

        cur = self._current
        cur.early_exit_step = early_exit_step
        cur.tokens_saved_estimate = saved
        breach_record = {
            "early_exit_step": early_exit_step,
            "max_steps": max_steps,
            "tokens_saved_estimate": saved,
            "timestamp": ts,
            **(details or {}),
        }
        cur.crc_breach_events.append(breach_record)

        tracing.log(
            "crc_early_exit",
            case_id=cur.case_id,
            early_exit_step=early_exit_step,
            tokens_saved=saved,
            timestamp=ts,
        )

    def record_cache_stats(
        self,
        cache_hit_ratio: float,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Update prompt caching metrics (e.g. from IndexPool or prefix cache)."""
        if self._current is None:
            self.start_task("cache_tracker")
        assert self._current is not None
        cur = self._current
        cur.cache_hit_ratio = max(0.0, min(1.0, float(cache_hit_ratio)))
        cur.cache_read_tokens += cache_read_tokens
        cur.cache_write_tokens += cache_write_tokens

    def calculate_cache_hit_ratio(self, task: TaskUsage | None = None) -> float:
        """Calculate prompt cache hit ratio: cache_read / (tokens_input + cache_read)."""
        t = task or self._current
        if t is None:
            return 0.0
        total_prompt = t.tokens_input + t.cache_read_tokens
        if total_prompt == 0:
            return t.cache_hit_ratio
        return round(float(t.cache_read_tokens) / float(total_prompt), 4)

