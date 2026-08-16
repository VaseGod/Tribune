"""Cost model — pricing is data, not code.

Backends are priced by entries in a JSON file (packaged default:
``tribune/eval/pricing.json``, overridable via ``TRIBUNE_PRICING_PATH``). Two
kinds of pricing are supported:

* **api** — per-million-token rates for input / output, with optional cache-read
  and cache-write rates,
* **self_hosted** — an amortized ``$/GPU-hour`` divided by measured throughput
  (tokens/second), so cost-per-task reflects what the deploying organization
  actually pays for its own hardware.

Every rate carries ``effective_from`` / ``effective_until`` dates so promotional
pricing expires correctly (e.g. a launch rate that reverts to list price on a
given day). Rate resolution picks the entry valid on the accounting date; when
several overlap, the most recently effective one wins.

The honest unit of account for TRIBUNE is **cost per completed verification**,
never per-token list price — and a correct abstention is a completed task,
reported at its actual (low) cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatch

from ..types import ModelCallUsage, TaskUsage

_PACKAGED_PRICING = os.path.join(os.path.dirname(__file__), "pricing.json")


@dataclass(frozen=True)
class Rate:
    effective_from: date
    effective_until: date | None
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    cache_read_per_m: float | None = None
    cache_write_per_m: float | None = None
    gpu_hour_usd: float | None = None
    throughput_tokens_per_s: float | None = None
    note: str = ""

    def covers(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.effective_until is None or on <= self.effective_until


@dataclass(frozen=True)
class BackendPricing:
    backend_id: str
    kind: str  # "api" | "self_hosted" | "free"
    model_patterns: tuple[str, ...]
    rates: tuple[Rate, ...]

    def matches(self, model_name: str) -> bool:
        name = model_name.lower()
        return any(fnmatch(name, p.lower()) for p in self.model_patterns)

    def rate_on(self, on: date) -> Rate | None:
        valid = [r for r in self.rates if r.covers(on)]
        if not valid:
            return None
        return max(valid, key=lambda r: r.effective_from)


def _parse_date(raw: str | None) -> date | None:
    return date.fromisoformat(raw) if raw else None


def _parse_rate(obj: dict) -> Rate:
    frm = _parse_date(obj.get("effective_from"))
    if frm is None:
        raise ValueError("rate is missing 'effective_from'")
    return Rate(
        effective_from=frm,
        effective_until=_parse_date(obj.get("effective_until")),
        input_per_m=float(obj.get("input_per_m", 0.0)),
        output_per_m=float(obj.get("output_per_m", 0.0)),
        cache_read_per_m=(
            float(obj["cache_read_per_m"]) if obj.get("cache_read_per_m") is not None else None
        ),
        cache_write_per_m=(
            float(obj["cache_write_per_m"]) if obj.get("cache_write_per_m") is not None else None
        ),
        gpu_hour_usd=(float(obj["gpu_hour_usd"]) if obj.get("gpu_hour_usd") is not None else None),
        throughput_tokens_per_s=(
            float(obj["throughput_tokens_per_s"])
            if obj.get("throughput_tokens_per_s") is not None
            else None
        ),
        note=str(obj.get("note", "")),
    )


@dataclass(frozen=True)
class ParetoPoint:
    """A point on the cost-versus-parity evaluation plane."""

    label: str
    backend_id: str
    cost_per_1k: float
    accuracy: float
    parity_score: float
    cost_savings_pct: float
    is_pareto_optimal: bool
    dominated_by: tuple[str, ...] = ()


class CostModel:
    def __init__(self, backends: list[BackendPricing]) -> None:
        self.backends = backends

    @classmethod
    def load(cls, path: str | None = None) -> CostModel:
        with open(path or _PACKAGED_PRICING, encoding="utf-8") as fh:
            payload = json.load(fh)
        backends: list[BackendPricing] = []
        for b in payload.get("backends", []):
            backends.append(
                BackendPricing(
                    backend_id=str(b["backend_id"]),
                    kind=str(b.get("kind", "api")),
                    model_patterns=tuple(b.get("model_patterns", [])),
                    rates=tuple(_parse_rate(r) for r in b.get("rates", [])),
                )
            )
        return cls(backends)

    def match(self, model_name: str) -> BackendPricing | None:
        """First backend whose pattern matches wins — order the data file accordingly."""
        for b in self.backends:
            if b.matches(model_name):
                return b
        return None

    # -- costing ------------------------------------------------------------- #

    def cost_of_call(self, call: ModelCallUsage, on: date) -> tuple[float, str | None]:
        backend = self.match(call.model)
        if backend is None:
            return 0.0, None
        rate = backend.rate_on(on)
        if rate is None:
            return 0.0, backend.backend_id
        if backend.kind == "free":
            return 0.0, backend.backend_id
        if backend.kind == "self_hosted":
            if not rate.gpu_hour_usd or not rate.throughput_tokens_per_s:
                return 0.0, backend.backend_id
            total = call.tokens_input + call.tokens_output
            hours = total / rate.throughput_tokens_per_s / 3600.0
            return hours * rate.gpu_hour_usd, backend.backend_id
        # kind == "api"
        cache_read = min(call.cache_read_tokens, call.tokens_input)
        uncached_input = call.tokens_input - cache_read
        read_rate = rate.cache_read_per_m if rate.cache_read_per_m is not None else rate.input_per_m
        write_rate = rate.cache_write_per_m if rate.cache_write_per_m is not None else 0.0
        cost = (
            uncached_input * rate.input_per_m
            + cache_read * read_rate
            + call.cache_write_tokens * write_rate
            + call.tokens_output * rate.output_per_m
        ) / 1_000_000.0
        return cost, backend.backend_id

    def cost_of_task(self, task: TaskUsage, on: date) -> tuple[float, str | None]:
        total = 0.0
        backend_ids: list[str] = []
        for call in task.calls:
            cost, backend_id = self.cost_of_call(call, on)
            total += cost
            if backend_id and backend_id not in backend_ids:
                backend_ids.append(backend_id)
        return total, (",".join(backend_ids) if backend_ids else None)

    def compute_pareto_frontier(
        self,
        points_data: list[dict[str, Any]],
        reference_label: str = "fp16",
    ) -> list[ParetoPoint]:
        """Compute the Pareto frontier optimizing (minimize cost, maximize accuracy/parity).

        A candidate A is dominated by candidate B if B has cost <= A and parity/accuracy >= A
        with at least one strict inequality.
        """
        if not points_data:
            return []

        # Find reference cost
        ref_cost = next((p["cost_per_1k"] for p in points_data if p["label"] == reference_label), None)
        if ref_cost is None or ref_cost == 0:
            ref_cost = max((p["cost_per_1k"] for p in points_data), default=1.0) or 1.0

        pareto_points: list[ParetoPoint] = []
        n = len(points_data)

        for i in range(n):
            p1 = points_data[i]
            cost1 = float(p1["cost_per_1k"])
            acc1 = float(p1.get("accuracy", 1.0))
            par1 = float(p1.get("parity_score", acc1))
            savings_pct = max(0.0, (1.0 - (cost1 / ref_cost)) * 100.0) if ref_cost > 0 else 0.0

            dominated_by: list[str] = []
            for j in range(n):
                if i == j:
                    continue
                p2 = points_data[j]
                cost2 = float(p2["cost_per_1k"])
                acc2 = float(p2.get("accuracy", 1.0))
                par2 = float(p2.get("parity_score", acc2))

                # Domination condition: p2 is as cheap as p1 and as accurate/parity as p1, and strictly better in one
                if cost2 <= cost1 and par2 >= par1 and (cost2 < cost1 or par2 > par1):
                    dominated_by.append(p2["label"])

            pareto_points.append(
                ParetoPoint(
                    label=p1["label"],
                    backend_id=p1.get("backend_id", p1["label"]),
                    cost_per_1k=cost1,
                    accuracy=acc1,
                    parity_score=par1,
                    cost_savings_pct=round(savings_pct, 2),
                    is_pareto_optimal=len(dominated_by) == 0,
                    dominated_by=tuple(dominated_by),
                )
            )

        return pareto_points


def default_cost_model() -> CostModel:
    """Return the CostModel loaded from the default packaged pricing.json or TRIBUNE_PRICING_PATH."""
    path = os.environ.get("TRIBUNE_PRICING_PATH", _PACKAGED_PRICING)
    return CostModel.load(path)


__all__ = [
    "Rate",
    "BackendPricing",
    "ParetoPoint",
    "CostModel",
    "default_cost_model",
]

