#!/usr/bin/env python3
"""Validate the TRIBUNE backend registry.

Schema check (always) + optional weight-URL link check (``--check-links``).

The schema is a pydantic model, so validation reuses the same strict-typing
discipline as the rest of TRIBUNE. The link check is opt-in because CI runs
offline/network-restricted and model-hub availability is flaky; it only ever
touches model-hub URLs, never a government or agency endpoint.

Exit code 0 on success, 1 on any schema or (when requested) link failure.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["proposer", "verifier"]
WeightsStatus = Literal["downloadable", "announced_only", "unverified", "api_only"]
Serving = Literal["llama.cpp", "vllm", "sglang", "api"]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    candidate_id: str
    name: str
    vendor: str
    params: str
    roles: list[Role]
    role_note: str = ""
    license: str
    weights_status: WeightsStatus
    weights_url: str | None = None
    weights_verified: bool = False
    serving: list[Serving]
    quant_formats: list[str] = Field(default_factory=list)
    context_window: int
    measured_kappa: float | None = None
    measured_cost_per_task: float | None = None
    promo_end_date: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> Candidate:
        if not self.roles:
            raise ValueError(f"{self.candidate_id}: must declare at least one role")
        # A downloadable model must carry a URL; a verified one must be downloadable.
        if self.weights_status == "downloadable" and not self.weights_url:
            raise ValueError(f"{self.candidate_id}: weights_status=downloadable requires a weights_url")
        # Only genuinely-available weights can be marked verified; this also
        # rejects verifying an announced-only or unverified candidate.
        if self.weights_verified and self.weights_status not in ("downloadable", "api_only"):
            raise ValueError(
                f"{self.candidate_id}: weights_verified=true only valid when downloadable/api_only"
            )
        if self.context_window <= 0:
            raise ValueError(f"{self.candidate_id}: context_window must be positive")
        return self


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    seed_set: str
    candidates: list[Candidate]

    @model_validator(mode="after")
    def _unique_ids(self) -> Registry:
        ids = [c.candidate_id for c in self.candidates]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate candidate_id(s): {sorted(dupes)}")
        return self


def load_registry(path: str) -> Registry:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return Registry.model_validate(payload)


def check_links(registry: Registry, timeout: float = 10.0) -> list[str]:
    """HEAD/GET each downloadable weight URL; return a list of problems."""
    problems: list[str] = []
    for c in registry.candidates:
        if c.weights_status not in ("downloadable",) or not c.weights_url:
            continue
        req = urllib.request.Request(c.weights_url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status >= 400:
                    problems.append(f"{c.candidate_id}: {c.weights_url} -> HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            # Some hubs reject HEAD; a 405/403 still means the URL exists.
            if exc.code not in (403, 405):
                problems.append(f"{c.candidate_id}: {c.weights_url} -> HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            problems.append(f"{c.candidate_id}: {c.weights_url} -> {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate backends/registry.yaml")
    parser.add_argument("--path", default="backends/registry.yaml")
    parser.add_argument("--check-links", action="store_true", help="also HEAD-check weight URLs")
    args = parser.parse_args(argv)

    try:
        registry = load_registry(args.path)
    except Exception as exc:  # noqa: BLE001 - surface any schema error to the user
        print(f"SCHEMA INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"schema OK: {len(registry.candidates)} candidate(s), seed_set={registry.seed_set}")
    downloadable = [c.candidate_id for c in registry.candidates if c.weights_status == "downloadable"]
    announced = [c.candidate_id for c in registry.candidates if c.weights_status == "announced_only"]
    print(f"  downloadable: {downloadable}")
    print(f"  announced-only (NOT deployable): {announced}")

    if args.check_links:
        problems = check_links(registry)
        if problems:
            print("LINK CHECK FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  ! {p}", file=sys.stderr)
            return 1
        print("link check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
