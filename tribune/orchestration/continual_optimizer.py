"""Continual Optimizer & Self-Patching Agent Harness Evolution Engine.

Ingests failed execution traces, schema deviations, and boundary violations across
pipeline runs and synthesizes candidate prompt modifications, routing adjustments,
and criteria clarifications for target agent harnesses.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from ..types import (
    AgentHarnessPatch,
    CaseRunResult,
    FailureCategory,
    FailureTrace,
    PatchStatus,
    PatchType,
    ProgramId,
    PromotionMetrics,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContinualOptimizer:
    """Ingests failure telemetry, clusters failure patterns, and synthesizes candidate agent patches."""

    def __init__(self) -> None:
        self.failure_traces: list[FailureTrace] = []
        self.candidate_patches: dict[str, AgentHarnessPatch] = {}
        self.active_promoted_patches: dict[str, AgentHarnessPatch] = {}
        self.patch_history: list[AgentHarnessPatch] = []

    def ingest_failure_trace(self, trace: FailureTrace | dict[str, Any]) -> FailureTrace:
        """Ingest an individual failure trace payload."""
        if isinstance(trace, dict):
            # Normalize category
            cat_str = trace.get("category", "general_failure")
            try:
                cat = FailureCategory(cat_str)
            except ValueError:
                cat = FailureCategory.GENERAL_FAILURE

            prog_val = trace.get("program")
            prog = None
            if prog_val:
                try:
                    prog = ProgramId(prog_val)
                except ValueError:
                    prog = None

            trace_obj = FailureTrace(
                trace_id=trace.get("trace_id") or f"trace_{len(self.failure_traces) + 1}_{hashlib.sha256(str(trace).encode()).hexdigest()[:8]}",
                case_id=trace.get("case_id", "unknown"),
                program=prog,
                agent_id=trace.get("agent_id", "unknown"),
                category=cat,
                error_message=trace.get("error_message") or "; ".join(trace.get("reasons", [])) or "Execution failure",
                context_data=dict(trace.get("details", trace)),
                timestamp=_utcnow(),
            )
        else:
            trace_obj = trace

        self.failure_traces.append(trace_obj)
        return trace_obj

    def ingest_pipeline_result(self, result: CaseRunResult) -> list[FailureTrace]:
        """Ingest all failure traces and unapproved outcomes from a CaseRunResult."""
        ingested: list[FailureTrace] = []
        for fail_dict in result.failure_traces:
            ingested.append(self.ingest_failure_trace(fail_dict))

        # Check for unapproved outcomes
        for outcome in result.outcomes:
            if outcome.verdict and not outcome.verdict.approved and outcome.assessment:
                trace_dict = {
                    "case_id": result.case_id,
                    "program": outcome.program.value,
                    "agent_id": "verifier",
                    "category": "predicate_error" if outcome.verdict.unsupported_claims else "citation_mismatch",
                    "error_message": "; ".join(outcome.verdict.reasons),
                    "details": {
                        "missing_citations": outcome.verdict.missing_citations,
                        "unsupported_claims": outcome.verdict.unsupported_claims,
                        "incomplete_coverage": outcome.verdict.incomplete_coverage,
                    },
                }
                ingested.append(self.ingest_failure_trace(trace_dict))
        return ingested

    def ingest_traces(self, traces: list[FailureTrace | dict[str, Any]]) -> list[FailureTrace]:
        """Ingest a batch of failure traces."""
        return [self.ingest_failure_trace(t) for t in traces]

    def cluster_failures(self) -> dict[str, list[FailureTrace]]:
        """Group failure traces by (category, target_agent, target_program) clusters."""
        clusters: dict[str, list[FailureTrace]] = {}
        for trace in self.failure_traces:
            prog_key = trace.program.value if trace.program else "all"
            cluster_key = f"{trace.category.value}::{trace.agent_id}::{prog_key}"
            clusters.setdefault(cluster_key, []).append(trace)
        return clusters

    def synthesize_patches(self) -> list[AgentHarnessPatch]:
        """Formulate candidate prompt modifications and routing patches based on clustered failure modes."""
        clusters = self.cluster_failures()
        synthesized: list[AgentHarnessPatch] = []

        for cluster_key, traces in clusters.items():
            if not traces:
                continue

            parts = cluster_key.split("::")
            cat_val = parts[0]
            agent_id = parts[1]
            prog_val = parts[2]
            target_prog = ProgramId(prog_val) if prog_val != "all" else None

            patch_id = f"patch_{cat_val}_{agent_id}_{prog_val}_{len(self.candidate_patches) + 1}"
            sample_errors = [t.error_message for t in traces[:3]]
            sample_err_text = "; ".join(sample_errors)

            if cat_val == FailureCategory.CITATION_MISMATCH.value:
                prompt_patch = (
                    "CRITICAL STATUTORY GROUNDING DIRECTIVE:\n"
                    "- You MUST cite an exact active statutory citation ID for every evaluated criterion outcome.\n"
                    "- Uncited claims or invented citations are strictly prohibited and will be rejected.\n"
                    f"- Enforce active citation keys for program {prog_val.upper()}."
                )
                patch = AgentHarnessPatch(
                    patch_id=patch_id,
                    target_agent=agent_id if agent_id != "unknown" else "proposer",
                    target_program=target_prog,
                    patch_type=PatchType.PROMPT_REFINEMENT,
                    description=f"Enforce strict statutory citation grounding to fix {len(traces)} citation mismatch(es).",
                    original_prompt_template="Standard proposer prompt",
                    patched_prompt_template=prompt_patch,
                    rationale=f"Observed citation failures: {sample_err_text[:200]}",
                    status=PatchStatus.PROPOSED,
                )
            elif cat_val == FailureCategory.GOVERNANCE_VIOLATION.value:
                guardrail_patch = (
                    "SECURITY & GOVERNANCE INVARIANT ENFORCEMENT:\n"
                    "- Never inspect hidden test suites, ground truth keys, or internal test files.\n"
                    "- Never override statutory rules or bypass ActionGate pre-conditions.\n"
                    "- Never emit non-deterministic reasoning monologues <think> or unverified thoughts in visible output."
                )
                patch = AgentHarnessPatch(
                    patch_id=patch_id,
                    target_agent=agent_id if agent_id != "unknown" else "action_gate",
                    target_program=target_prog,
                    patch_type=PatchType.GUARDRAIL_TUNING,
                    description=f"Harden sandbox and eliminate policy override attempts ({len(traces)} incident(s)).",
                    original_prompt_template="Standard governance policy",
                    patched_prompt_template=guardrail_patch,
                    routing_overrides={"sandbox_mode": True, "strict_preconditions": True},
                    rationale=f"Observed governance violations: {sample_err_text[:200]}",
                    status=PatchStatus.PROPOSED,
                )
            elif cat_val == FailureCategory.PREDICATE_ERROR.value:
                predicate_patch = (
                    "STATUTORY PREDICATE DETERMINATION GUIDANCE:\n"
                    "- Compute exact arithmetic against 2026 FPL guidelines, income limits, and asset caps.\n"
                    "- If evidence is ambiguous, missing, or borderline, abstain and set outcome to indeterminate/unknown.\n"
                    "- Re-derive boolean rule predicates step-by-step with zero assumptions."
                )
                patch = AgentHarnessPatch(
                    patch_id=patch_id,
                    target_agent=agent_id if agent_id != "unknown" else "proposer",
                    target_program=target_prog,
                    patch_type=PatchType.CRITERIA_CLARIFICATION,
                    description=f"Clarify predicate evaluation logic to eliminate {len(traces)} recomputation mismatch(es).",
                    original_prompt_template="Standard predicate evaluation prompt",
                    patched_prompt_template=predicate_patch,
                    rationale=f"Observed predicate errors: {sample_err_text[:200]}",
                    status=PatchStatus.PROPOSED,
                )
            elif cat_val == FailureCategory.COVERAGE_GAP.value:
                coverage_patch = (
                    "EXHAUSTIVE STATUTORY COVERAGE REQUIREMENT:\n"
                    "- You must evaluate every mandatory criterion defined in the program ruleset.\n"
                    "- Omission of any required statutory criterion constitutes an automatic verification failure."
                )
                patch = AgentHarnessPatch(
                    patch_id=patch_id,
                    target_agent=agent_id if agent_id != "unknown" else "proposer",
                    target_program=target_prog,
                    patch_type=PatchType.PROMPT_REFINEMENT,
                    description=f"Enforce mandatory full criteria coverage for {prog_val.upper()}.",
                    original_prompt_template="Standard criteria coverage",
                    patched_prompt_template=coverage_patch,
                    rationale=f"Observed coverage gaps: {sample_err_text[:200]}",
                    status=PatchStatus.PROPOSED,
                )
            else:
                patch = AgentHarnessPatch(
                    patch_id=patch_id,
                    target_agent=agent_id if agent_id != "unknown" else "proposer",
                    target_program=target_prog,
                    patch_type=PatchType.ROUTING_ADJUSTMENT,
                    description=f"Dynamic Pareto routing escalation for {prog_val.upper()} failures.",
                    routing_overrides={"force_tier2_escalation": True, "retry_budget": 3},
                    rationale=f"Observed general failures: {sample_err_text[:200]}",
                    status=PatchStatus.PROPOSED,
                )

            self.candidate_patches[patch_id] = patch
            synthesized.append(patch)

        return synthesized

    def propose_patch(
        self,
        target_agent: str,
        target_program: ProgramId | None,
        patch_type: PatchType,
        description: str,
        patched_prompt_template: str = "",
        routing_overrides: dict[str, Any] | None = None,
        rationale: str = "",
    ) -> AgentHarnessPatch:
        """Manually construct and register a proposed harness patch."""
        patch_id = f"patch_{patch_type.value}_{target_agent}_{len(self.candidate_patches) + 1}"
        patch = AgentHarnessPatch(
            patch_id=patch_id,
            target_agent=target_agent,
            target_program=target_program,
            patch_type=patch_type,
            description=description,
            patched_prompt_template=patched_prompt_template,
            routing_overrides=routing_overrides or {},
            rationale=rationale,
            status=PatchStatus.PROPOSED,
        )
        self.candidate_patches[patch_id] = patch
        return patch

    def apply_candidate_patch(self, patch: AgentHarnessPatch) -> AgentHarnessPatch:
        """Stage a candidate patch for canary evaluation."""
        staged = patch.model_copy(update={"status": PatchStatus.CANARY_TESTED})
        self.candidate_patches[patch.patch_id] = staged
        return staged

    def promote_patch(self, patch_id: str, metrics: PromotionMetrics) -> AgentHarnessPatch:
        """Commit a candidate patch to active configuration if promotion metrics approved."""
        if patch_id not in self.candidate_patches:
            raise KeyError(f"No candidate patch found with id '{patch_id}'")

        if not metrics.approved:
            rejected = self.candidate_patches[patch_id].model_copy(update={"status": PatchStatus.REJECTED})
            self.candidate_patches[patch_id] = rejected
            self.patch_history.append(rejected)
            logger.warning(f"Patch '{patch_id}' rejected by promotion gate: {'; '.join(metrics.reasons)}")
            return rejected

        promoted = self.candidate_patches[patch_id].model_copy(update={"status": PatchStatus.PROMOTED})
        self.active_promoted_patches[patch_id] = promoted
        self.candidate_patches[patch_id] = promoted
        self.patch_history.append(promoted)
        logger.info(f"Patch '{patch_id}' successfully promoted with parity ratio {metrics.baseline_parity_ratio:.4f}")
        return promoted

    def rollback_patch(self, patch_id: str, reason: str = "") -> AgentHarnessPatch:
        """Rollback an active promoted patch."""
        if patch_id in self.active_promoted_patches:
            del self.active_promoted_patches[patch_id]

        if patch_id in self.candidate_patches:
            rolled_back = self.candidate_patches[patch_id].model_copy(
                update={"status": PatchStatus.ROLLED_BACK, "rationale": f"Rolled back: {reason}"}
            )
            self.candidate_patches[patch_id] = rolled_back
            self.patch_history.append(rolled_back)
            return rolled_back

        raise KeyError(f"Cannot rollback: patch '{patch_id}' not found")

    def get_active_patches(self) -> list[AgentHarnessPatch]:
        """Return all active promoted patches."""
        return list(self.active_promoted_patches.values())

    def get_patch(self, patch_id: str) -> AgentHarnessPatch | None:
        """Retrieve patch by ID."""
        return self.candidate_patches.get(patch_id)

    def stats(self) -> dict[str, Any]:
        """Return telemetry and patch status statistics."""
        return {
            "failure_traces_count": len(self.failure_traces),
            "candidate_patches_count": len(self.candidate_patches),
            "promoted_patches_count": len(self.active_promoted_patches),
            "patch_history_count": len(self.patch_history),
        }


__all__ = ["ContinualOptimizer"]
