"""CasePipeline — Plugin-Based Modular Execution Harness & Asynchronous Tool Cache.

For a synthetic (or real-intake) case it:
1. Decomposes the sub-task DAG (navigator).
2. Ingests documents into access-controlled case memory and shared workspace context (gather).
3. Executes modular plugins (pre-execution hooks, post-execution validators, audit loggers,
   governance judges, and tool caching).
4. Runs the ASSESS -> VERIFY -> {PREPARE | ABSTAIN | REPLAN} state machine per target program.
5. Emits an immutable cryptographic audit trail and auto-checkpoints for crash recovery.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..abstention.calibration import Calibrator
from ..agents.eligibility import EligibilityProposer
from ..agents.navigator import Navigator
from ..agents.preparer import Preparer
from ..agents.verifier import Verifier
from ..config import TribuneSettings, get_settings
from ..context.workspace import WorkspaceContext
from ..corpus.rule_store import make_rule_store
from ..eval.costmodel import CostModel
from ..governance.action_gate import ActionGate
from ..governance.audit import AuditLog, CheckpointManager
from ..governance.judge import get_default_judge
from ..ingestion.base import make_doc_ingest
from ..instrumentation.tracing import init_tracing, span
from ..instrumentation.usage import UsageRecorder
from ..memory.consolidation import MemoryConsolidator, extract_statutory_constraints
from ..memory.partitions import CasePartition, PartitionManager
from ..providers.base import get_provider_for_role
from ..types import (
    CaseRunResult,
    DeltaPatch,
    Evidence,
    PatchOperationType,
    PatchProvenance,
    ProgramOutcome,
    SMState,
    SyntheticCase,
)
from .dag import AsyncDAGRunner, DAGRunner, Task
from .router import RecoveryPlan, Router
from .state_machine import CaseStateMachine


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TrajectoryFrame:
    """An immutable record of one step within the case trajectory."""
    frame_id: str
    state: SMState
    agent: str
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class TrajectoryBuffer:
    """Immutable append-only trajectory buffer optimizing KV-cache prefix stability."""
    static_prefix: str
    frames: tuple[TrajectoryFrame, ...] = ()

    def append(
        self, state: SMState, agent: str, action: str, data: dict[str, Any] | None = None
    ) -> TrajectoryBuffer:
        new_frame = TrajectoryFrame(
            frame_id=f"frame_{len(self.frames) + 1}",
            state=state,
            agent=agent,
            action=action,
            data=dict(data) if data else {},
        )
        return TrajectoryBuffer(
            static_prefix=self.static_prefix,
            frames=(*self.frames, new_frame),
        )

    @property
    def latest_evidence(self) -> list[Evidence]:
        for frame in reversed(self.frames):
            if "evidence" in frame.data and isinstance(frame.data["evidence"], list):
                return list(frame.data["evidence"])
        return []


# --------------------------------------------------------------------------- #
# Asynchronous Tool-Result Cache (LRU + TTL + SHA-256 Keying)
# --------------------------------------------------------------------------- #


@dataclass
class _CacheEntry:
    value: Any
    created_at: float
    expires_at: float
    last_accessed_at: float
    hits: int = 0


class ToolResultCache:
    """Thread-safe & async-compatible Key-Value Tool Result Cache with LRU + TTL eviction.

    Intercepts identical statutory corpus lookups and deterministic tool calls using a
    deterministic SHA-256 hash of (tool_name, normalized_arguments, jurisdiction_scope).
    Targets a ~35% reduction in external provider latency.
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000) -> None:
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.estimated_latency_savings_ms = 0.0

    @staticmethod
    def compute_cache_key(tool_name: str, arguments: dict[str, Any] | Any, jurisdiction: str = "EX") -> str:
        """Compute deterministic SHA-256 cache key from tool name, normalized arguments, and jurisdiction."""
        try:
            norm_args = json.dumps(arguments, sort_keys=True, default=str)
        except Exception:
            norm_args = str(arguments)
        key_raw = f"{tool_name.strip().lower()}::{norm_args}::{jurisdiction.strip().upper()}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get(self, tool_name: str, arguments: dict[str, Any] | Any, jurisdiction: str = "EX") -> Any | None:
        """Synchronously get cached result if valid and unexpired."""
        with self._lock:
            key = self.compute_cache_key(tool_name, arguments, jurisdiction)
            now = time.time()
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None
            if now > entry.expires_at:
                del self._cache[key]
                self.evictions += 1
                self.misses += 1
                return None

            entry.hits += 1
            entry.last_accessed_at = now
            self.hits += 1
            self.estimated_latency_savings_ms += 120.0  # Estimated average tool latency saved
            return copy.deepcopy(entry.value)

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any] | Any,
        result: Any,
        jurisdiction: str = "EX",
        ttl: float | None = None,
    ) -> None:
        """Synchronously cache a tool result."""
        with self._lock:
            key = self.compute_cache_key(tool_name, arguments, jurisdiction)
            now = time.time()
            effective_ttl = ttl if ttl is not None else self.default_ttl

            # Evict LRU if full
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].last_accessed_at)
                del self._cache[oldest_key]
                self.evictions += 1

            self._cache[key] = _CacheEntry(
                value=copy.deepcopy(result),
                created_at=now,
                expires_at=now + effective_ttl,
                last_accessed_at=now,
            )

    async def get_async(self, tool_name: str, arguments: dict[str, Any] | Any, jurisdiction: str = "EX") -> Any | None:
        """Asynchronously get cached result."""
        async with self._async_lock:
            return self.get(tool_name, arguments, jurisdiction)

    async def set_async(
        self,
        tool_name: str,
        arguments: dict[str, Any] | Any,
        result: Any,
        jurisdiction: str = "EX",
        ttl: float | None = None,
    ) -> None:
        """Asynchronously cache a tool result."""
        async with self._async_lock:
            self.set(tool_name, arguments, result, jurisdiction, ttl)

    async def get_or_compute_async(
        self,
        tool_name: str,
        arguments: dict[str, Any] | Any,
        jurisdiction: str,
        compute_fn: Any,
        ttl: float | None = None,
    ) -> Any:
        """Fetch from cache or execute async compute function and store."""
        cached = await self.get_async(tool_name, arguments, jurisdiction)
        if cached is not None:
            return cached

        if inspect.iscoroutinefunction(compute_fn):
            result = await compute_fn()
        else:
            result = await asyncio.to_thread(compute_fn)

        await self.set_async(tool_name, arguments, result, jurisdiction, ttl)
        return result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 4),
                "evictions": self.evictions,
                "estimated_latency_savings_ms": round(self.estimated_latency_savings_ms, 2),
            }


# --------------------------------------------------------------------------- #
# Modular Pipeline Plugin Architecture
# --------------------------------------------------------------------------- #


class PipelinePlugin:
    """Base interface for swappable Pipeline Plugins.

    Plugins participate in pipeline lifecycle: pre-execution context injection,
    in-flight validation, tool caching, continuous governance auditing, and post-task telemetry.
    """

    name: str = "base_plugin"

    def on_pipeline_start(self, case: SyntheticCase, workspace: WorkspaceContext | None) -> None:
        """Invoked immediately prior to DAG execution."""
        pass

    def pre_task_execute(
        self,
        task: Task,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Invoked before executing an individual task in the DAG. May return augmented context dict."""
        return None

    def post_task_execute(
        self,
        task: Task,
        outcome: ProgramOutcome | None,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
    ) -> None:
        """Invoked immediately after executing an individual task in the DAG."""
        pass

    def on_pipeline_end(
        self,
        case: SyntheticCase,
        result: CaseRunResult,
        workspace: WorkspaceContext | None,
    ) -> None:
        """Invoked after all DAG tasks and outcomes have concluded."""
        pass


class AuditLoggingPlugin(PipelinePlugin):
    """Pipeline plugin logging all state machine transitions to the audit log."""

    name = "audit_logging"

    def __init__(self, audit_log: AuditLog) -> None:
        self.audit = audit_log

    def on_pipeline_start(self, case: SyntheticCase, workspace: WorkspaceContext | None) -> None:
        self.audit.append(
            case.case_id, SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
        )


class WorkspaceSyncPlugin(PipelinePlugin):
    """Pipeline plugin syncing structured delta patches into WorkspaceContext."""

    name = "workspace_sync"

    def on_pipeline_start(self, case: SyntheticCase, workspace: WorkspaceContext | None) -> None:
        if workspace:
            workspace.apply_patch(
                DeltaPatch(
                    run_id=case.case_id,
                    agent_id="navigator",
                    operation=PatchOperationType.REPLACE,
                    path="/documents",
                    value=[d.model_dump(mode="json") for d in case.documents],
                    provenance=PatchProvenance(agent_id="navigator"),
                )
            )

    def post_task_execute(
        self,
        task: Task,
        outcome: ProgramOutcome | None,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
    ) -> None:
        if not workspace or outcome is None or task.program is None:
            return
        prog_val = task.program.value
        if outcome.assessment:
            workspace.apply_patch(
                DeltaPatch(
                    run_id=outcome.assessment.case_id,
                    agent_id=f"proposer_{prog_val}",
                    operation=PatchOperationType.REPLACE,
                    path=f"/assessments/{prog_val}",
                    value=outcome.assessment.model_dump(mode="json"),
                    provenance=PatchProvenance(
                        agent_id=f"proposer_{prog_val}",
                        task_id=task.task_id,
                        citation_keys=[c.citation_id for c in outcome.assessment.citations],
                        confidence=outcome.assessment.self_confidence,
                    ),
                )
            )
        if outcome.verdict:
            workspace.apply_patch(
                DeltaPatch(
                    run_id=workspace.case_id,
                    agent_id=f"verifier_{prog_val}",
                    operation=PatchOperationType.REPLACE,
                    path=f"/verification_verdicts/{prog_val}",
                    value=outcome.verdict.model_dump(mode="json"),
                    provenance=PatchProvenance(
                        agent_id=f"verifier_{prog_val}",
                        task_id=task.task_id,
                        confidence=outcome.verdict.self_testing_score,
                    ),
                )
            )
        if outcome.materials:
            workspace.apply_patch(
                DeltaPatch(
                    run_id=workspace.case_id,
                    agent_id=f"preparer_{prog_val}",
                    operation=PatchOperationType.REPLACE,
                    path=f"/materials/{prog_val}",
                    value=outcome.materials.model_dump(mode="json"),
                    provenance=PatchProvenance(
                        agent_id=f"preparer_{prog_val}",
                        task_id=task.task_id,
                    ),
                )
            )


class GovernanceJudgePlugin(PipelinePlugin):
    """Pipeline plugin running continuous real-time judge auditing over verifier verdicts."""

    name = "governance_judge"

    def __init__(self, audit_log: AuditLog) -> None:
        self.audit = audit_log
        self.judge = get_default_judge()

    def post_task_execute(
        self,
        task: Task,
        outcome: ProgramOutcome | None,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
    ) -> None:
        if outcome and outcome.assessment and outcome.verdict:
            self.audit.evaluate_and_log_verifier(
                case_id=outcome.assessment.case_id,
                assessment=outcome.assessment,
                verdict=outcome.verdict,
                evidence=trajectory.latest_evidence,
                jurisdiction=outcome.assessment.jurisdiction,
                judge=self.judge,
            )


class ToolCachePlugin(PipelinePlugin):
    """Pipeline plugin managing the ToolResultCache."""

    name = "tool_cache"

    def __init__(self, cache: ToolResultCache) -> None:
        self.cache = cache


# --------------------------------------------------------------------------- #
# CasePipeline Core Loop
# --------------------------------------------------------------------------- #


def _make_recorder(settings: TribuneSettings) -> UsageRecorder:
    pricing_date = date.fromisoformat(settings.pricing_date) if settings.pricing_date else None
    cost_model = CostModel.load(settings.pricing_path or None)
    return UsageRecorder(cost_model=cost_model, pricing_date=pricing_date)


class CasePipeline:
    """Plugin-based orchestrator for public benefits eligibility determinations."""

    def __init__(
        self,
        settings: TribuneSettings | None = None,
        checkpoint_path: str = ".tribune/last_run.json",
        plugins: list[PipelinePlugin] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.checkpoint_path = checkpoint_path
        init_tracing()

        self.rule_store = make_rule_store(self.settings)
        self.recorder = _make_recorder(self.settings)
        proposer_provider = get_provider_for_role("proposer", self.settings, self.recorder)
        verifier_provider = get_provider_for_role("verifier", self.settings, self.recorder)

        self.proposer = EligibilityProposer(proposer_provider, self.rule_store)
        self.verifier = Verifier(verifier_provider, self.rule_store)
        self.calibrator = Calibrator(threshold=self.settings.abstention_threshold)
        self.preparer = Preparer(ActionGate())
        self.router = Router()
        self.navigator = Navigator(self.rule_store)
        self.partitions = PartitionManager()
        self.ingest = make_doc_ingest(self.settings)
        self.audit = AuditLog()
        self.runner = DAGRunner()
        self.async_runner = AsyncDAGRunner()
        self.workspace: WorkspaceContext | None = None

        # Asynchronous tool cache
        self.tool_cache = ToolResultCache(default_ttl=300.0, max_size=1000)

        # State machine
        self.sm = CaseStateMachine(
            proposer=self.proposer,
            verifier=self.verifier,
            calibrator=self.calibrator,
            preparer=self.preparer,
            router=self.router,
            audit=self.audit,
            recorder=self.recorder,
        )

        # Failure telemetry buffer for continual harness evolution
        self.failure_telemetry: list[dict[str, Any]] = []

        # Plugins
        self._plugins: list[PipelinePlugin] = []
        if plugins is not None:
            self._plugins.extend(plugins)
        else:
            # Default plugin chain
            self._plugins.append(AuditLoggingPlugin(self.audit))
            self._plugins.append(WorkspaceSyncPlugin())
            self._plugins.append(ToolCachePlugin(self.tool_cache))
            self._plugins.append(GovernanceJudgePlugin(self.audit))

    def record_failure_trace(self, trace: dict[str, Any]) -> None:
        """Record an execution failure or governance violation trace."""
        self.failure_telemetry.append(trace)

    def get_failure_telemetry(self) -> list[dict[str, Any]]:
        """Retrieve all recorded failure telemetry traces."""
        return list(self.failure_telemetry)

    def clear_failure_telemetry(self) -> None:
        """Clear recorded failure telemetry traces."""
        self.failure_telemetry.clear()

    def register_plugin(self, plugin: PipelinePlugin) -> None:
        """Register a custom pipeline plugin."""
        self._plugins.append(plugin)


    def _save_milestone_checkpoint(
        self,
        case_id: str,
        jurisdiction: str,
        dag: Any,
        completed_task_ids: list[str],
        outcomes: list[ProgramOutcome],
        partition: CasePartition,
    ) -> None:
        """Atomically persist pipeline checkpoint to disk for crash recovery."""
        outcomes_data = [o.model_dump(mode="json") for o in outcomes]
        CheckpointManager.save_checkpoint(
            case_id=case_id,
            jurisdiction=jurisdiction,
            dag_dict=dag.to_dict(),
            completed_task_ids=completed_task_ids,
            outcomes_data=outcomes_data,
            memory_snapshot=partition.snapshot(),
            audit_records=self.audit.records(case_id),
            path=self.checkpoint_path,
        )

    def run_case(self, case: SyntheticCase) -> CaseRunResult:
        with span("run_case", case_id=case.case_id, jurisdiction=case.jurisdiction):
            start_t = time.perf_counter()

            # Initialize Shared Workspace Context for multi-agent execution
            self.workspace = WorkspaceContext(case_id=case.case_id, jurisdiction=case.jurisdiction)

            # Plugin hook: on_pipeline_start
            for plugin in self._plugins:
                plugin.on_pipeline_start(case, self.workspace)

            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            constraints = extract_statutory_constraints(case.evidence)
            static_prefix = (
                f"TRIBUNE System Instructions (Static Prefix v1.0)\n"
                f"Jurisdiction: {case.jurisdiction} | Case: {case.case_id}\n"
                f"{constraints.to_system_header()}"
            )
            trajectory = TrajectoryBuffer(static_prefix=static_prefix)
            trajectory = trajectory.append(
                SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            dag = self.navigator.plan(case)
            result = CaseRunResult(case_id=case.case_id, jurisdiction=case.jurisdiction)
            ocr_lat = 0.0
            completed_task_ids: list[str] = []

            def executor(task: Task):
                nonlocal trajectory, ocr_lat
                task_ctx: dict[str, Any] = {}

                # Plugin hook: pre_task_execute
                for plugin in self._plugins:
                    aug = plugin.pre_task_execute(task, trajectory, self.workspace, task_ctx)
                    if aug:
                        task_ctx.update(aug)

                if task.kind == "gather":
                    self.audit.append(
                        case.case_id,
                        SMState.GATHER,
                        "navigator",
                        f"ingest {len(case.documents)} document(s) via '{self.ingest.name}' adapter",
                    )
                    g_start = time.perf_counter()
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                    ocr_lat = getattr(self.ingest, "last_latency_ms", (time.perf_counter() - g_start) * 1000.0)
                    result.ocr_latency_ms = ocr_lat
                    evidence = consolidator.consolidate_evidence()

                    # Write structured delta patch to workspace context
                    if self.workspace:
                        self.workspace.apply_patch(
                            DeltaPatch(
                                run_id=case.case_id,
                                agent_id="gather",
                                operation=PatchOperationType.REPLACE,
                                path="/evidence",
                                value=[e.model_dump(mode="json") for e in evidence],
                                provenance=PatchProvenance(agent_id="gather"),
                            )
                        )

                    trajectory = trajectory.append(
                        SMState.GATHER,
                        "navigator",
                        f"ingested {len(evidence)} evidence item(s)",
                        {"evidence": evidence},
                    )
                    completed_task_ids.append(task.task_id)
                    self._save_milestone_checkpoint(
                        case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                    )

                    # Plugin hook: post_task_execute
                    for plugin in self._plugins:
                        plugin.post_task_execute(task, None, trajectory, self.workspace)
                    return None

                program = task.program
                assert program is not None

                # Allocate dedicated, isolated SubagentMemoryPartition for worktree isolation
                subagent_id = task.subagent_id or f"subagent_{program.value}"
                evidence_recs = partition.read_all("evidence")
                subagent_partition = self.partitions.open_subagent(
                    case_id=case.case_id,
                    subagent_id=subagent_id,
                    program=program,
                    initial_records=evidence_recs,
                )
                subagent_consolidator = MemoryConsolidator(subagent_partition)

                self.recorder.start_task(program)
                current_evidence = trajectory.latest_evidence or case.evidence
                try:
                    outcome = self.sm.run_program(
                        case.case_id, case.jurisdiction, program, current_evidence, subagent_consolidator
                    )
                    outcome.ocr_latency_ms = ocr_lat
                    result.citation_latency_ms = max(result.citation_latency_ms, outcome.citation_latency_ms)
                    result.llm_latency_ms += outcome.llm_latency_ms

                    # Capture verifier failure telemetry if uncertified or has discrepancies
                    if outcome.verdict and outcome.assessment:
                        fail_payload = self.verifier.extract_failure_payload(outcome.assessment, outcome.verdict)
                        if fail_payload:
                            fail_payload["task_id"] = task.task_id
                            self.record_failure_trace(fail_payload)
                            result.failure_traces.append(fail_payload)

                    trajectory = trajectory.append(
                        outcome.final_state,
                        "state_machine",
                        f"program {program.value} completed with state {outcome.final_state.value}",
                        {"program": program.value, "abstained": outcome.abstained},
                    )
                    self.partitions.merge_subagent(subagent_partition, partition)
                except Exception as exc:
                    fail_payload = {
                        "case_id": case.case_id,
                        "program": program.value,
                        "agent_id": "state_machine",
                        "category": "governance_violation" if "ActionBlocked" in type(exc).__name__ or "Security" in type(exc).__name__ else "general_failure",
                        "error_message": str(exc),
                        "task_id": task.task_id,
                    }
                    self.record_failure_trace(fail_payload)
                    result.failure_traces.append(fail_payload)

                    self.audit.append(
                        case.case_id,
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        payload={"error": str(exc)[:200]},
                    )
                    trajectory = trajectory.append(
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        {"error": str(exc)[:200]},
                    )
                    outcome = ProgramOutcome(
                        program=program, abstained=True, final_state=SMState.ABSTAIN
                    )

                # Capture any ActionGate failure payloads
                if hasattr(self.preparer, "action_gate"):
                    for gate_fail in self.preparer.action_gate.get_failure_payloads():
                        if gate_fail not in result.failure_traces:
                            self.record_failure_trace(gate_fail)
                            result.failure_traces.append(gate_fail)

                outcome.usage = self.recorder.finish_task()
                result.outcomes.append(outcome)
                completed_task_ids.append(task.task_id)

                self._save_milestone_checkpoint(
                    case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )

                # Plugin hook: post_task_execute
                for plugin in self._plugins:
                    plugin.post_task_execute(task, outcome, trajectory, self.workspace)

                return outcome

            self.runner.run(dag, executor)
            result.total_latency_ms = (time.perf_counter() - start_t) * 1000.0
            result.audit = self.audit.records(case.case_id)


            if self.workspace:
                result.workspace_version = self.workspace.version
                agent_count = max(4, len(case.target_programs) * 2 + 2)
                result.token_reduction = self.workspace.calculate_token_reduction(agent_count=agent_count)

            # Plugin hook: on_pipeline_end
            for plugin in self._plugins:
                plugin.on_pipeline_end(case, result, self.workspace)

            CheckpointManager.clear_checkpoint(self.checkpoint_path)
            return result

    async def run_case_async(self, case: SyntheticCase) -> CaseRunResult:
        """Asynchronous execution of case DAG with parallel subagent wave fan-out and plugin hooks."""
        with span("run_case_async", case_id=case.case_id, jurisdiction=case.jurisdiction):
            start_t = time.perf_counter()

            self.workspace = WorkspaceContext(case_id=case.case_id, jurisdiction=case.jurisdiction)

            # Plugin hook: on_pipeline_start
            for plugin in self._plugins:
                plugin.on_pipeline_start(case, self.workspace)

            self.recorder.start_case(case.case_id, case.language)
            partition = self.partitions.open(case.case_id)
            consolidator = MemoryConsolidator(partition)

            constraints = extract_statutory_constraints(case.evidence)
            static_prefix = (
                f"TRIBUNE System Instructions (Static Prefix v1.0)\n"
                f"Jurisdiction: {case.jurisdiction} | Case: {case.case_id}\n"
                f"{constraints.to_system_header()}"
            )
            trajectory = TrajectoryBuffer(static_prefix=static_prefix)
            trajectory = trajectory.append(
                SMState.PLAN, "navigator", "decompose situation into a sub-task DAG"
            )
            dag = self.navigator.plan(case)
            result = CaseRunResult(case_id=case.case_id, jurisdiction=case.jurisdiction)
            ocr_lat = 0.0
            completed_task_ids: list[str] = []

            async def async_executor(task: Task):
                nonlocal trajectory, ocr_lat
                task_ctx: dict[str, Any] = {}

                # Plugin hook: pre_task_execute
                for plugin in self._plugins:
                    aug = plugin.pre_task_execute(task, trajectory, self.workspace, task_ctx)
                    if aug:
                        task_ctx.update(aug)

                if task.kind == "gather":
                    self.audit.append(
                        case.case_id,
                        SMState.GATHER,
                        "navigator",
                        f"ingest {len(case.documents)} document(s) via '{self.ingest.name}' adapter",
                    )
                    g_start = time.perf_counter()
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                    ocr_lat = getattr(self.ingest, "last_latency_ms", (time.perf_counter() - g_start) * 1000.0)
                    result.ocr_latency_ms = ocr_lat
                    evidence = consolidator.consolidate_evidence()

                    if self.workspace:
                        self.workspace.apply_patch(
                            DeltaPatch(
                                run_id=case.case_id,
                                agent_id="gather",
                                operation=PatchOperationType.REPLACE,
                                path="/evidence",
                                value=[e.model_dump(mode="json") for e in evidence],
                                provenance=PatchProvenance(agent_id="gather"),
                            )
                        )

                    trajectory = trajectory.append(
                        SMState.GATHER,
                        "navigator",
                        f"ingested {len(evidence)} evidence item(s)",
                        {"evidence": evidence},
                    )
                    completed_task_ids.append(task.task_id)
                    self._save_milestone_checkpoint(
                        case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                    )

                    for plugin in self._plugins:
                        plugin.post_task_execute(task, None, trajectory, self.workspace)
                    return None

                program = task.program
                assert program is not None

                subagent_id = task.subagent_id or f"subagent_{program.value}"
                evidence_recs = partition.read_all("evidence")
                subagent_partition = self.partitions.open_subagent(
                    case_id=case.case_id,
                    subagent_id=subagent_id,
                    program=program,
                    initial_records=evidence_recs,
                )
                subagent_consolidator = MemoryConsolidator(subagent_partition)

                self.recorder.start_task(program)
                current_evidence = trajectory.latest_evidence or case.evidence
                try:
                    outcome = await asyncio.to_thread(
                        self.sm.run_program,
                        case.case_id,
                        case.jurisdiction,
                        program,
                        current_evidence,
                        subagent_consolidator,
                    )
                    outcome.ocr_latency_ms = ocr_lat
                    result.citation_latency_ms = max(result.citation_latency_ms, outcome.citation_latency_ms)
                    result.llm_latency_ms += outcome.llm_latency_ms

                    # Capture verifier failure telemetry if uncertified or has discrepancies
                    if outcome.verdict and outcome.assessment:
                        fail_payload = self.verifier.extract_failure_payload(outcome.assessment, outcome.verdict)
                        if fail_payload:
                            fail_payload["task_id"] = task.task_id
                            self.record_failure_trace(fail_payload)
                            result.failure_traces.append(fail_payload)

                    trajectory = trajectory.append(
                        outcome.final_state,
                        "state_machine",
                        f"program {program.value} completed with state {outcome.final_state.value}",
                        {"program": program.value, "abstained": outcome.abstained},
                    )
                    self.partitions.merge_subagent(subagent_partition, partition)
                except Exception as exc:
                    fail_payload = {
                        "case_id": case.case_id,
                        "program": program.value,
                        "agent_id": "state_machine",
                        "category": "governance_violation" if "ActionBlocked" in type(exc).__name__ or "Security" in type(exc).__name__ else "general_failure",
                        "error_message": str(exc),
                        "task_id": task.task_id,
                    }
                    self.record_failure_trace(fail_payload)
                    result.failure_traces.append(fail_payload)

                    self.audit.append(
                        case.case_id,
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        payload={"error": str(exc)[:200]},
                    )
                    trajectory = trajectory.append(
                        SMState.ABSTAIN,
                        "navigator",
                        "fail-safe abstain due to internal error",
                        {"error": str(exc)[:200]},
                    )
                    outcome = ProgramOutcome(
                        program=program, abstained=True, final_state=SMState.ABSTAIN
                    )

                # Capture any ActionGate failure payloads
                if hasattr(self.preparer, "action_gate"):
                    for gate_fail in self.preparer.action_gate.get_failure_payloads():
                        if gate_fail not in result.failure_traces:
                            self.record_failure_trace(gate_fail)
                            result.failure_traces.append(gate_fail)


                outcome.usage = self.recorder.finish_task()
                result.outcomes.append(outcome)
                completed_task_ids.append(task.task_id)

                self._save_milestone_checkpoint(
                    case.case_id, case.jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )

                for plugin in self._plugins:
                    plugin.post_task_execute(task, outcome, trajectory, self.workspace)

                return outcome

            await self.async_runner.run_async(dag, async_executor)
            result.total_latency_ms = (time.perf_counter() - start_t) * 1000.0
            result.audit = self.audit.records(case.case_id)

            if self.workspace:
                result.workspace_version = self.workspace.version
                agent_count = max(4, len(case.target_programs) * 2 + 2)
                result.token_reduction = self.workspace.calculate_token_reduction(agent_count=agent_count)

            for plugin in self._plugins:
                plugin.on_pipeline_end(case, result, self.workspace)

            CheckpointManager.clear_checkpoint(self.checkpoint_path)
            return result

    def resume_from_checkpoint(
        self,
        checkpoint_path: str | None = None,
        case: SyntheticCase | None = None,
    ) -> CaseRunResult:
        """Resume pipeline execution from checkpoint after an unexpected interruption or crash."""
        path = checkpoint_path or self.checkpoint_path
        recovery_plan: RecoveryPlan | None = self.router.inspect_checkpoint(path)
        if not recovery_plan:
            if case is not None:
                return self.run_case(case)
            raise FileNotFoundError(f"No valid incomplete checkpoint found at '{path}' to resume from.")

        case_id = recovery_plan.case_id
        jurisdiction = recovery_plan.jurisdiction
        dag = recovery_plan.reconstructed_dag

        self.audit.load_records(case_id, recovery_plan.restored_audit)

        partition = self.partitions.open(case_id)
        partition.restore(recovery_plan.memory_snapshot)
        consolidator = MemoryConsolidator(partition)

        result = CaseRunResult(case_id=case_id, jurisdiction=jurisdiction)
        result.outcomes.extend(recovery_plan.restored_outcomes)

        completed_task_ids = list(recovery_plan.completed_task_ids)

        def resume_executor(task: Task):
            if task.task_id in completed_task_ids:
                return None

            if task.kind == "gather":
                if case is not None:
                    consolidator.store_evidence(self.ingest.ingest_many(case.documents))
                completed_task_ids.append(task.task_id)
                self._save_milestone_checkpoint(
                    case_id, jurisdiction, dag, completed_task_ids, result.outcomes, partition
                )
                return None

            program = task.program
            assert program is not None

            subagent_id = task.subagent_id or f"subagent_{program.value}"
            evidence_recs = partition.read_all("evidence")
            subagent_partition = self.partitions.open_subagent(
                case_id=case_id,
                subagent_id=subagent_id,
                program=program,
                initial_records=evidence_recs,
            )
            subagent_consolidator = MemoryConsolidator(subagent_partition)

            self.recorder.start_task(program)
            evidence = consolidator.consolidate_evidence()
            try:
                outcome = self.sm.run_program(
                    case_id, jurisdiction, program, evidence, subagent_consolidator
                )
                self.partitions.merge_subagent(subagent_partition, partition)
            except Exception as exc:
                self.audit.append(
                    case_id,
                    SMState.ABSTAIN,
                    "navigator",
                    "fail-safe abstain on recovery due to internal error",
                    payload={"error": str(exc)[:200]},
                )
                outcome = ProgramOutcome(
                    program=program, abstained=True, final_state=SMState.ABSTAIN
                )

            outcome.usage = self.recorder.finish_task()
            result.outcomes.append(outcome)
            completed_task_ids.append(task.task_id)

            self._save_milestone_checkpoint(
                case_id, jurisdiction, dag, completed_task_ids, result.outcomes, partition
            )
            return outcome

        self.runner.run(dag, resume_executor)
        result.audit = self.audit.records(case_id)
        CheckpointManager.clear_checkpoint(path)
        return result

    def run_cases(self, cases: list[SyntheticCase]) -> list[CaseRunResult]:
        return [self.run_case(c) for c in cases]


__all__ = [
    "TrajectoryFrame",
    "TrajectoryBuffer",
    "ToolResultCache",
    "PipelinePlugin",
    "AuditLoggingPlugin",
    "WorkspaceSyncPlugin",
    "GovernanceJudgePlugin",
    "ToolCachePlugin",
    "CasePipeline",
]
