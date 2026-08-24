"""Unit & integration tests for Plugin-Based Pipeline Harness, ToolResultCache, and DiskBackedSessionManager."""

import asyncio
import time

import pytest

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.context.workspace import (
    DiskBackedSessionManager,
    WorkspaceContext,
)
from tribune.orchestration.dag import Task
from tribune.orchestration.pipeline import (
    CasePipeline,
    PipelinePlugin,
    ToolResultCache,
    TrajectoryBuffer,
)
from tribune.types import (
    CaseRunResult,
    DeltaPatch,
    PatchOperationType,
    PatchProvenance,
    ProgramId,
    ProgramOutcome,
    SyntheticCase,
)

# --------------------------------------------------------------------------- #
# 1. Pipeline Plugin Harness Tests
# --------------------------------------------------------------------------- #


class MockRecordingPlugin(PipelinePlugin):
    name = "mock_recording_plugin"

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_pipeline_start(self, case: SyntheticCase, workspace: WorkspaceContext | None) -> None:
        self.events.append(f"start:{case.case_id}")

    def pre_task_execute(
        self,
        task: Task,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
        context: dict,
    ) -> dict | None:
        self.events.append(f"pre_task:{task.task_id}")
        return {"injected_flag": True}

    def post_task_execute(
        self,
        task: Task,
        outcome: ProgramOutcome | None,
        trajectory: TrajectoryBuffer,
        workspace: WorkspaceContext | None,
    ) -> None:
        self.events.append(f"post_task:{task.task_id}")

    def on_pipeline_end(
        self,
        case: SyntheticCase,
        result: CaseRunResult,
        workspace: WorkspaceContext | None,
    ) -> None:
        self.events.append(f"end:{case.case_id}")


def test_custom_plugin_registration_and_execution_hooks():
    plugin = MockRecordingPlugin()
    pipe = CasePipeline(plugins=[plugin])

    gen = SyntheticCaseGenerator(seed=42)
    case = gen.build_case(
        "case_plugin_test",
        "EX",
        {"monthly_income": 1000.0, "household_size": 2},
        [ProgramId.SNAP],
    )

    result = pipe.run_case(case)
    assert result.case_id == "case_plugin_test"
    assert "start:case_plugin_test" in plugin.events
    assert "pre_task:gather" in plugin.events
    assert "post_task:gather" in plugin.events
    assert "pre_task:assess:snap" in plugin.events
    assert "post_task:assess:snap" in plugin.events
    assert "end:case_plugin_test" in plugin.events


# --------------------------------------------------------------------------- #
# 2. ToolResultCache Tests (LRU + TTL + SHA-256)
# --------------------------------------------------------------------------- #


def test_tool_result_cache_lru_and_ttl():
    cache = ToolResultCache(default_ttl=0.2, max_size=3)

    # 1. Deterministic SHA-256 keying
    k1 = cache.compute_cache_key("eval_rule", {"limit": 100, "income": 50}, "EX")
    k2 = cache.compute_cache_key("eval_rule", {"income": 50, "limit": 100}, "EX")
    assert k1 == k2  # Normalized arguments yield identical hash

    # 2. Set and Get
    cache.set("eval_rule", {"income": 50}, {"result": "satisfied"}, "EX")
    val = cache.get("eval_rule", {"income": 50}, "EX")
    assert val == {"result": "satisfied"}
    assert cache.hits == 1

    # 3. Expiration by TTL
    time.sleep(0.25)
    expired = cache.get("eval_rule", {"income": 50}, "EX")
    assert expired is None
    assert cache.evictions >= 1

    # 4. LRU Eviction on capacity
    cache.set("t1", {"x": 1}, 10, "EX", ttl=10.0)
    cache.set("t2", {"x": 2}, 20, "EX", ttl=10.0)
    cache.set("t3", {"x": 3}, 30, "EX", ttl=10.0)
    assert cache.stats()["size"] == 3

    # Add 4th item -> should evict t1 (least recently used)
    cache.get("t2", {"x": 2}, "EX")  # Touch t2
    cache.set("t4", {"x": 4}, 40, "EX", ttl=10.0)
    assert cache.get("t1", {"x": 1}, "EX") is None
    assert cache.get("t2", {"x": 2}, "EX") == 20
    assert cache.get("t4", {"x": 4}, "EX") == 40


@pytest.mark.asyncio
async def test_tool_result_cache_async_get_or_compute():
    cache = ToolResultCache(default_ttl=5.0, max_size=100)
    compute_count = 0

    async def expensive_computation():
        nonlocal compute_count
        compute_count += 1
        await asyncio.sleep(0.01)
        return {"computed_status": "likely_eligible", "score": 0.95}

    # First call: computes and stores
    res1 = await cache.get_or_compute_async("statute_lookup", {"sec": "7CFR273"}, "EX", expensive_computation)
    assert res1["computed_status"] == "likely_eligible"
    assert compute_count == 1

    # Second call: hits cache, compute_count remains 1
    res2 = await cache.get_or_compute_async("statute_lookup", {"sec": "7CFR273"}, "EX", expensive_computation)
    assert res2["computed_status"] == "likely_eligible"
    assert compute_count == 1
    assert cache.hits >= 1


# --------------------------------------------------------------------------- #
# 3. DiskBackedSessionManager Tests
# --------------------------------------------------------------------------- #


def test_disk_backed_session_manager_spill_and_rehydrate(tmp_path):
    storage_dir = str(tmp_path / "sessions")
    manager = DiskBackedSessionManager(storage_dir=storage_dir, idle_timeout_seconds=0.1, max_active_sessions=2)

    # 1. Open session and apply delta patches
    session1 = manager.open_session("case_session_1", "EX")
    session1.apply_patch(
        DeltaPatch(
            run_id="case_session_1",
            agent_id="gather",
            operation=PatchOperationType.REPLACE,
            path="/evidence",
            value=[{"evidence_id": "ev_1", "type": "monthly_income", "value": 1500.0}],
            provenance=PatchProvenance(agent_id="gather"),
        )
    )

    manager.open_session("case_session_2", "EX")
    assert manager.stats()["active_sessions_count"] == 2

    # 2. Add 3rd session -> triggers capacity spill of oldest session1
    manager.open_session("case_session_3", "EX")
    assert manager.stats()["spilled_sessions_count"] >= 1

    # 3. Transparent rehydration on access
    rehydrated = manager.get_session("case_session_1")
    assert rehydrated is not None
    assert rehydrated.case_id == "case_session_1"
    evidence = rehydrated.read_path("/evidence")
    assert len(evidence) == 1
    assert evidence[0]["evidence_id"] == "ev_1"


@pytest.mark.asyncio
async def test_disk_backed_session_manager_async_get(tmp_path):
    storage_dir = str(tmp_path / "sessions_async")
    manager = DiskBackedSessionManager(storage_dir=storage_dir, idle_timeout_seconds=0.05, max_active_sessions=5)

    sess = manager.open_session("case_async_1", "EX")
    sess.apply_patch(
        DeltaPatch(
            run_id="case_async_1",
            agent_id="test",
            operation=PatchOperationType.REPLACE,
            path="/shared_facts/verified",
            value=True,
            provenance=PatchProvenance(agent_id="test"),
        )
    )

    # Manually spill
    disk_path = manager.spill_session("case_async_1")
    assert disk_path is not None
    assert disk_path.endswith(".json.gz")

    # Async rehydration
    rehydrated = await manager.get_session_async("case_async_1")
    assert rehydrated is not None
    assert rehydrated.read_path("/shared_facts/verified") is True
