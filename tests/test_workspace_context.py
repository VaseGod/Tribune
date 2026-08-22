"""Unit tests for Central Shared Workspace Context and JSON Delta Patches."""

from __future__ import annotations

import tempfile
import unittest

from tribune.context.workspace import (
    PatchValidationError,
    VersionConflictError,
    WorkspaceContext,
    WorkspaceState,
)
from tribune.types import DeltaPatch, PatchOperationType, PatchProvenance


class TestWorkspaceContext(unittest.TestCase):
    def test_state_basic_and_pointer_navigation(self) -> None:
        state = WorkspaceState(case_id="case_123", jurisdiction="EX")
        self.assertEqual(state.get_value_at_path("/case_id"), "case_123")
        self.assertEqual(state.get_value_at_path("/jurisdiction"), "EX")
        self.assertEqual(state.get_value_at_path("/documents"), [])

    def test_delta_patch_add_and_replace(self) -> None:
        ctx = WorkspaceContext(case_id="c1", jurisdiction="EX")

        # 1. Add evidence item
        patch1 = DeltaPatch(
            run_id="c1",
            agent_id="gather",
            operation=PatchOperationType.ADD,
            path="/evidence/-",
            value={"evidence_id": "ev_1", "type": "monthly_income", "value": 1500.0},
            provenance=PatchProvenance(agent_id="gather"),
        )
        snap1 = ctx.apply_patch(patch1)
        self.assertEqual(ctx.version, 1)
        self.assertEqual(len(snap1.state_data["evidence"]), 1)
        self.assertEqual(snap1.state_data["evidence"][0]["evidence_id"], "ev_1")

        # 2. Replace metadata
        patch2 = DeltaPatch(
            run_id="c1",
            agent_id="proposer_snap",
            operation=PatchOperationType.REPLACE,
            path="/criteria_outcomes/snap",
            value={"gross_income": "satisfied"},
            provenance=PatchProvenance(agent_id="proposer_snap"),
        )
        snap2 = ctx.apply_patch(patch2)
        self.assertEqual(ctx.version, 2)
        self.assertEqual(snap2.state_data["criteria_outcomes"]["snap"]["gross_income"], "satisfied")

    def test_optimistic_concurrency_conflict_detection(self) -> None:
        ctx = WorkspaceContext(case_id="c1", jurisdiction="EX")

        # Apply patch at version 0 expecting version 0
        patch1 = DeltaPatch(
            run_id="c1",
            agent_id="agent_a",
            operation=PatchOperationType.REPLACE,
            path="/shared_facts/flag",
            value=True,
            expected_version=0,
            provenance=PatchProvenance(agent_id="agent_a"),
        )
        ctx.apply_patch(patch1)
        self.assertEqual(ctx.version, 1)

        # Apply conflicting patch expecting version 0 when state is at version 1
        patch2 = DeltaPatch(
            run_id="c1",
            agent_id="agent_b",
            operation=PatchOperationType.REPLACE,
            path="/shared_facts/flag",
            value=False,
            expected_version=0,  # Stale version
            conflict_strategy="error",
            provenance=PatchProvenance(agent_id="agent_b"),
        )
        with self.assertRaises(VersionConflictError):
            ctx.apply_patch(patch2)

    def test_scoped_state_slicing(self) -> None:
        ctx = WorkspaceContext(case_id="c1", jurisdiction="EX")
        ctx.apply_patch(
            DeltaPatch(
                run_id="c1",
                agent_id="gather",
                operation=PatchOperationType.REPLACE,
                path="/evidence",
                value=[{"evidence_id": "e1"}],
                provenance=PatchProvenance(agent_id="gather"),
            )
        )
        ctx.apply_patch(
            DeltaPatch(
                run_id="c1",
                agent_id="proposer_snap",
                operation=PatchOperationType.REPLACE,
                path="/assessments/snap",
                value={"status": "likely_eligible"},
                provenance=PatchProvenance(agent_id="proposer_snap"),
            )
        )

        # Request slice with only /assessments/snap
        sliced = ctx.get_slice(["/assessments/snap"])
        self.assertIn("assessments", sliced)
        self.assertEqual(sliced["assessments"]["snap"]["status"], "likely_eligible")
        # Ensure /evidence was excluded from slice
        self.assertNotIn("evidence", sliced)

    def test_broadcast_subscriptions(self) -> None:
        ctx = WorkspaceContext(case_id="c1", jurisdiction="EX")
        received_patches: list[DeltaPatch] = []

        ctx.subscribe("/assessments", lambda p: received_patches.append(p))

        # Patch matching subscribed path
        patch_match = DeltaPatch(
            run_id="c1",
            agent_id="proposer_snap",
            operation=PatchOperationType.REPLACE,
            path="/assessments/snap",
            value={"status": "likely_eligible"},
            provenance=PatchProvenance(agent_id="proposer_snap"),
        )
        ctx.apply_patch(patch_match)

        # Patch not matching subscribed path
        patch_nomatch = DeltaPatch(
            run_id="c1",
            agent_id="gather",
            operation=PatchOperationType.REPLACE,
            path="/evidence",
            value=[],
            provenance=PatchProvenance(agent_id="gather"),
        )
        ctx.apply_patch(patch_nomatch)

        self.assertEqual(len(received_patches), 1)
        self.assertEqual(received_patches[0].path, "/assessments/snap")

    def test_deterministic_replay(self) -> None:
        ctx1 = WorkspaceContext(case_id="c1", jurisdiction="EX")
        patches = [
            DeltaPatch(
                run_id="c1",
                agent_id="agent1",
                operation=PatchOperationType.REPLACE,
                path="/shared_facts/a",
                value=10,
                provenance=PatchProvenance(agent_id="agent1"),
            ),
            DeltaPatch(
                run_id="c1",
                agent_id="agent2",
                operation=PatchOperationType.REPLACE,
                path="/shared_facts/b",
                value=20,
                provenance=PatchProvenance(agent_id="agent2"),
            ),
            DeltaPatch(
                run_id="c1",
                agent_id="agent1",
                operation=PatchOperationType.REPLACE,
                path="/shared_facts/c",
                value=30,
                provenance=PatchProvenance(agent_id="agent1"),
            ),
        ]
        ctx1.apply_patches(patches)

        # Replay from patch list
        ctx2 = WorkspaceContext.replay(case_id="c1", jurisdiction="EX", patches=ctx1.patch_history)
        self.assertEqual(ctx1.snapshot().state_data, ctx2.snapshot().state_data)
        self.assertEqual(ctx2.version, 3)

    def test_token_reduction_metric_calculation(self) -> None:
        ctx = WorkspaceContext(case_id="c1", jurisdiction="EX")
        ctx.apply_patch(
            DeltaPatch(
                run_id="c1",
                agent_id="gather",
                operation=PatchOperationType.REPLACE,
                path="/evidence",
                value=[{"type": f"fact_{i}", "value": i} for i in range(20)],
                provenance=PatchProvenance(agent_id="gather"),
            )
        )
        metric = ctx.calculate_token_reduction(agent_count=8, turns_per_agent=2)
        self.assertGreater(metric.baseline_tokens, metric.workspace_tokens)
        self.assertGreaterEqual(metric.reduction_percentage, 40.0)
        self.assertEqual(metric.agent_count, 8)


if __name__ == "__main__":
    unittest.main()
