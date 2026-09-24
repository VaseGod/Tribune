"""Unit Tests for Hardened Evaluation Reporting and Comparative Metrics.

Validates:
- Comparative metrics aggregation across rungs
- Markdown summary generation with Dual-Agent Trajectory Efficiency section
- JSON metrics serialization and schema
- CSV table export
- File writing helpers (write_hardened_reports)
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from tribune.eval.quant_sensitivity.backends import smoke_ladder
from tribune.eval.quant_sensitivity.ladder import run_ladder
from tribune.eval.quant_sensitivity.report import (
    compute_hardened_comparative_report,
    render_eval_note,
    render_hardened_csv_table,
    render_hardened_json_metrics,
    write_hardened_reports,
)
from tribune.eval.quant_sensitivity.seedset import build_seed_set


@pytest.fixture(scope="module")
def sample_ladder_result():
    """Run a tiny 5-case smoke ladder covering all 5 programs for deterministic reporting tests."""
    from tribune.types import ProgramId

    rungs = smoke_ladder()
    cases = build_seed_set(
        limit_per_program={
            ProgramId.SNAP: 1,
            ProgramId.MEDICAID: 1,
            ProgramId.HOUSING: 1,
            ProgramId.UNEMPLOYMENT: 1,
            ProgramId.APPEALS: 1,
        }
    )
    return run_ladder(rungs=rungs, cases=cases)


def test_markdown_report_includes_hardened_sections(sample_ladder_result):
    """Verify rendered markdown contains trajectory efficiency and operational insights."""
    note = render_eval_note(sample_ladder_result)
    assert "# TRIBUNE Eval Note #1" in note
    assert "## Hardened Dual-Agent Trajectory Efficiency & Economics" in note
    assert "| rung | interventions | loops avoided | aux calls | net efficiency |" in note
    assert "Turn Reduction:" in note
    assert "Low-Bit Recovery:" in note
    assert "Economic Offset:" in note


def test_hardened_json_metrics_structure(sample_ladder_result):
    """Verify JSON metrics export produces valid JSON with required keys."""
    raw_json = render_hardened_json_metrics(sample_ladder_result)
    data = json.loads(raw_json)

    assert "reference_label" in data
    assert "weights" in data
    assert "rungs" in data
    assert len(data["rungs"]) == len(sample_ladder_result.rungs)

    rung0 = data["rungs"][0]
    assert "label" in rung0
    assert "total_turns" in rung0
    assert "net_efficiency_score" in rung0
    assert "interventions_count" in rung0
    assert "avoided_loops_count" in rung0


def test_hardened_csv_table_output(sample_ladder_result):
    """Verify CSV export contains header and data rows."""
    csv_text = render_hardened_csv_table(sample_ladder_result)
    lines = csv_text.strip().splitlines()
    assert len(lines) >= 3  # Header + at least 2 rungs
    assert lines[0].startswith("rung,quant_format,total_turns")


def test_write_hardened_reports(sample_ladder_result):
    """Verify write_hardened_reports persists markdown, JSON, and CSV artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = write_hardened_reports(sample_ladder_result, output_dir=tmpdir)
        assert os.path.exists(paths["markdown"])
        assert os.path.exists(paths["json"])
        assert os.path.exists(paths["csv"])

        with open(paths["json"], encoding="utf-8") as fh:
            data = json.load(fh)
            assert "rungs" in data
