"""Unit & integration tests for Phase 4: Multimodal Witness Deposition Simulation Engine & World Model."""

from __future__ import annotations

import pytest

from tribune.casegen.simulation import (
    FacialExpression,
    FalMiniMaxStreamingClient,
    QuestionPressure,
    WitnessDepositionSimulator,
)
from tribune.casegen.world_model import (
    CourtroomWorldModel,
    ExhibitStatus,
    ObjectionType,
)


def test_witness_deposition_stress_and_facial_expression_dynamics():
    """Verify dynamic adjustment of emotional stress, facial expressions, and micro-hesitations."""
    simulator = WitnessDepositionSimulator()
    witness = simulator.register_witness(
        witness_id="wit_01",
        name="Robert Hayes",
        role="claimant",
        baseline_stress=0.20,
    )
    assert witness.witness_id == "wit_01"

    # 1. Neutral questioning turn -> stress remains low, expression neutral
    turn_neutral = simulator.interrogate(
        witness_id="wit_01",
        question="Please state your full name and current residential address for the record.",
        pressure=QuestionPressure.NEUTRAL,
    )
    assert turn_neutral.witness_state.emotional_stress == pytest.approx(0.20, abs=0.05)
    assert turn_neutral.witness_state.facial_expression == FacialExpression.NEUTRAL
    assert turn_neutral.witness_hesitation_observed_ms < 400

    # 2. Accusatory questioning with contradicting evidentiary document
    turn_pressure = simulator.interrogate(
        witness_id="wit_01",
        question="Isn't it true that your pay stubs from Apex Logistics show overtime earnings during June?",
        pressure=QuestionPressure.ACCUSATORY,
        contradicting_evidence=["Wage records dated June 15 (Exhibit B)"],
    )
    # Stress spikes due to accusatory pressure (+0.20) and contradiction (+0.25)
    assert turn_pressure.witness_state.emotional_stress >= 0.60
    assert turn_pressure.witness_state.facial_expression in (
        FacialExpression.FURROWED_BROW,
        FacialExpression.TENSE,
        FacialExpression.NERVOUS_SMILE,
    )
    assert turn_pressure.witness_hesitation_observed_ms > turn_neutral.witness_hesitation_observed_ms
    assert turn_pressure.witness_state.contradictions_detected >= 1

    # 3. Highly confrontational questioning turn -> expression becomes DEFIANT or LOOKING_AWAY
    turn_confrontational = simulator.interrogate(
        witness_id="wit_01",
        question="You intentionally concealed this secondary income source from the agency, didn't you?",
        pressure=QuestionPressure.CONFRONTATIONAL,
        contradicting_evidence=["Bank deposit slips", "Employer 1099 filing"],
    )
    assert turn_confrontational.witness_state.emotional_stress >= 0.75
    assert turn_confrontational.witness_state.facial_expression in (
        FacialExpression.DEFIANT,
        FacialExpression.LOOKING_AWAY,
        FacialExpression.TENSE,
    )


def test_fal_minimax_streaming_client_and_subsecond_rtf():
    """Verify Fal MiniMax live streaming client and offline mock driver deliver RTF < 1.0."""
    client = FalMiniMaxStreamingClient(offline_mode=True)
    simulator = WitnessDepositionSimulator(client=client)
    witness = simulator.register_witness("wit_02", name="Elena Vance")
    assert witness.name == "Elena Vance"

    turn = simulator.interrogate(
        witness_id="wit_02",
        question="Did you receive the determination notice on September 1st?",
        pressure=QuestionPressure.NEUTRAL,
        duration_s=1.0,
    )

    # Validate streaming video chunks
    chunks = turn.video_chunks
    assert len(chunks) >= 4  # 1.0s / 250ms chunks = 4 chunks
    for chunk in chunks:
        assert len(chunk.frame_bytes) == 1024
        assert len(chunk.audio_pcm_bytes) > 0
        assert chunk.witness_state.witness_id == "wit_02"
        # Real-Time Factor MUST be < 1.0 for real-time sub-second streaming
        assert chunk.rtf < 1.0

    assert turn.average_rtf < 1.0
    assert turn.max_rtf < 1.0


def test_courtroom_world_model_exhibit_admission_workflow():
    """Verify document marking, formal offering, objections, and judicial admission."""
    world_model = CourtroomWorldModel()

    # 1. Mark Exhibit
    exhibit = world_model.mark_exhibit(
        document_id="doc_payroll_01",
        label="Exhibit A",
        description="Certified quarterly payroll records from Apex Logistics",
        stamp_pos=(0.82, 0.08),
    )
    assert exhibit.status == ExhibitStatus.MARKED
    assert exhibit.label == "Exhibit A"
    assert exhibit.stamp_coordinates == (0.82, 0.08)

    # 2. Offer for Admission
    offered = world_model.offer_exhibit(exhibit.exhibit_id)
    assert offered.status == ExhibitStatus.OFFERED

    # 3. Raise Objection (e.g. Hearsay)
    objected = world_model.raise_objection(
        exhibit_id=exhibit.exhibit_id,
        objection_type=ObjectionType.HEARSAY,
        rationale="Records lack custodian certificate of authenticity under Rule 803(6).",
    )
    assert objected.status == ExhibitStatus.OBJECTED
    assert len(objected.objection_history) == 1
    assert objected.objection_history[0]["objection_type"] == "hearsay"

    # 4. Judicial Ruling: Overrule and Admit under business records exception
    admitted = world_model.admit_exhibit(
        exhibit_id=exhibit.exhibit_id,
        sustain_objection=False,
        judicial_rationale="Overruled. Covered under statutory business records exception.",
    )
    assert admitted.status == ExhibitStatus.ADMITTED
    assert "Overruled" in admitted.ruling


def test_courtroom_world_model_ui_screen_rendering_and_error_handling():
    """Verify visual screen rendering of case-management software and UI error banner recovery."""
    world_model = CourtroomWorldModel()
    ex = world_model.mark_exhibit("doc_lease_1", "Exhibit 1", "Rental Agreement Lease")
    assert ex.label == "Exhibit 1"

    # 1. Render normal display state
    state = world_model.render_screen_mock()
    assert state.screen_resolution == (1920, 1080)
    assert state.active_window == "CourtroomEvidenceViewer"
    assert state.active_exhibit is not None
    assert state.active_exhibit.label == "Exhibit 1"
    assert state.error_banner is None

    # Check UI elements present
    elem_ids = [el.element_id for el in state.elements]
    assert "btn_mark_exhibit" in elem_ids
    assert "btn_offer_admission" in elem_ids
    assert "viewport_evidence_display" in elem_ids

    # 2. Simulate UI error
    err_state = world_model.simulate_ui_error(
        error_code="E_VIEWPORT_TIMEOUT",
        error_message="Evidence rendering engine timed out loading high-res scan.",
    )
    assert err_state.error_banner is not None
    assert "E_VIEWPORT_TIMEOUT" in err_state.error_banner
    assert err_state.modal_dialog is not None
    assert err_state.modal_dialog["title"] == "System Warning"

    # 3. Multimodal agent handles error
    handled = world_model.handle_ui_error()
    assert handled is True

    recovered_state = world_model.render_screen_mock()
    assert recovered_state.error_banner is None
    assert recovered_state.modal_dialog is None
