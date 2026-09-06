"""Courtroom & Case-Management World Model.

Provides visual mock representations of case-management software and courtroom exhibit screens.
Enables multimodal agents to practice visual document admission, evidence marking,
objection handling, and UI error handling.
"""

from __future__ import annotations

import copy
import enum
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any


class ExhibitStatus(str, enum.Enum):
    UNMARKED = "unmarked"
    MARKED = "marked"
    OFFERED = "offered"
    OBJECTED = "objected"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class ObjectionType(str, enum.Enum):
    HEARSAY = "hearsay"
    RELEVANCE = "relevance"
    LACK_OF_FOUNDATION = "lack_of_foundation"
    SPECULATION = "speculation"
    AUTHENTICATION = "authentication"


@dataclass
class CourtroomExhibit:
    """A marked visual document or physical evidence exhibit presented in court proceedings."""

    exhibit_id: str
    label: str  # e.g., "Exhibit A", "Exhibit 1"
    description: str
    document_id: str
    status: ExhibitStatus = ExhibitStatus.UNMARKED
    stamp_coordinates: tuple[float, float] = (0.80, 0.08)  # Normalized (x, y) placement
    ruling: str | None = None
    objection_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ScreenUIElement:
    """A visual UI element in the case-management or evidence-viewer software interface."""

    element_id: str
    label: str
    element_type: str  # "button" | "evidence_viewer" | "status_banner" | "modal_dialog"
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseManagementDisplayState:
    """Visual mock representation of the case-management software and exhibit screens."""

    screen_resolution: tuple[int, int] = (1920, 1080)
    active_window: str = "CourtroomEvidenceViewer"
    elements: list[ScreenUIElement] = field(default_factory=list)
    active_exhibit: CourtroomExhibit | None = None
    error_banner: str | None = None
    modal_dialog: dict[str, Any] | None = None

    def find_element(self, element_id: str) -> ScreenUIElement | None:
        for elem in self.elements:
            if elem.element_id == element_id:
                return elem
        return None


class CourtroomWorldModel:
    """World model managing visual courtroom exhibits, evidence admission workflows, and UI states."""

    def __init__(self) -> None:
        self._exhibits: dict[str, CourtroomExhibit] = {}
        self._active_exhibit_id: str | None = None
        self._current_error: str | None = None
        self._lock = threading.RLock()

    def mark_exhibit(
        self,
        document_id: str,
        label: str,
        description: str,
        stamp_pos: tuple[float, float] = (0.80, 0.08),
    ) -> CourtroomExhibit:
        """Mark a document with an official visual courtroom exhibit stamp."""
        with self._lock:
            exhibit_id = f"exhibit_{secrets.token_hex(4)}_{label.replace(' ', '_').lower()}"
            exhibit = CourtroomExhibit(
                exhibit_id=exhibit_id,
                label=label,
                description=description,
                document_id=document_id,
                status=ExhibitStatus.MARKED,
                stamp_coordinates=stamp_pos,
            )
            self._exhibits[exhibit_id] = exhibit
            self._active_exhibit_id = exhibit_id
            return exhibit

    def offer_exhibit(self, exhibit_id: str) -> CourtroomExhibit:
        """Formally offer marked exhibit for admission into the evidentiary record."""
        with self._lock:
            exhibit = self._get_exhibit(exhibit_id)
            if exhibit.status != ExhibitStatus.MARKED:
                raise ValueError(f"Cannot offer exhibit with status '{exhibit.status.value}'. Must be MARKED.")
            exhibit.status = ExhibitStatus.OFFERED
            return exhibit

    def raise_objection(
        self,
        exhibit_id: str,
        objection_type: ObjectionType,
        rationale: str,
        objecting_party: str = "opposing_counsel",
    ) -> CourtroomExhibit:
        """Record formal legal objection to exhibit admission."""
        with self._lock:
            exhibit = self._get_exhibit(exhibit_id)
            exhibit.status = ExhibitStatus.OBJECTED
            record = {
                "objection_type": objection_type.value,
                "rationale": rationale,
                "objecting_party": objecting_party,
            }
            exhibit.objection_history.append(record)
            return exhibit

    def admit_exhibit(
        self,
        exhibit_id: str,
        sustain_objection: bool = False,
        judicial_rationale: str = "Admitted under statutory business records exception.",
    ) -> CourtroomExhibit:
        """Administrative Law Judge ruling on exhibit admission."""
        with self._lock:
            exhibit = self._get_exhibit(exhibit_id)
            if sustain_objection:
                exhibit.status = ExhibitStatus.REJECTED
                exhibit.ruling = f"Sustained: {judicial_rationale}"
            else:
                exhibit.status = ExhibitStatus.ADMITTED
                exhibit.ruling = f"Overruled / Admitted: {judicial_rationale}"
            return exhibit

    def render_screen_mock(self, active_exhibit_id: str | None = None) -> CaseManagementDisplayState:
        """Render mock visual UI representation of the courtroom case-management screen."""
        with self._lock:
            ex_id = active_exhibit_id or self._active_exhibit_id
            active_ex = self._exhibits.get(ex_id) if ex_id else None

            elements = [
                ScreenUIElement(
                    element_id="btn_mark_exhibit",
                    label="Mark Exhibit",
                    element_type="button",
                    bbox=(20, 20, 160, 40),
                ),
                ScreenUIElement(
                    element_id="btn_offer_admission",
                    label="Offer for Admission",
                    element_type="button",
                    bbox=(200, 20, 180, 40),
                ),
                ScreenUIElement(
                    element_id="viewport_evidence_display",
                    label="Active Exhibit Viewer",
                    element_type="evidence_viewer",
                    bbox=(20, 80, 1400, 950),
                    metadata={"stamp_visible": active_ex is not None and active_ex.status != ExhibitStatus.UNMARKED},
                ),
                ScreenUIElement(
                    element_id="panel_objections",
                    label="Objection History",
                    element_type="status_banner",
                    bbox=(1440, 80, 460, 950),
                ),
            ]

            error_banner = self._current_error
            modal = None
            if self._current_error:
                modal = {
                    "dialog_id": "dlg_ui_error",
                    "title": "System Warning",
                    "message": self._current_error,
                    "actions": ["Dismiss", "Retry"],
                }

            return CaseManagementDisplayState(
                screen_resolution=(1920, 1080),
                active_window="CourtroomEvidenceViewer",
                elements=elements,
                active_exhibit=copy.deepcopy(active_ex),
                error_banner=error_banner,
                modal_dialog=modal,
            )

    def simulate_ui_error(self, error_code: str, error_message: str) -> CaseManagementDisplayState:
        """Simulate unexpected case-management software error banner or UI modal failure."""
        with self._lock:
            self._current_error = f"[{error_code}] {error_message}"
            return self.render_screen_mock()

    def handle_ui_error(self) -> bool:
        """Acknowledge and clear active UI error state."""
        with self._lock:
            if self._current_error is not None:
                self._current_error = None
                return True
            return False

    def _get_exhibit(self, exhibit_id: str) -> CourtroomExhibit:
        if exhibit_id not in self._exhibits:
            raise KeyError(f"Exhibit '{exhibit_id}' not found in courtroom world model.")
        return self._exhibits[exhibit_id]


__all__ = [
    "ExhibitStatus",
    "ObjectionType",
    "CourtroomExhibit",
    "ScreenUIElement",
    "CaseManagementDisplayState",
    "CourtroomWorldModel",
]
