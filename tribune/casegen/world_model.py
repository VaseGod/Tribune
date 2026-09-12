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

    def score_transition(
        self,
        current_state: Any,
        proposed_turn: dict[str, Any] | Any,
    ) -> float:
        """Score state transition against courtroom evidentiary procedural invariants.

        Returns transition confidence Vt in [0.0, 1.0].
        """
        with self._lock:
            turn_data = proposed_turn if isinstance(proposed_turn, dict) else getattr(proposed_turn, "data", {})
            action = turn_data.get("action", "") or (proposed_turn.get("action") if isinstance(proposed_turn, dict) else getattr(proposed_turn, "action", ""))
            exhibit_id = turn_data.get("exhibit_id")

            # Check exhibit status transition validity
            if action in ("offer", "offer_exhibit") and exhibit_id:
                ex = self._exhibits.get(exhibit_id)
                if not ex or ex.status != ExhibitStatus.MARKED:
                    return 0.10  # Breach: Offering an exhibit before marking
            elif action in ("admit", "admit_exhibit") and exhibit_id:
                ex = self._exhibits.get(exhibit_id)
                if not ex or ex.status not in (ExhibitStatus.OFFERED, ExhibitStatus.OBJECTED):
                    return 0.05  # Breach: Admitting unoffered or unmarked exhibit

            # Check for unresolved critical UI errors
            if self._current_error and action not in ("handle_error", "dismiss_error"):
                return 0.25

            return 0.98

    def _get_exhibit(self, exhibit_id: str) -> CourtroomExhibit:
        if exhibit_id not in self._exhibits:
            raise KeyError(f"Exhibit '{exhibit_id}' not found in courtroom world model.")
        return self._exhibits[exhibit_id]


@dataclass
class StatutoryInvariant:
    """A symbolic legal constraint evaluated during trajectory transitions."""

    invariant_id: str
    description: str
    predicate_type: str  # "income_bound" | "temporal_window" | "resource_limit" | "consistency"
    penalty: float = 0.80  # Penalty deducted from confidence upon breach


class StatutoryWorldModel:
    """World model enforcing symbolic ontology, statutory invariants, and state logic.

    Evaluates state transitions in real time to yield conformal transition scores Vt in [0.0, 1.0].
    """

    def __init__(
        self,
        jurisdiction: str = "EX",
        fpl_base: float = 1255.0,  # Monthly Federal Poverty Level base (1-person)
        fpl_per_person: float = 438.0,  # Additional per household member
        snap_gross_pct: float = 1.30,  # 130% FPL
        appeal_window_days: int = 90,  # Statutory deadline
        asset_limit: float = 2750.0,  # Liquid resource limit
        verifier_gateway: Any = None,  # Optional Phase 2 process verifier
    ) -> None:
        self.jurisdiction = jurisdiction
        self.fpl_base = fpl_base
        self.fpl_per_person = fpl_per_person
        self.snap_gross_pct = snap_gross_pct
        self.appeal_window_days = appeal_window_days
        self.asset_limit = asset_limit
        self.verifier_gateway = verifier_gateway
        self._lock = threading.RLock()

    def get_fpl_monthly_limit(self, household_size: int, pct: float | None = None) -> float:
        """Compute statutory monthly poverty limit."""
        size = max(1, household_size)
        base = self.fpl_base + (size - 1) * self.fpl_per_person
        multiplier = pct if pct is not None else self.snap_gross_pct
        return round(base * multiplier, 2)

    def score_transition(
        self,
        current_state: dict[str, Any] | Any,
        proposed_turn: dict[str, Any] | Any,
    ) -> float:
        """Evaluate statutory invariants on state transition (current_state, proposed_turn).

        Returns:
            Transition score Vt in [0.0, 1.0] (probability/confidence of statutory validity).
        """
        with self._lock:
            # Delegate to high-factuality process verifier gateway if configured
            if self.verifier_gateway is not None and hasattr(self.verifier_gateway, "verify_statutory_invariant"):
                try:
                    res = self.verifier_gateway.verify_statutory_invariant(current_state, proposed_turn)
                    if isinstance(res, dict) and "score" in res:
                        return float(res["score"])
                except Exception:
                    pass

            # Extract state and turn representations
            curr = current_state if isinstance(current_state, dict) else getattr(current_state, "facts", {})
            turn = proposed_turn if isinstance(proposed_turn, dict) else getattr(proposed_turn, "data", {})
            if not turn and hasattr(proposed_turn, "action"):
                turn = {"action": proposed_turn.action}

            hh_size = int(turn.get("household_size", curr.get("household_size", 1)))
            income = float(turn.get("monthly_income", turn.get("reported_income", curr.get("monthly_income", 0.0))))
            assets = float(turn.get("liquid_assets", curr.get("liquid_assets", 0.0)))
            days_denial = turn.get("days_since_denial", curr.get("days_since_denial"))

            confidence = 1.0

            # 1. Statutory Gross Income Invariant (e.g. SNAP 130% FPL)
            limit = self.get_fpl_monthly_limit(hh_size)
            if income > limit:
                excess_ratio = (income - limit) / max(1.0, limit)
                # Significant overshoot incurs immediate heavy penalty
                penalty = min(0.90, 0.40 + excess_ratio * 0.50)
                confidence -= penalty

            # 2. Statutory Appeal Window Invariant (Must be <= appeal_window_days)
            if days_denial is not None:
                days = float(days_denial)
                if days > self.appeal_window_days:
                    # Time-barred appeal past statutory window
                    excess_days = days - self.appeal_window_days
                    confidence -= min(0.85, 0.50 + (excess_days / 30.0) * 0.35)
                elif days < 0:
                    # Negative elapsed days (chronological impossibility)
                    confidence -= 0.80

            # 3. Statutory Asset Limitation Invariant
            if assets > self.asset_limit:
                excess = (assets - self.asset_limit) / self.asset_limit
                confidence -= min(0.70, 0.30 + excess * 0.40)
            elif assets < 0:
                confidence -= 0.75  # Impossible negative assets

            # 4. Latent Contradiction Invariants
            # If discovery reveals unverified income or asset contradictions
            if turn.get("has_contradiction", False) or turn.get("contradiction_detected", False):
                confidence -= 0.65
            if turn.get("unverified_secondary_wages", 0.0) > 0.0:
                total = income + float(turn["unverified_secondary_wages"])
                if total > limit:
                    confidence -= 0.70

            # 5. Evidentiary Procedural Invariant
            if turn.get("exhibit_status") == "admitted" and not curr.get("evidence_offered", True):
                confidence -= 0.75

            # Bounded return
            return round(min(1.0, max(0.0, confidence)), 4)

    def compute_nonconformity(
        self,
        current_state: dict[str, Any] | Any,
        proposed_turn: dict[str, Any] | Any,
    ) -> float:
        """Compute non-conformity score St = 1.0 - Vt."""
        return round(1.0 - self.score_transition(current_state, proposed_turn), 4)


__all__ = [
    "ExhibitStatus",
    "ObjectionType",
    "CourtroomExhibit",
    "ScreenUIElement",
    "CaseManagementDisplayState",
    "CourtroomWorldModel",
    "StatutoryInvariant",
    "StatutoryWorldModel",
]

