"""Plain-language disclosure of a run.

Renders an artifact a person or caseworker can actually read and contest: which
programs were considered, what the rules say, which evidence was used, the
verifier's verdict, and — crucially — where TRIBUNE abstained and routed to a
human. This is the "here's why" explanation built from the audit trail.
"""

from __future__ import annotations

from typing import Any

from ..corpus.citations import format_citation
from ..types import (
    CaseRunResult,
    CriterionOutcome,
    EligibilityStatus,
    ProgramOutcome,
    RecommendedAction,
    WaitlistStatus,
)

_PROGRAM_NAMES = {
    "snap": "SNAP (food assistance)",
    "unemployment": "Unemployment insurance",
    "medicaid": "Medicaid / health coverage",
    "housing": "Housing assistance (Housing Choice Voucher)",
    "appeals": "Appeal / fair hearing",
}

_STATUS_PHRASE = {
    EligibilityStatus.LIKELY_ELIGIBLE: "your situation appears to support eligibility",
    EligibilityStatus.LIKELY_INELIGIBLE: "your situation does not appear to meet the rules",
    EligibilityStatus.INDETERMINATE: "there is not enough information to say",
}

_ACTION_PHRASE = {
    RecommendedAction.PREPARE_APPLICATION: "TRIBUNE can help you prepare an application.",
    RecommendedAction.GATHER_MORE_EVIDENCE: "More documentation is needed before this can be assessed.",
    RecommendedAction.ABSTAIN_AND_ESCALATE: "TRIBUNE is not confident enough to say; a navigator should review this with you.",
}

_HEADER = (
    "TRIBUNE — Plain-language explanation of this review\n"
    "This is informational only. It is NOT legal advice and NOT an official "
    "determination. A navigator, caseworker, or the administering agency makes the "
    "final decision. You can always ask a person for help.\n"
)


def _humanize_program(value: str) -> str:
    return _PROGRAM_NAMES.get(value, value)


def _render_outcome(o: ProgramOutcome) -> str:
    name = _humanize_program(o.program.value)
    lines = [f"== {name} =="]

    if o.assessment is None:
        lines.append("  No assessment was produced.")
        return "\n".join(lines)

    a = o.assessment
    if o.abstained:
        lines.append("  Outcome: TRIBUNE ABSTAINED and is routing you to a human.")
        if o.abstention is not None:
            lines.append(
                f"  Why: {o.abstention.reason} "
                f"(confidence {o.abstention.calibrated_confidence:.2f} < "
                f"threshold {o.abstention.threshold:.2f})."
            )
        lines.append(f"  Human path: {o.human_path}")
    else:
        lines.append(f"  Outcome: {_STATUS_PHRASE[a.status]}.")
        lines.append(f"  Next step: {_ACTION_PHRASE[a.recommended_action]}")

    # What the rules said and which evidence was used.
    lines.append("  What the rules require, and how your evidence lined up:")
    for crit in a.criteria:
        mark = {
            CriterionOutcome.SATISFIED: "appears met",
            CriterionOutcome.NOT_SATISFIED: "does not appear met",
            CriterionOutcome.UNKNOWN: "not enough evidence yet",
        }[crit.outcome]
        lines.append(f"    - {crit.description}: {mark}")

    if a.citations:
        lines.append("  Governing rules cited:")
        for c in a.citations:
            lines.append(f"    - {format_citation(c)}")

    if o.program.value == "housing" and a.waitlist_status is not None:
        ws = a.waitlist_status
        note = {
            WaitlistStatus.OPEN: "the waitlist is open, but being eligible does not guarantee or speed up assistance",
            WaitlistStatus.CLOSED: "the waitlist is CLOSED — even if eligible, you cannot apply for a voucher right now",
            WaitlistStatus.UNKNOWN: "the waitlist status is unknown — eligibility is not the same as access; check with the housing agency",
        }[ws]
        lines.append(f"  Important — eligibility is not access: {note}.")

    if o.verdict is not None:
        verdict = "passed independent verification" if o.verdict.approved else "did NOT pass independent verification"
        lines.append(f"  Verifier: this assessment {verdict}.")
        for r in o.verdict.reasons:
            lines.append(f"    · {r}")

    if o.materials is not None:
        lines.append(
            "  Prepared (NOT submitted): "
            + ", ".join(o.materials.document_checklist[:6])
            + (" ..." if len(o.materials.document_checklist) > 6 else "")
        )
        lines.append(f"  {o.materials.action_required_note}")

    return "\n".join(lines)


def generate_determination_notice(
    outcome: ProgramOutcome,
    verification_report: Any | None = None,
    action_gate: Any | None = None,
) -> str:
    """Pass 2 (Notice Generation): Generate a formal determination and disclosure notice.

    Only executes after Pass 1 milestone verification has certified candidate eligibility
    findings and statutory citations via the ActionGate.
    """
    if outcome.assessment is None:
        raise ValueError("Cannot generate determination notice without an assessment.")

    if action_gate is not None:
        action_gate.authorize_notice_generation(
            assessment=outcome.assessment,
            verification_report=verification_report,
        )
    elif verification_report is not None and not getattr(verification_report, "is_certified", False):
        from .action_gate import ActionBlocked
        raise ActionBlocked(
            "Formal determination notice blocked: evaluation failed milestone verification."
        )

    assessment = outcome.assessment
    prog_name = _humanize_program(outcome.program.value)

    lines = [
        "================================================================================",
        f"OFFICIAL DETERMINATION & DISCLOSURE NOTICE — {prog_name.upper()}",
        f"Jurisdiction: {assessment.jurisdiction} | Case ID: {assessment.case_id} | Reference: {assessment.assessment_id}",
        "================================================================================",
        "",
        "1. PRELIMINARY DETERMINATION STATUS",
        f"   Status: {assessment.status.value.upper()} ({_STATUS_PHRASE[assessment.status]})",
        f"   Recommended Action: {_ACTION_PHRASE[assessment.recommended_action]}",
        f"   Assessment Confidence: {assessment.self_confidence:.2%}",
        "",
        "2. STATUTORY LEGAL BASIS & GOVERNING CITATIONS",
    ]

    for c in assessment.citations:
        lines.append(f"   • {format_citation(c)}")
        if c.text:
            lines.append(f"     Rule Summary: {c.text[:140]}...")

    lines.extend([
        "",
        "3. EVALUATED STATUTORY CRITERIA & EVIDENCE FINDINGS",
    ])

    for crit in assessment.criteria:
        mark = {
            CriterionOutcome.SATISFIED: "[MET] Appears satisfied based on documentation",
            CriterionOutcome.NOT_SATISFIED: "[NOT MET] Does not meet statutory threshold",
            CriterionOutcome.UNKNOWN: "[INCOMPLETE] Insufficient documentation to resolve",
        }[crit.outcome]
        cids = f" (Citations: {', '.join(crit.citation_ids)})" if crit.citation_ids else ""
        lines.append(f"   • {crit.description}: {mark}{cids}")

    lines.extend([
        "",
        "4. TWO-STAGE VERIFICATION CERTIFICATION",
    ])
    if outcome.verdict is not None and outcome.verdict.approved:
        lines.append("   [CERTIFIED] Passed independent statutory re-derivation and milestone verification.")
        for r in outcome.verdict.reasons:
            lines.append(f"   - {r}")
    elif verification_report is not None and getattr(verification_report, "is_certified", False):
        lines.append("   [CERTIFIED] Passed Pass 1 statutory milestone verification.")
    else:
        lines.append("   [PENDING/UNVERIFIED] Not yet formally certified by independent verifier.")

    lines.extend([
        "",
        "5. ADVERSE ACTION & FAIR HEARING APPEAL RIGHTS",
        "   If you disagree with any portion of this preliminary determination:",
        "   - You have the right to request an administrative Fair Hearing.",
        "   - Appeal requests must be submitted within the statutory window from receipt of notice.",
        "   - You have the right to inspect all evidence in your case file and be represented.",
        "   - If currently receiving benefits, you may request continuation of benefits pending appeal.",
        "",
        "6. IMPORTANT NOTICE",
        "   This document is generated by the TRIBUNE automated administrative assistant.",
        "   It is NOT binding legal advice. The administering public agency makes all final determinations.",
        "================================================================================",
    ])

    return "\n".join(lines)


def render(result: CaseRunResult) -> str:
    parts = [_HEADER, f"Case: {result.case_id}  |  Jurisdiction: {result.jurisdiction}\n"]
    for o in result.outcomes:
        parts.append(_render_outcome(o))
        parts.append("")
    parts.append(
        "Audit: every step above was written to an append-only, hash-chained audit "
        f"log ({len(result.audit)} records) that you or a caseworker can inspect."
    )
    return "\n".join(parts)


__all__ = ["render", "generate_determination_notice"]
