"""Deterministic Task Complexity Classifier & Model Routing Engine.

Assesses task complexity prior to inference dispatch using deterministic rule-based
signals:
1. Document text length and token density.
2. Number and density of statutory / reporter citations.
3. Party count and multi-party liability indicators.
4. Presence of constitutional, jurisdictional, or standing arguments.
5. Procedural deadline complexity.
6. History of prior verifier gate failures (escalation trigger).

Routes tasks between:
- Worker Tier (DeepSeek-V4.1-Flash / Swift-Qwen3.8-27B): Administrative workflows,
  notice date verifications, form completions, docketing.
- Lead Tier (Frontier / Astra / Claude Opus / GPT-4o): Complex statutory interpretation,
  constitutional standing, appellate argument synthesis.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class TaskComplexityClass(str, enum.Enum):
    """Complexity classification tier for incoming legal tasks."""

    STANDARD_ADMINISTRATIVE = "STANDARD_ADMINISTRATIVE"
    COMPLEX_STATUTORY = "COMPLEX_STATUTORY"


class TaskWorkflowCategory(str, enum.Enum):
    """Specific workflow category within benefits adjudication."""

    NOTICE_DATE_VERIFICATION = "notice_date_verification"
    PROCEDURAL_DEADLINE_TRACKING = "procedural_deadline_tracking"
    TEMPLATE_GENERATION = "template_generation"
    DOCKET_ENTRY_DRAFTING = "docket_entry_drafting"
    SIMPLE_FORM_COMPLETION = "simple_form_completion"
    CONSTITUTIONAL_STANDING = "constitutional_standing"
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    MULTI_PARTY_LIABILITY = "multi_party_liability"
    APPELLATE_ARGUMENT_SYNTHESIS = "appellate_argument_synthesis"
    GENERAL_LEGAL_ANALYSIS = "general_legal_analysis"


@dataclass
class TaskRoutingDecision:
    """Routing metadata emitted prior to inference dispatch."""

    task_class: TaskComplexityClass
    selected_tier: str  # "lead" | "worker"
    selected_model: str
    expected_token_budget: int
    routing_reason: str
    complexity_score: float
    signals: dict[str, Any] = field(default_factory=dict)


# Keywords indicating complex statutory and constitutional evaluations
_COMPLEX_SIGNALS = [
    (re.compile(r"\b(?:constitutional|due\s+process|equal\s+protection|first\s+amendment|fourteenth\s+amendment)\b", re.IGNORECASE), "Constitutional rights or protections"),
    (re.compile(r"\b(?:standing|article\s+iii|ripeness|mootness|subject\s+matter\s+jurisdiction)\b", re.IGNORECASE), "Jurisdictional or standing threshold"),
    (re.compile(r"\b(?:multi-party|joint\s+and\s+several|indemnif|contribution|cross-claim)\b", re.IGNORECASE), "Multi-party liability issue"),
    (re.compile(r"\b(?:statutory\s+interpretation|ambigu|chevron|skidmore|canons?\s+of\s+construction)\b", re.IGNORECASE), "Statutory interpretation ambiguity"),
    (re.compile(r"\b(?:appellate\s+brief|oral\s+argument\s+prep|standard\s+of\s+review|de\s+novo|clear\s+error)\b", re.IGNORECASE), "Appellate synthesis"),
]

# Keywords indicating standard administrative tasks
_ADMIN_SIGNALS = [
    (re.compile(r"\b(?:notice\s+date|filing\s+date|timestamp|calendar\s+days|deadline\s+calculation)\b", re.IGNORECASE), "Procedural deadline check"),
    (re.compile(r"\b(?:docket\s+entry|clerk\s+minute|status\s+update|case\s+log)\b", re.IGNORECASE), "Docket entry drafting"),
    (re.compile(r"\b(?:intake\s+form|applicant\s+questionnaire|income\s+w2\s+fill|form\s+completion)\b", re.IGNORECASE), "Form completion"),
    (re.compile(r"\b(?:template\s+fill|standard\s+denial\s+letter|acknowledgment\s+notice)\b", re.IGNORECASE), "Standard notice generation"),
]


class DeterministicTaskClassifier:
    """Rule-based, deterministic task classifier and routing engine."""

    def __init__(
        self,
        worker_model: str = "deepseek-v4.1-flash",
        lead_model: str = "gpt-4o",
        complexity_threshold: float = 0.40,
    ) -> None:
        self.worker_model = worker_model
        self.lead_model = lead_model
        self.complexity_threshold = complexity_threshold

    def classify(
        self,
        task_text: str,
        citations_count: int = 0,
        party_count: int = 1,
        prior_verifier_failures: int = 0,
        explicit_category: TaskWorkflowCategory | None = None,
        jurisdictional_risk: bool = False,
    ) -> TaskRoutingDecision:
        """Deterministically assess complexity and compute routing decision."""
        score = 0.0
        signals: dict[str, Any] = {
            "citations_count": citations_count,
            "party_count": party_count,
            "prior_verifier_failures": prior_verifier_failures,
            "jurisdictional_risk": jurisdictional_risk,
            "text_length": len(task_text),
        }
        reasons: list[str] = []

        # 1. Check explicit category if provided
        if explicit_category:
            signals["explicit_category"] = explicit_category.value
            if explicit_category in (
                TaskWorkflowCategory.NOTICE_DATE_VERIFICATION,
                TaskWorkflowCategory.PROCEDURAL_DEADLINE_TRACKING,
                TaskWorkflowCategory.TEMPLATE_GENERATION,
                TaskWorkflowCategory.DOCKET_ENTRY_DRAFTING,
                TaskWorkflowCategory.SIMPLE_FORM_COMPLETION,
            ):
                return TaskRoutingDecision(
                    task_class=TaskComplexityClass.STANDARD_ADMINISTRATIVE,
                    selected_tier="worker",
                    selected_model=self.worker_model,
                    expected_token_budget=1_500,
                    routing_reason=f"Explicit administrative category '{explicit_category.value}'",
                    complexity_score=0.15,
                    signals=signals,
                )
            elif explicit_category in (
                TaskWorkflowCategory.CONSTITUTIONAL_STANDING,
                TaskWorkflowCategory.STATUTORY_INTERPRETATION,
                TaskWorkflowCategory.MULTI_PARTY_LIABILITY,
                TaskWorkflowCategory.APPELLATE_ARGUMENT_SYNTHESIS,
            ):
                return TaskRoutingDecision(
                    task_class=TaskComplexityClass.COMPLEX_STATUTORY,
                    selected_tier="lead",
                    selected_model=self.lead_model,
                    expected_token_budget=8_000,
                    routing_reason=f"Explicit high-entropy category '{explicit_category.value}'",
                    complexity_score=0.85,
                    signals=signals,
                )

        # 2. Score text against complex legal keywords
        for pattern, desc in _COMPLEX_SIGNALS:
            if pattern.search(task_text):
                score += 0.35
                reasons.append(desc)

        # 3. Document length signal
        if len(task_text) > 8000:
            score += 0.20
            reasons.append(f"Long document length ({len(task_text)} chars)")
        elif len(task_text) > 3000:
            score += 0.10

        # 4. Citations count signal
        if citations_count >= 5:
            score += 0.25
            reasons.append(f"High citation density ({citations_count} citations)")
        elif citations_count >= 2:
            score += 0.10

        # 5. Multi-party signal
        if party_count > 2:
            score += 0.25
            reasons.append(f"Multiple parties involved ({party_count} parties)")

        # 6. Prior verifier failures trigger immediate escalation to lead tier
        if prior_verifier_failures > 0:
            score += 0.40
            reasons.append(f"Escalation from {prior_verifier_failures} prior verifier gate rejections")

        # 7. Jurisdictional risk flag
        if jurisdictional_risk:
            score += 0.20
            reasons.append("High jurisdictional divergence risk")

        clamped_score = min(1.0, round(score, 3))
        signals["matched_reasons"] = reasons

        if clamped_score >= self.complexity_threshold:
            decision_class = TaskComplexityClass.COMPLEX_STATUTORY
            tier = "lead"
            model = self.lead_model
            budget = 8_000
            primary_reason = f"Complex statutory workload (score {clamped_score:.2f} >= {self.complexity_threshold}): {', '.join(reasons) if reasons else 'High entropy'}"
        else:
            decision_class = TaskComplexityClass.STANDARD_ADMINISTRATIVE
            tier = "worker"
            model = self.worker_model
            budget = 2_000
            primary_reason = f"Standard administrative workload (score {clamped_score:.2f} < {self.complexity_threshold})"

        logger.info(f"[TaskClassifier] Routed task to tier={tier} (model={model}, score={clamped_score})")

        return TaskRoutingDecision(
            task_class=decision_class,
            selected_tier=tier,
            selected_model=model,
            expected_token_budget=budget,
            routing_reason=primary_reason,
            complexity_score=clamped_score,
            signals=signals,
        )
