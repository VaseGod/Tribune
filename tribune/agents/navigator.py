from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..instrumentation.usage import estimate_tokens
from ..orchestration.dag import DAG, Task
from ..types import ProgramId, SyntheticCase

if TYPE_CHECKING:
    from ..corpus.rule_store import RuleStore

logger = logging.getLogger(__name__)

# Constants
MAX_INTERMEDIATE_TOOL_TOKENS: int = 512
DEFAULT_ROLLING_WINDOW: int = 50
DEFAULT_MINIMUM_SAMPLES: int = 5
DEFAULT_SAFETY_MULTIPLIER: float = 1.25
DEFAULT_HORIZON_STEPS: int = 15
ABSOLUTE_MAX_STEPS: int = 50


# --------------------------------------------------------------------------- #
# Visual Layout Cross-Referencing & Agency Rendering Rule Enforcer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgencyLayoutRule:
    """An agency document rendering rule governing element placement and geometry."""

    rule_name: str
    description: str
    target_type: str  # "HEADER" | "KEY_VALUE" | "PARAGRAPH" | "SIGNATURE_BLOCK" | "NOTICE_DATE"
    min_y: float = 0.0
    max_y: float = 1.0
    min_x: float = 0.0
    max_x: float = 1.0
    required: bool = True


@dataclass
class LayoutVerificationReport:
    """Structured report validating generated document layout against statutory rendering rules."""

    is_compliant: bool
    compliance_score: float  # [0.0, 1.0]
    violations: list[str] = field(default_factory=list)
    detected_elements: list[str] = field(default_factory=list)
    column_ordering_valid: bool = True
    header_placement_valid: bool = True
    signature_block_valid: bool = True
    reasons: list[str] = field(default_factory=list)


class VisualLayoutCrossReferencer:
    """Cross-references document visual layouts against agency rendering and geometry specifications."""

    DEFAULT_AGENCY_RULES: list[AgencyLayoutRule] = [
        AgencyLayoutRule(
            rule_name="header_top_placement",
            description="Agency header must be rendered in the top quadrant (y <= 0.25).",
            target_type="HEADER",
            min_y=0.0,
            max_y=0.25,
            min_x=0.0,
            max_x=1.0,
            required=True,
        ),
        AgencyLayoutRule(
            rule_name="signature_block_bottom_placement",
            description="Signature / authorization block must be rendered in lower quadrant (y >= 0.65).",
            target_type="SIGNATURE_BLOCK",
            min_y=0.65,
            max_y=1.0,
            min_x=0.2,
            max_x=1.0,
            required=False,
        ),
        AgencyLayoutRule(
            rule_name="notice_date_header_alignment",
            description="Determination issuance date must be in upper half of page.",
            target_type="NOTICE_DATE",
            min_y=0.0,
            max_y=0.40,
            min_x=0.4,
            max_x=1.0,
            required=False,
        ),
    ]

    def __init__(self, rules: list[AgencyLayoutRule] | None = None) -> None:
        self.rules = rules or self.DEFAULT_AGENCY_RULES

    def verify_layout(
        self,
        document_layout: Any,
        template_name: str = "standard_notice",
    ) -> LayoutVerificationReport:
        """Verify visual layout elements, column reading-order DAG, and bounding box compliance."""
        tokens_data = getattr(
            document_layout,
            "tokens",
            document_layout.get("tokens", []) if isinstance(document_layout, dict) else [],
        )
        edges_data = getattr(
            document_layout,
            "edges",
            document_layout.get("edges", []) if isinstance(document_layout, dict) else [],
        )

        violations: list[str] = []
        reasons: list[str] = []
        detected_elements: list[str] = []

        header_valid = True
        sig_valid = True
        col_valid = True

        parsed_tokens = []
        for t in tokens_data:
            if hasattr(t, "token_type") and hasattr(t, "bbox"):
                parsed_tokens.append({
                    "token_id": t.token_id,
                    "text": t.text,
                    "token_type": t.token_type,
                    "bbox": [t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1],
                })
            elif isinstance(t, dict):
                parsed_tokens.append(t)

        for tok in parsed_tokens:
            ttype = tok.get("token_type", "PARAGRAPH")
            detected_elements.append(ttype)
            bbox = tok.get("bbox", [0.0, 0.0, 1.0, 1.0])
            y0, y1 = bbox[1], bbox[3]

            # 1. Header Placement Check
            if ttype == "HEADER":
                if y0 > 0.25:
                    header_valid = False
                    violations.append(
                        f"Header '{tok.get('text', '')[:30]}' displaced to y={y0:.2f} (exceeds max y=0.25)"
                    )

            # 2. Signature Block Placement Check
            if ttype == "SIGNATURE_BLOCK":
                if y1 < 0.65:
                    sig_valid = False
                    violations.append(
                        f"Signature block displaced to y={y1:.2f} (must be in bottom area y >= 0.65)"
                    )

        # 3. Column Ordering & Monotonicity Check across Reading-Order Edges
        tok_map = {t.get("token_id"): t for t in parsed_tokens}
        for u_id, v_id in edges_data:
            if u_id in tok_map and v_id in tok_map:
                u_y = tok_map[u_id].get("bbox", [0, 0, 1, 1])[1]
                v_y = tok_map[v_id].get("bbox", [0, 0, 1, 1])[1]
                if v_y < u_y - 0.35:
                    col_valid = False
                    violations.append(
                        f"Inverted reading order flow between '{u_id}' (y={u_y:.2f}) and '{v_id}' (y={v_y:.2f})"
                    )

        # 4. Mandatory Rules Check
        for rule in self.rules:
            if rule.required:
                matching = [t for t in parsed_tokens if t.get("token_type") == rule.target_type]
                if not matching:
                    violations.append(
                        f"Missing mandatory visual layout element '{rule.target_type}' ({rule.rule_name})"
                    )

        is_compliant = (len(violations) == 0) and header_valid and sig_valid and col_valid
        score = max(0.0, 1.0 - (len(violations) * 0.25))

        if is_compliant:
            reasons.append(
                "Visual document layout fully satisfies agency rendering and column ordering specifications."
            )
        else:
            reasons.extend(violations)

        return LayoutVerificationReport(
            is_compliant=is_compliant,
            compliance_score=round(score, 4),
            violations=violations,
            detected_elements=detected_elements,
            column_ordering_valid=col_valid,
            header_placement_valid=header_valid,
            signature_block_valid=sig_valid,
            reasons=reasons,
        )


# --------------------------------------------------------------------------- #
# Elastic Horizon Controller & Persistence Store
# --------------------------------------------------------------------------- #


class TrajectoryPersistenceStore:
    """Swappable in-memory and file-backed trajectory persistence store."""

    def __init__(self, persistence_file: str | None = None) -> None:
        self.persistence_file = persistence_file
        self._successful_trajectories: list[int] = []
        if self.persistence_file and os.path.exists(self.persistence_file):
            self._load_from_file()

    def _load_from_file(self) -> None:
        try:
            with open(self.persistence_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if record.get("success", False):
                            self._successful_trajectories.append(int(record["length"]))
        except Exception as exc:
            logger.warning(f"Could not load trajectory persistence file: {exc}")

    def record(self, length: int, success: bool = True, metadata: dict[str, Any] | None = None) -> None:
        if success:
            self._successful_trajectories.append(length)
        if self.persistence_file:
            try:
                record = {
                    "length": length,
                    "success": success,
                    "metadata": metadata or {},
                }
                with open(self.persistence_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                logger.warning(f"Failed to persist trajectory record: {exc}")

    def get_successful_lengths(self, limit: int = DEFAULT_ROLLING_WINDOW) -> list[int]:
        return self._successful_trajectories[-limit:]

    def clear(self) -> None:
        self._successful_trajectories.clear()


class ElasticHorizonController:
    """Dynamically bounds planning and execution rollouts using p90 of rolling history."""

    def __init__(
        self,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
        safety_multiplier: float = DEFAULT_SAFETY_MULTIPLIER,
        default_horizon: int = DEFAULT_HORIZON_STEPS,
        absolute_max_steps: int = ABSOLUTE_MAX_STEPS,
        store: TrajectoryPersistenceStore | None = None,
    ) -> None:
        self.rolling_window = rolling_window
        self.minimum_samples = minimum_samples
        self.safety_multiplier = safety_multiplier
        self.default_horizon = default_horizon
        self.absolute_max_steps = absolute_max_steps
        self.store = store or TrajectoryPersistenceStore()

    def record_successful_trajectory(self, length: int, metadata: dict[str, Any] | None = None) -> None:
        """Record a successful trajectory length into the rolling store."""
        self.store.record(length, success=True, metadata=metadata)

    def calculate_p90(self) -> float | None:
        """Compute the 90th percentile trajectory length from recent successful samples."""
        lengths = self.store.get_successful_lengths(self.rolling_window)
        if len(lengths) < self.minimum_samples:
            return None
        sorted_lens = sorted(lengths)
        idx = int(math.ceil(0.90 * len(sorted_lens))) - 1
        return float(sorted_lens[max(0, min(idx, len(sorted_lens) - 1))])

    def get_current_horizon(self) -> int:
        """Compute upper step bound for future rollouts."""
        p90 = self.calculate_p90()
        if p90 is None:
            return self.default_horizon
        computed = int(math.ceil(p90 * self.safety_multiplier))
        return min(self.absolute_max_steps, max(1, computed))

    def check_bound(self, current_step: int) -> tuple[bool, int, str]:
        """Check if current step count breaches elastic horizon bound.

        Returns (is_breached, bound, reason).
        """
        bound = self.get_current_horizon()
        if current_step > bound:
            reason = (
                f"Trajectory exceeded dynamic elastic horizon limit ({current_step} > {bound} steps). "
                "Halting exploratory rollout to prevent trajectory sprawl."
            )
            return True, bound, reason
        return False, bound, ""


# --------------------------------------------------------------------------- #
# Anti-Verbosity & Loop Detector
# --------------------------------------------------------------------------- #


class LoopDetector:
    """Detects repeated identical tool calls, repetitive reasoning blocks, and no-op cycles."""

    def __init__(self, max_consecutive_duplicates: int = 2) -> None:
        self.max_consecutive_duplicates = max_consecutive_duplicates
        self._recent_tool_signatures: list[str] = []
        self._recent_reasoning_hashes: list[str] = []

    def clear(self) -> None:
        self._recent_tool_signatures.clear()
        self._recent_reasoning_hashes.clear()

    @staticmethod
    def _hash_text(text: str) -> str:
        clean = " ".join(text.lower().split())
        return str(hash(clean))

    def record_and_check(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        thought_or_reasoning: str = "",
    ) -> tuple[bool, str]:
        """Check for verbosity loops or repetitive no-op behavior.

        Returns (loop_detected, reason).
        """
        # 1. Check repeated identical tool calls
        if tool_name:
            args_str = json.dumps(tool_args or {}, sort_keys=True)
            tool_sig = f"{tool_name}:{args_str}"
            self._recent_tool_signatures.append(tool_sig)

            # Check consecutive duplicates
            if len(self._recent_tool_signatures) >= self.max_consecutive_duplicates:
                last_n = self._recent_tool_signatures[-self.max_consecutive_duplicates :]
                if len(set(last_n)) == 1:
                    return (
                        True,
                        f"Verbosity loop detected: tool '{tool_name}' invoked with identical arguments "
                        f"{self.max_consecutive_duplicates} consecutive times.",
                    )

        # 2. Check repeated reasoning blocks without progress
        if thought_or_reasoning:
            r_hash = self._hash_text(thought_or_reasoning)
            self._recent_reasoning_hashes.append(r_hash)
            if len(self._recent_reasoning_hashes) >= 3:
                last_3 = self._recent_reasoning_hashes[-3:]
                if len(set(last_3)) == 1:
                    return (
                        True,
                        "Verbosity loop detected: agent repeating identical reasoning block without tool action.",
                    )

        # 3. Check no-op step (empty tool and empty thought)
        if not tool_name and not thought_or_reasoning.strip():
            return True, "No-op step detected: neither tool invocation nor reasoning progress produced."

        return False, ""


@dataclass
class TrajectoryTelemetry:
    """Metrics tracking token economy, step counts, and guardrail enforcements."""

    total_trajectory_tokens: int = 0
    planning_tokens: int = 0
    tool_dispatch_tokens: int = 0
    reasoning_tokens: int = 0
    number_of_steps: int = 0
    aborted_steps: int = 0
    halted_horizons: int = 0
    verbosity_violations: int = 0
    subagent_executions: int = 0
    subagent_dispatch_reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Programmatic Navigator Tools
# --------------------------------------------------------------------------- #


class ProgrammaticNavigatorTools:
    """Typed Python stubs executed directly in-code by navigator agent loops."""

    @staticmethod
    def build_dag(target_programs: list[str]) -> dict:
        """Construct execution DAG tasks for evidence gathering and per-program assessment."""
        tasks = [{"task_id": "gather", "kind": "gather"}]
        for prog in target_programs:
            tasks.append({"task_id": f"assess:{prog}", "kind": "assess", "deps": ["gather"], "program": prog})
        return {"tasks": tasks}

    @staticmethod
    def verify_visual_layout(layout_dict: dict) -> dict:
        """Verify layout geometry against agency specifications."""
        verifier = VisualLayoutCrossReferencer()
        report = verifier.verify_layout(layout_dict)
        return {
            "is_compliant": report.is_compliant,
            "compliance_score": report.compliance_score,
            "violations": report.violations,
        }

    @classmethod
    def get_tool_signatures(cls) -> str:
        """Expose typed Python signatures for model prompt generation."""
        return (
            "class ProgrammaticNavigatorTools:\n"
            "    @staticmethod\n"
            "    def build_dag(target_programs: list[str]) -> dict: ...\n"
            "    @staticmethod\n"
            "    def verify_visual_layout(layout_dict: dict) -> dict: ...\n"
        )


# --------------------------------------------------------------------------- #
# Navigator Agent
# --------------------------------------------------------------------------- #


class Navigator:
    """Navigator agent enforcing single-agent execution topology, Elastic Horizon, and anti-verbosity."""

    def __init__(
        self,
        rule_store: RuleStore | None = None,
        enable_subagents: bool = False,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
        safety_multiplier: float = DEFAULT_SAFETY_MULTIPLIER,
        default_horizon: int = DEFAULT_HORIZON_STEPS,
        max_intermediate_tokens: int = MAX_INTERMEDIATE_TOOL_TOKENS,
        store: TrajectoryPersistenceStore | None = None,
    ) -> None:
        self.rule_store = rule_store
        self.tools = ProgrammaticNavigatorTools()
        self.layout_cross_referencer = VisualLayoutCrossReferencer()

        # Single-agent topology control (default False: no swarm spawning)
        self.enable_subagents = enable_subagents

        # Elastic Horizon controller
        self.horizon_controller = ElasticHorizonController(
            rolling_window=rolling_window,
            safety_multiplier=safety_multiplier,
            default_horizon=default_horizon,
            store=store,
        )

        # Anti-verbosity and loop detection
        self.max_intermediate_tokens = max_intermediate_tokens
        self.loop_detector = LoopDetector()

        # Trajectory telemetry
        self.telemetry = TrajectoryTelemetry()

    def generate_prompt(self, target_programs: list[ProgramId], jurisdiction: str = "EX") -> str:
        """Generate prompt incorporating programmatic Python tool signatures strictly scoped to target programs."""
        progs = [p.value for p in target_programs]
        scoped_schemas = ""
        if self.rule_store is not None:
            schemas = [self.rule_store.get_scoped_schema(p, jurisdiction) for p in target_programs]
            scoped_schemas = f"\nScoped Program Schemas (Unselected domain schemas pruned): {schemas}\n"

        # Anti-verbosity prompt constraints
        anti_verbosity_instructions = (
            "\nANTI-VERBOSITY CONSTRAINTS:\n"
            f"- Limit intermediate thoughts before tool dispatch to {self.max_intermediate_tokens} tokens.\n"
            "- Do not output conversational explanations or filler commentary during tool calls.\n"
            "- Do not repeat identical tool calls or reasoning patterns.\n"
        )

        return (
            f"You are the navigator agent planning tasks for programs: {progs}.\n"
            f"{scoped_schemas}"
            f"{anti_verbosity_instructions}"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticNavigatorTools.get_tool_signatures()}\n"
            "Use these tools directly to decompose workflow plans into execution DAGs and cross-reference visual layouts."
        )

    def plan(
        self,
        case: SyntheticCase,
        decoupled_modules: list[str] | None = None,
    ) -> DAG:
        """Construct execution DAG enforcing unified single-agent execution topology by default."""
        dag = DAG()

        # Check whether subagents are allowed and strictly decoupled
        subagents_permitted = self.enable_subagents and bool(decoupled_modules)

        if subagents_permitted:
            self.telemetry.subagent_executions += 1
            reason = f"Explicitly enabled for decoupled modules: {decoupled_modules}"
            self.telemetry.subagent_dispatch_reasons.append(reason)
            logger.info(f"[NAVIGATOR] Spawning subagents: {reason}")
            gather_subagent = "subagent_gather"
        else:
            gather_subagent = None  # Unified single-agent topology

        dag.add(Task(task_id="gather", kind="gather", subagent_id=gather_subagent))

        for program in case.target_programs:
            prog_subagent = f"subagent_{program.value}" if subagents_permitted else None
            dag.add(
                Task(
                    task_id=f"assess:{program.value}",
                    kind="assess",
                    deps=["gather"],
                    program=program,
                    subagent_id=prog_subagent,
                )
            )

        plan_tokens = estimate_tokens(str([t.task_id for t in dag.tasks.values()]))
        self.telemetry.planning_tokens += plan_tokens
        self.telemetry.total_trajectory_tokens += plan_tokens
        return dag

    def execute_trajectory_step(
        self,
        current_step: int,
        intermediate_thought: str = "",
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a trajectory step under Elastic Horizon and anti-verbosity guardrails."""
        self.telemetry.number_of_steps += 1

        # 1. Elastic Horizon Check
        is_breached, bound, reason = self.horizon_controller.check_bound(current_step)
        if is_breached:
            self.telemetry.halted_horizons += 1
            self.telemetry.aborted_steps += 1
            logger.warning(f"[ELASTIC_HORIZON] {reason}")
            return {
                "halted": True,
                "reason": reason,
                "status": "horizon_limit_reached",
                "escalate_to_arbitration": True,
                "bound": bound,
            }

        # 2. Enforce 512-token ceiling on intermediate tool thought
        thought_tokens = estimate_tokens(intermediate_thought)
        self.telemetry.reasoning_tokens += thought_tokens
        if thought_tokens > self.max_intermediate_tokens:
            self.telemetry.verbosity_violations += 1
            logger.warning(
                f"[ANTI_VERBOSITY] Intermediate thought ({thought_tokens} tokens) "
                f"exceeds ceiling ({self.max_intermediate_tokens}). Clamping output."
            )
            # Suppress conversational verbosity
            char_limit = self.max_intermediate_tokens * 4
            intermediate_thought = intermediate_thought[:char_limit] + "\n[Thought suppressed for brevity]"
            thought_tokens = self.max_intermediate_tokens

        # 3. Loop Detection
        is_loop, loop_reason = self.loop_detector.record_and_check(
            tool_name=tool_name,
            tool_args=tool_args,
            thought_or_reasoning=intermediate_thought,
        )
        if is_loop:
            self.telemetry.aborted_steps += 1
            logger.warning(f"[LOOP_DETECTOR] {loop_reason}")
            return {
                "halted": True,
                "reason": loop_reason,
                "status": "loop_detected",
                "escalate_to_arbitration": True,
            }

        # 4. Accounting
        tool_dispatch_tokens = estimate_tokens(f"{tool_name}:{json.dumps(tool_args or {})}")
        self.telemetry.tool_dispatch_tokens += tool_dispatch_tokens
        self.telemetry.total_trajectory_tokens += (thought_tokens + tool_dispatch_tokens)

        return {
            "halted": False,
            "step": current_step,
            "thought": intermediate_thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "thought_tokens": thought_tokens,
            "tool_dispatch_tokens": tool_dispatch_tokens,
        }

    def verify_layout_compliance(
        self,
        document_layout: Any,
        template_name: str = "standard_notice",
    ) -> LayoutVerificationReport:
        """Cross-reference visual layout elements against agency rendering rules."""
        return self.layout_cross_referencer.verify_layout(document_layout, template_name)

    def cross_reference_claims(
        self,
        claims: list[dict[str, Any]] | list[str],
        program: str | ProgramId,
        jurisdiction: str = "EX",
    ) -> dict[str, Any]:
        """Cross-reference navigator statutory eligibility claims against local rules."""
        from ..providers.local_rules import cross_evaluate_rule_citations

        prog_str = program.value if isinstance(program, ProgramId) else str(program)
        citations = []
        for c in claims:
            cid = c.get("citation_id") or c.get("citation") if isinstance(c, dict) else str(c)
            if cid:
                citations.append(str(cid))
        return cross_evaluate_rule_citations(citations, program=prog_str, jurisdiction=jurisdiction)

    @staticmethod
    def target_programs(case: SyntheticCase) -> list[ProgramId]:
        return list(case.target_programs)


__all__ = [
    "AgencyLayoutRule",
    "LayoutVerificationReport",
    "VisualLayoutCrossReferencer",
    "TrajectoryPersistenceStore",
    "ElasticHorizonController",
    "LoopDetector",
    "TrajectoryTelemetry",
    "ProgrammaticNavigatorTools",
    "Navigator",
]
