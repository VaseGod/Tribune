from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..orchestration.dag import DAG, Task
from ..types import ProgramId, SyntheticCase

if TYPE_CHECKING:
    from ..corpus.rule_store import RuleStore


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
        tokens_data = getattr(document_layout, "tokens", document_layout.get("tokens", []) if isinstance(document_layout, dict) else [])
        edges_data = getattr(document_layout, "edges", document_layout.get("edges", []) if isinstance(document_layout, dict) else [])

        violations: list[str] = []
        reasons: list[str] = []
        detected_elements: list[str] = []

        header_valid = True
        sig_valid = True
        col_valid = True

        # Convert to token dicts
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
                    violations.append(f"Header '{tok.get('text', '')[:30]}' displaced to y={y0:.2f} (exceeds max y=0.25)")

            # 2. Signature Block Placement Check
            if ttype == "SIGNATURE_BLOCK":
                if y1 < 0.65:
                    sig_valid = False
                    violations.append(f"Signature block displaced to y={y1:.2f} (must be in bottom area y >= 0.65)")

        # 3. Column Ordering & Monotonicity Check across Reading-Order Edges
        tok_map = {t.get("token_id"): t for t in parsed_tokens}
        for u_id, v_id in edges_data:
            if u_id in tok_map and v_id in tok_map:
                u_y = tok_map[u_id].get("bbox", [0, 0, 1, 1])[1]
                v_y = tok_map[v_id].get("bbox", [0, 0, 1, 1])[1]
                # If reading order jumps upwards by more than 0.35 page height without column break
                if v_y < u_y - 0.35:
                    col_valid = False
                    violations.append(f"Inverted reading order flow between '{u_id}' (y={u_y:.2f}) and '{v_id}' (y={v_y:.2f})")

        # 4. Mandatory Rules Check
        for rule in self.rules:
            if rule.required:
                matching = [t for t in parsed_tokens if t.get("token_type") == rule.target_type]
                if not matching:
                    violations.append(f"Missing mandatory visual layout element '{rule.target_type}' ({rule.rule_name})")

        is_compliant = (len(violations) == 0) and header_valid and sig_valid and col_valid
        score = max(0.0, 1.0 - (len(violations) * 0.25))

        if is_compliant:
            reasons.append("Visual document layout fully satisfies agency rendering and column ordering specifications.")
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


class Navigator:
    def __init__(self, rule_store: RuleStore | None = None) -> None:
        self.rule_store = rule_store
        self.tools = ProgrammaticNavigatorTools()
        self.layout_cross_referencer = VisualLayoutCrossReferencer()

    def generate_prompt(self, target_programs: list[ProgramId], jurisdiction: str = "EX") -> str:
        """Generate prompt incorporating programmatic Python tool signatures strictly scoped to target programs."""
        progs = [p.value for p in target_programs]
        scoped_schemas = ""
        if self.rule_store is not None:
            schemas = [self.rule_store.get_scoped_schema(p, jurisdiction) for p in target_programs]
            scoped_schemas = f"\nScoped Program Schemas (Unselected domain schemas pruned): {schemas}\n"
        return (
            f"You are the navigator agent planning tasks for programs: {progs}.\n"
            f"{scoped_schemas}"
            "You have access to the following executable Python stubs:\n\n"
            f"{ProgrammaticNavigatorTools.get_tool_signatures()}\n"
            "Use these tools directly to decompose workflow plans into execution DAGs and cross-reference visual layouts."
        )

    def plan(self, case: SyntheticCase) -> DAG:
        """Dynamically bind and expose only DAG tasks for programs in the case intake payload."""
        dag = DAG()
        dag.add(Task(task_id="gather", kind="gather", subagent_id="subagent_gather"))
        for program in case.target_programs:
            dag.add(
                Task(
                    task_id=f"assess:{program.value}",
                    kind="assess",
                    deps=["gather"],
                    program=program,
                    subagent_id=f"subagent_{program.value}",
                )
            )
        return dag

    def verify_layout_compliance(
        self,
        document_layout: Any,
        template_name: str = "standard_notice",
    ) -> LayoutVerificationReport:
        """Cross-reference visual layout elements against agency rendering rules."""
        return self.layout_cross_referencer.verify_layout(document_layout, template_name)

    @staticmethod
    def target_programs(case: SyntheticCase) -> list[ProgramId]:
        return list(case.target_programs)


__all__ = [
    "AgencyLayoutRule",
    "LayoutVerificationReport",
    "VisualLayoutCrossReferencer",
    "ProgrammaticNavigatorTools",
    "Navigator",
]


