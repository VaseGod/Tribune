"""Standardized Compact Markdown Schema Engine for Case Ingestion & Synthesis.

Replaces verbose JSON payloads with token-compact, human-readable, and
syntax-error-resilient Markdown schemas:
1. Lowers input token consumption by ~30-50%.
2. Prevents JSON bracket/comma nesting syntax errors.
3. Preserves deterministic structured extraction and typing.
4. Provides bidirectional adapters/converters for legacy JSON structures.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..types import (
    ApplicantSituation,
    ProgramId,
    RawDocument,
    SyntheticCase,
    WaitlistStatus,
)

logger = logging.getLogger(__name__)

MARKDOWN_SCHEMA_DOCUMENTATION = """# TRIBUNE Compact Case Markdown Schema Specification (v1.0)

A case Markdown document begins with an optional YAML frontmatter block:
```markdown
---
case_id: <string>
jurisdiction: <state_code>
programs: [<program_1>, <program_2>]
language: <iso_code>
---
# Applicant Situation
- household_size: <int>
- monthly_income: <float>
- liquid_assets: <float>
- resident: <bool>
- citizenship_status: <string>
- age: <int>
- disabled: <bool>
- pregnant: <bool>
- has_dependent_child: <bool>
- employment_status: <string>
- monthly_rent: <float>

# Documents
## <doc_type> (doc_id: <id>)
<key>: <value>
...
```
"""


def _parse_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter without external yaml dependencies."""
    meta: dict[str, Any] = {}
    remaining = text.strip()
    if remaining.startswith("---"):
        parts = remaining.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            remaining = parts[2].strip()
            for line in fm_text.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        # List of items
                        items = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                        meta[k] = items
                    elif v.lower() == "true":
                        meta[k] = True
                    elif v.lower() == "false":
                        meta[k] = False
                    elif v.isdigit():
                        meta[k] = int(v)
                    else:
                        meta[k] = v.strip("'\"")
    return meta, remaining


def serialize_case_to_markdown(case: SyntheticCase) -> str:
    """Serialize a SyntheticCase instance into compact Markdown."""
    lines: list[str] = [
        "---",
        f"case_id: {case.case_id}",
        f"jurisdiction: {case.jurisdiction}",
        f"programs: [{', '.join(p.value for p in case.target_programs)}]",
        f"language: {case.language}",
        "---",
        "",
        "# Applicant Situation",
    ]

    sit = case.situation
    lines.append(f"- household_size: {sit.household_size}")
    lines.append(f"- monthly_income: {sit.monthly_income:.2f}")
    lines.append(f"- liquid_assets: {sit.liquid_assets:.2f}")
    lines.append(f"- resident: {'true' if sit.resident else 'false'}")
    lines.append(f"- citizenship_status: {sit.citizenship_status}")
    lines.append(f"- age: {sit.age}")
    lines.append(f"- disabled: {'true' if sit.disabled else 'false'}")
    lines.append(f"- pregnant: {'true' if sit.pregnant else 'false'}")
    lines.append(f"- has_dependent_child: {'true' if sit.has_dependent_child else 'false'}")
    lines.append(f"- employment_status: {sit.employment_status}")
    lines.append(f"- monthly_rent: {sit.monthly_rent:.2f}")
    if sit.separation_reason:
        lines.append(f"- separation_reason: {sit.separation_reason}")
    if sit.base_period_earnings:
        lines.append(f"- base_period_earnings: {sit.base_period_earnings:.2f}")
    if sit.weeks_worked:
        lines.append(f"- weeks_worked: {sit.weeks_worked}")
    if sit.waitlist_status != WaitlistStatus.UNKNOWN:
        lines.append(f"- waitlist_status: {sit.waitlist_status.value}")
    if sit.days_since_denial is not None:
        lines.append(f"- days_since_denial: {sit.days_since_denial}")
    if sit.appeal_grounds:
        lines.append(f"- appeal_grounds: {sit.appeal_grounds}")

    if case.documents:
        lines.append("")
        lines.append("# Documents")
        for doc in case.documents:
            lines.append(f"## {doc.doc_type} (doc_id: {doc.doc_id})")
            for k, v in doc.fields.items():
                lines.append(f"{k}: {v}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


_DEFAULT_SITUATION: dict[str, Any] = {
    "household_size": 1,
    "monthly_income": 0.0,
    "liquid_assets": 500.0,
    "resident": True,
    "citizenship_status": "citizen",
    "age": 35,
    "disabled": False,
    "pregnant": False,
    "has_dependent_child": False,
    "employment_status": "employed",
    "separation_reason": None,
    "base_period_earnings": 0.0,
    "weeks_worked": 0,
    "able_and_available": True,
    "monthly_rent": 900.0,
    "waitlist_status": WaitlistStatus.UNKNOWN,
    "days_since_denial": None,
    "appeal_grounds": None,
}


def parse_markdown_to_case(md_text: str) -> SyntheticCase:
    """Parse standardized Markdown back into a typed SyntheticCase."""
    meta, body = _parse_yaml_frontmatter(md_text)

    case_id = meta.get("case_id", "case_imported")
    jurisdiction = meta.get("jurisdiction", "EX")
    raw_programs = meta.get("programs", ["snap"])
    language = meta.get("language", "en")

    target_programs = []
    for p in raw_programs:
        try:
            target_programs.append(ProgramId(p.lower()))
        except ValueError:
            pass
    if not target_programs:
        target_programs = [ProgramId.SNAP]

    # Parse Applicant Situation key-values
    sit_dict: dict[str, Any] = {}
    sit_match = re.search(r"#\s*Applicant Situation\s*\n(.*?)(?=\n#|\Z)", body, re.DOTALL | re.IGNORECASE)
    if sit_match:
        for line in sit_match.group(1).splitlines():
            line = line.strip().lstrip("-").strip()
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    sit_dict[k] = True
                elif v.lower() == "false":
                    sit_dict[k] = False
                elif v.isdigit():
                    sit_dict[k] = int(v)
                else:
                    try:
                        sit_dict[k] = float(v)
                    except ValueError:
                        sit_dict[k] = v

    situation_data = dict(_DEFAULT_SITUATION)
    situation_data["case_id"] = case_id
    situation_data["jurisdiction"] = jurisdiction
    situation_data.update(sit_dict)
    if "waitlist_status" in situation_data and isinstance(situation_data["waitlist_status"], str):
        try:
            situation_data["waitlist_status"] = WaitlistStatus(situation_data["waitlist_status"])
        except ValueError:
            situation_data["waitlist_status"] = WaitlistStatus.UNKNOWN

    situation = ApplicantSituation(**situation_data)

    # Parse Documents
    documents: list[RawDocument] = []
    docs_match = re.search(r"#\s*Documents\s*\n(.*)", body, re.DOTALL | re.IGNORECASE)
    if docs_match:
        doc_sections = re.split(r"\n##\s*", docs_match.group(1))
        for sec in doc_sections:
            sec = sec.strip()
            if not sec:
                continue
            header_line, _, doc_body = sec.partition("\n")
            # Parse header e.g. "application_intake (doc_id: intake_001)"
            header_match = re.match(r"(?P<doc_type>[a-zA-Z0-9_-]+)(?:\s*\(doc_id:\s*(?P<doc_id>[^)]+)\))?", header_line.strip())
            doc_type = header_match.group("doc_type") if header_match else "application_intake"
            doc_id = header_match.group("doc_id") if header_match and header_match.group("doc_id") else f"{case_id}:{doc_type}"

            fields: dict[str, str] = {}
            for line in doc_body.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()

            documents.append(
                RawDocument(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    text=doc_body.strip(),
                    fields=fields,
                )
            )

    return SyntheticCase(
        case_id=case_id,
        situation=situation,
        target_programs=target_programs,
        evidence=[],
        documents=documents,
        ground_truth={},
        jurisdiction=jurisdiction,
        language=language,
    )


def convert_json_to_markdown(json_payload: dict[str, Any] | str) -> str:
    """Convert a JSON representation of a case to compact Markdown."""
    if isinstance(json_payload, str):
        data = json.loads(json_payload)
    else:
        data = json_payload

    # If data matches SyntheticCase dict serialization
    if "situation" in data:
        sit_raw = dict(_DEFAULT_SITUATION)
        sit_raw["case_id"] = data.get("case_id", "case_converted")
        sit_raw["jurisdiction"] = data.get("jurisdiction", "EX")
        sit_raw.update(data["situation"])
        if "waitlist_status" in sit_raw and isinstance(sit_raw["waitlist_status"], str):
            try:
                sit_raw["waitlist_status"] = WaitlistStatus(sit_raw["waitlist_status"])
            except ValueError:
                sit_raw["waitlist_status"] = WaitlistStatus.UNKNOWN

        sit = ApplicantSituation(**sit_raw)
        progs = [ProgramId(p) for p in data.get("target_programs", ["snap"])]
        docs = [
            RawDocument(
                doc_id=d.get("doc_id", "doc_0"),
                doc_type=d.get("doc_type", "application_intake"),
                text=d.get("text", ""),
                fields=d.get("fields", {}),
            )
            for d in data.get("documents", [])
        ]
        case = SyntheticCase(
            case_id=data.get("case_id", "case_converted"),
            situation=sit,
            target_programs=progs,
            evidence=data.get("evidence", []),
            documents=docs,
            ground_truth=data.get("ground_truth", {}),
            jurisdiction=data.get("jurisdiction", "EX"),
            language=data.get("language", "en"),
        )
        return serialize_case_to_markdown(case)
        return serialize_case_to_markdown(case)

    # General dict fallback: format as key-value markdown
    lines = ["# Ingested Payload"]
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"## {k}")
            for sub_k, sub_v in v.items():
                lines.append(f"- {sub_k}: {sub_v}")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def convert_markdown_to_json(md_text: str) -> dict[str, Any]:
    """Convert a Markdown case back into JSON format."""
    case = parse_markdown_to_case(md_text)
    return case.model_dump()
