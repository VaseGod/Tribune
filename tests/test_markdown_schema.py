"""Tests for standardized compact Markdown schemas and JSON adapters."""

import pytest
from tribune.casegen.markdown_schema import (
    convert_json_to_markdown,
    convert_markdown_to_json,
    parse_markdown_to_case,
    serialize_case_to_markdown,
)
from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.types import ApplicantSituation, ProgramId, RawDocument, SyntheticCase


def test_markdown_schema_round_trip():
    case = SyntheticCase(
        case_id="case_md_test_001",
        situation=ApplicantSituation(
            case_id="case_md_test_001",
            jurisdiction="EX",
            household_size=3,
            monthly_income=1250.50,
            liquid_assets=450.00,
            resident=True,
            citizenship_status="citizen",
            age=29,
            disabled=False,
            pregnant=False,
            has_dependent_child=True,
            employment_status="employed",
            monthly_rent=850.00,
        ),
        target_programs=[ProgramId.SNAP, ProgramId.MEDICAID],
        evidence=[],
        ground_truth={},
        documents=[
            RawDocument(
                doc_id="doc_intake_001",
                doc_type="application_intake",
                text="monthly_income: 1250.50\nhousehold_size: 3",
                fields={"monthly_income": "1250.50", "household_size": "3"},
            )
        ],
        jurisdiction="EX",
        language="en",
    )

    # 1. Serialize to Markdown
    md_text = serialize_case_to_markdown(case)
    assert "---" in md_text
    assert "case_id: case_md_test_001" in md_text
    assert "monthly_income: 1250.50" in md_text
    assert "household_size: 3" in md_text

    # 2. Parse back to case
    parsed_case = parse_markdown_to_case(md_text)
    assert parsed_case.case_id == case.case_id
    assert parsed_case.jurisdiction == "EX"
    assert parsed_case.situation.monthly_income == pytest.approx(1250.50)
    assert parsed_case.situation.household_size == 3
    assert parsed_case.situation.resident is True
    assert ProgramId.SNAP in parsed_case.target_programs
    assert len(parsed_case.documents) == 1
    assert parsed_case.documents[0].fields["monthly_income"] == "1250.50"


def test_convenience_methods_on_synthetic_case():
    generator = SyntheticCaseGenerator(seed=42)
    demo_cases = generator.generate_demo_set()
    base_case = demo_cases[0]

    # Test .to_markdown() and .from_markdown()
    md_text = base_case.to_markdown()
    assert isinstance(md_text, str)
    assert "Applicant Situation" in md_text

    reconstructed = SyntheticCase.from_markdown(md_text)
    assert reconstructed.case_id == base_case.case_id
    assert reconstructed.situation.household_size == base_case.situation.household_size


def test_json_to_markdown_and_back_adapter():
    json_data = {
        "case_id": "case_json_conv",
        "jurisdiction": "EX",
        "target_programs": ["snap"],
        "situation": {
            "household_size": 2,
            "monthly_income": 900.0,
            "liquid_assets": 100.0,
            "resident": True,
            "citizenship_status": "citizen",
            "age": 40,
            "disabled": False,
            "pregnant": False,
            "has_dependent_child": False,
            "employment_status": "unemployed",
            "monthly_rent": 600.0,
        },
        "documents": [],
    }

    # Convert JSON to Markdown
    md = convert_json_to_markdown(json_data)
    assert "case_id: case_json_conv" in md
    assert "monthly_income: 900.00" in md

    # Convert Markdown to JSON
    json_back = convert_markdown_to_json(md)
    assert json_back["case_id"] == "case_json_conv"
    assert json_back["situation"]["monthly_income"] == pytest.approx(900.0)
