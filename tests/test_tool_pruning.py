"""Tests for Phase 3: Dynamic Skill Retrieval & Tool Pruning."""

from tribune.agents.eligibility import EligibilityProposer
from tribune.agents.navigator import Navigator
from tribune.corpus.rule_store import LocalRuleStore
from tribune.providers.local_rules import LocalRulesProvider
from tribune.types import ProgramId


def test_navigator_dynamic_rule_binding_and_pruning():
    store = LocalRuleStore()
    navigator = Navigator(rule_store=store)

    # Prompt generated for only SNAP program
    snap_prompt = navigator.generate_prompt(target_programs=[ProgramId.SNAP], jurisdiction="EX")
    assert "snap" in snap_prompt.lower()
    assert "medicaid" not in snap_prompt.lower()
    assert "housing" not in snap_prompt.lower()
    assert "unemployment" not in snap_prompt.lower()

    # Prompt generated for Medicaid program only
    medicaid_prompt = navigator.generate_prompt(target_programs=[ProgramId.MEDICAID], jurisdiction="EX")
    assert "medicaid" in medicaid_prompt.lower()
    assert "housing" not in medicaid_prompt.lower()
    assert "snap" not in medicaid_prompt.lower()


def test_eligibility_proposer_scoped_tools_and_schema_pruning():
    store = LocalRuleStore()
    provider = LocalRulesProvider(role="proposer")
    proposer = EligibilityProposer(provider=provider, rule_store=store)

    # Generate prompt for SNAP
    snap_prompt = proposer.generate_prompt(program=ProgramId.SNAP, jurisdiction="EX")
    assert "ProgrammaticSnapTools" in snap_prompt
    assert "calculate_snap_gross_income" in snap_prompt
    # Ensure unselected domain schemas are completely pruned
    assert "ProgrammaticMedicaidTools" not in snap_prompt
    assert "ProgrammaticHousingTools" not in snap_prompt
    assert "ProgrammaticUnemploymentTools" not in snap_prompt
    assert "ProgrammaticAppealsTools" not in snap_prompt

    # Generate prompt for Medicaid
    medicaid_prompt = proposer.generate_prompt(program=ProgramId.MEDICAID, jurisdiction="EX")
    assert "ProgrammaticMedicaidTools" in medicaid_prompt
    assert "evaluate_medicaid_magi_income" in medicaid_prompt
    # Ensure unselected domain schemas are pruned
    assert "ProgrammaticSnapTools" not in medicaid_prompt
    assert "ProgrammaticHousingTools" not in medicaid_prompt

    # Generate prompt for Housing
    housing_prompt = proposer.generate_prompt(program=ProgramId.HOUSING, jurisdiction="EX")
    assert "ProgrammaticHousingTools" in housing_prompt
    assert "evaluate_housing_ami_limit" in housing_prompt
    assert "ProgrammaticSnapTools" not in housing_prompt


def test_rule_store_scoped_schema():
    store = LocalRuleStore()
    schema = store.get_scoped_schema(ProgramId.SNAP, "EX")
    assert schema["program"] == "snap"
    assert schema["jurisdiction"] == "EX"
    assert len(schema["required_criteria"]) > 0
    assert "gross_income" in schema["required_criteria"]
    assert any(r["criterion_id"] == "gross_income" for r in schema["rules"])

