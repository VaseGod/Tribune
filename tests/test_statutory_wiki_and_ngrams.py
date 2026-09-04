"""Tests for Decoupled Statutory Memory, Persistent Knowledge Wiki, and Token-Local N-Gram Lookups.

Validates:
1. StatutoryWiki deductions, BBCE rules, UI monetary tables, and procedural timelines.
2. Casegen SNAP and Unemployment labelers querying external statutory wiki memory.
3. TokenLocalNGramLookup hash table and text interception with token savings.
4. query_filtered_program_rules backed by FilterableHNSWIndex.
"""

from __future__ import annotations

import pytest

from tribune.casegen.programs.snap import ground_truth as snap_ground_truth
from tribune.casegen.programs.unemployment import ground_truth as ui_ground_truth
from tribune.corpus.programs.jurisdictions import JurisdictionProfile
from tribune.corpus.wiki import StatutoryWiki, get_statutory_wiki
from tribune.providers.local_rules import (
    TokenLocalNGramLookup,
    intercept_statutory_ngrams,
    query_filtered_program_rules,
)
from tribune.types import ApplicantSituation, GroundTruthLabel, ProgramId


@pytest.fixture
def wiki() -> StatutoryWiki:
    return get_statutory_wiki()


@pytest.fixture
def ex_profile() -> JurisdictionProfile:
    return JurisdictionProfile(
        code="EX",
        name="Example State",
        medicaid_expansion=True,
    )


def test_statutory_wiki_snap_deductions(wiki: StatutoryWiki):
    d1 = wiki.lookup_snap_deductions(household_size=1)
    assert d1["standard_deduction"] == 198.0
    assert d1["earned_income_deduction_pct"] == 0.20
    assert d1["excess_shelter_cap"] == 672.0
    assert "heating_cooling_sua" in d1["sua_table"]

    d4 = wiki.lookup_snap_deductions(household_size=4)
    assert d4["standard_deduction"] == 208.0


def test_statutory_wiki_bbce_rules(wiki: StatutoryWiki):
    bbce_ex = wiki.lookup_snap_bbce("EX")
    assert bbce_ex.bbce_active is True
    assert bbce_ex.waives_asset_test is True
    assert bbce_ex.gross_income_fpl_pct == 1.30

    bbce_nx = wiki.lookup_snap_bbce("NX")
    assert bbce_nx.bbce_active is False
    assert bbce_nx.waives_asset_test is False


def test_statutory_wiki_ui_monetary(wiki: StatutoryWiki):
    ui_ex = wiki.lookup_unemployment_monetary("EX")
    assert ui_ex.min_base_period_earnings == 5000.0
    assert ui_ex.min_weeks_worked == 1

    ui_nx = wiki.lookup_unemployment_monetary("NX")
    assert ui_nx.min_base_period_earnings == 4500.0
    assert ui_nx.min_weeks_worked == 2


def test_statutory_wiki_procedural_statutes(wiki: StatutoryWiki):
    proc = wiki.lookup_procedural_statute(ProgramId.UNEMPLOYMENT, "EX")
    assert proc.appeal_filing_window_days == 90
    assert "unsafe_working_conditions" in proc.good_cause_reasons


def test_snap_casegen_queries_external_wiki(ex_profile: JurisdictionProfile):
    sit = ApplicantSituation(
        case_id="case_snap_test_1",
        jurisdiction="EX",
        household_size=3,
        monthly_income=1200.0,
        liquid_assets=500.0,
        resident=True,
        citizenship_status="citizen",
        age=35,
        disabled=False,
        pregnant=False,
        has_dependent_child=True,
        employment_status="employed",
    )
    gt = snap_ground_truth(sit, ex_profile)
    assert gt.label == GroundTruthLabel.ELIGIBLE
    # Ensure rationale was loaded dynamically from StatutoryWiki
    assert "SNAP turns on" in gt.rationale
    assert "130%" in gt.rationale


def test_unemployment_casegen_queries_external_wiki(ex_profile: JurisdictionProfile):
    sit = ApplicantSituation(
        case_id="case_ui_test_1",
        jurisdiction="EX",
        household_size=1,
        monthly_income=0.0,
        liquid_assets=1000.0,
        resident=True,
        citizenship_status="citizen",
        age=40,
        disabled=False,
        pregnant=False,
        has_dependent_child=False,
        employment_status="unemployed",
        base_period_earnings=8000.0,
        weeks_worked=15,
        separation_reason="laid_off",
        able_and_available=True,
    )
    gt = ui_ground_truth(sit, ex_profile)
    assert gt.label == GroundTruthLabel.ELIGIBLE
    assert "Unemployment turns on" in gt.rationale


def test_token_local_ngram_lookup():
    ngram_lookup = TokenLocalNGramLookup()
    text = (
        "The applicant met the gross income eligibility standard and broad-based "
        "categorical eligibility applies under 7 CFR 273.2. Separation was quit with good cause."
    )
    res = ngram_lookup.intercept_text(text)
    assert res["intercepted_count"] >= 2
    assert res["tokens_saved"] > 0
    matched_phrases = {item["phrase"] for item in res["matched_ngrams"]}
    assert "gross income eligibility standard" in matched_phrases or "quit with good cause" in matched_phrases


def test_intercept_statutory_ngrams_speculative_tool():
    res = intercept_statutory_ngrams("Claimant has base period earnings and is able and available.")
    assert res["intercepted_count"] >= 1
    assert res["tokens_saved"] > 0


def test_query_filtered_program_rules_hnsw():
    res = query_filtered_program_rules(
        query="income limits and poverty guidelines",
        program="snap",
        jurisdiction="EX",
        k=3,
        required_only=True,
    )
    assert res["program"] == "snap"
    assert len(res["rules"]) > 0
    # Rule returned should have criterion_id, citation_id, score
    first = res["rules"][0]
    assert "criterion_id" in first
    assert "score" in first
