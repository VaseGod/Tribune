"""Persistent Statutory Knowledge Repository (Statutory Wiki).

Externalizes statutory tables, income deduction schedules, categorical eligibility limits,
unemployment monetary formulas, and procedural deadlines into a structured, queryable knowledge wiki.
Decouples hardcoded statutory parameters from case generation and adjudication rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import EvidenceType, ProgramId


@dataclass(frozen=True)
class SNAPDeductionSchedule:
    """SNAP income deduction schedule under 7 CFR 273.9(d)."""

    standard_deduction: dict[int, float]  # household size -> monthly standard deduction
    earned_income_deduction_pct: float  # e.g., 0.20 (20%)
    excess_shelter_cap_non_elderly: float  # standard shelter cap (e.g., $672)
    standard_utility_allowances: dict[str, float]  # heating_cooling, basic, telephone
    homeless_shelter_deduction: float


@dataclass(frozen=True)
class SNAPBBCERules:
    """SNAP Broad-Based Categorical Eligibility (BBCE) rules under 7 CFR 273.2(j)."""

    jurisdiction: str
    bbce_active: bool
    gross_income_fpl_pct: float  # e.g., 1.30 (130%) or up to 2.00 (200%)
    waives_asset_test: bool
    standard_asset_limit: float
    elderly_disabled_asset_limit: float


@dataclass(frozen=True)
class UnemploymentMonetaryRules:
    """Unemployment Insurance monetary eligibility thresholds under state statute."""

    jurisdiction: str
    min_base_period_earnings: float
    min_weeks_worked: int
    max_weekly_benefit_amount: float
    high_quarter_fraction: float  # e.g. 1/26 of high quarter wages


@dataclass(frozen=True)
class ProceduralStatute:
    """Procedural filing windows, notice periods, and good-cause definitions."""

    program: ProgramId
    jurisdiction: str
    initial_filing_window_days: int
    appeal_filing_window_days: int
    good_cause_reasons: tuple[str, ...]
    citation: str


class StatutoryWiki:
    """Persistent, sovereign knowledge wiki for statutory rules, tables, and procedural laws."""

    def __init__(self) -> None:
        # SNAP standard deduction schedule (FY 2026 baseline under 7 CFR 273.9(d))
        self._snap_deductions = SNAPDeductionSchedule(
            standard_deduction={
                1: 198.0,
                2: 198.0,
                3: 198.0,
                4: 208.0,
                5: 244.0,
                6: 279.0,
                7: 279.0,
                8: 279.0,
            },
            earned_income_deduction_pct=0.20,
            excess_shelter_cap_non_elderly=672.0,
            standard_utility_allowances={
                "heating_cooling_sua": 537.0,
                "limited_utility_allowance": 382.0,
                "telephone_allowance": 54.0,
            },
            homeless_shelter_deduction=179.66,
        )

        # SNAP BBCE rules by jurisdiction
        self._snap_bbce: dict[str, SNAPBBCERules] = {
            "EX": SNAPBBCERules(
                jurisdiction="EX",
                bbce_active=True,
                gross_income_fpl_pct=1.30,
                waives_asset_test=True,
                standard_asset_limit=3000.0,
                elderly_disabled_asset_limit=4500.0,
            ),
            "NX": SNAPBBCERules(
                jurisdiction="NX",
                bbce_active=False,
                gross_income_fpl_pct=1.30,
                waives_asset_test=False,
                standard_asset_limit=3000.0,
                elderly_disabled_asset_limit=4500.0,
            ),
        }

        # UI monetary rules by jurisdiction
        self._ui_monetary: dict[str, UnemploymentMonetaryRules] = {
            "EX": UnemploymentMonetaryRules(
                jurisdiction="EX",
                min_base_period_earnings=5000.0,
                min_weeks_worked=1,
                max_weekly_benefit_amount=600.0,
                high_quarter_fraction=0.04,  # 1/25th or 1/26th
            ),
            "NX": UnemploymentMonetaryRules(
                jurisdiction="NX",
                min_base_period_earnings=4500.0,
                min_weeks_worked=2,
                max_weekly_benefit_amount=450.0,
                high_quarter_fraction=0.038,
            ),
        }

        # Procedural statutes
        self._procedural: dict[tuple[ProgramId, str], ProceduralStatute] = {
            (ProgramId.SNAP, "EX"): ProceduralStatute(
                program=ProgramId.SNAP,
                jurisdiction="EX",
                initial_filing_window_days=30,
                appeal_filing_window_days=90,
                good_cause_reasons=(
                    "medical_emergency",
                    "disaster_displacement",
                    "agency_misinformation",
                ),
                citation="7 CFR 273.13 / 273.15",
            ),
            (ProgramId.SNAP, "NX"): ProceduralStatute(
                program=ProgramId.SNAP,
                jurisdiction="NX",
                initial_filing_window_days=30,
                appeal_filing_window_days=60,
                good_cause_reasons=(
                    "medical_emergency",
                    "disaster_displacement",
                ),
                citation="7 CFR 273.13 / 273.15",
            ),
            (ProgramId.UNEMPLOYMENT, "EX"): ProceduralStatute(
                program=ProgramId.UNEMPLOYMENT,
                jurisdiction="EX",
                initial_filing_window_days=14,
                appeal_filing_window_days=90,
                good_cause_reasons=(
                    "unsafe_working_conditions",
                    "domestic_violence_relocation",
                    "substantial_wage_reduction",
                    "illness_of_immediate_family",
                ),
                citation="EX Unemployment Insurance Code §1256 / §1328",
            ),
            (ProgramId.UNEMPLOYMENT, "NX"): ProceduralStatute(
                program=ProgramId.UNEMPLOYMENT,
                jurisdiction="NX",
                initial_filing_window_days=14,
                appeal_filing_window_days=60,
                good_cause_reasons=(
                    "unsafe_working_conditions",
                    "substantial_wage_reduction",
                ),
                citation="NX Unemployment Insurance Code §1256 / §1328",
            ),
            (ProgramId.APPEALS, "EX"): ProceduralStatute(
                program=ProgramId.APPEALS,
                jurisdiction="EX",
                initial_filing_window_days=90,
                appeal_filing_window_days=90,
                good_cause_reasons=(
                    "illness_hospitalization",
                    "lack_of_timely_notice",
                    "postal_failure",
                ),
                citation="EX Administrative Procedure Act §11500",
            ),
            (ProgramId.APPEALS, "NX"): ProceduralStatute(
                program=ProgramId.APPEALS,
                jurisdiction="NX",
                initial_filing_window_days=60,
                appeal_filing_window_days=60,
                good_cause_reasons=(
                    "illness_hospitalization",
                    "lack_of_timely_notice",
                ),
                citation="NX Administrative Procedure Act §11500",
            ),
        }

        # Canonical rationales decoupled from code
        self._rationales: dict[ProgramId, str] = {
            ProgramId.SNAP: (
                "SNAP turns on residency, citizenship/immigration status, and gross monthly "
                "income vs. 130% of the poverty guideline (assets often waived under BBCE)."
            ),
            ProgramId.UNEMPLOYMENT: (
                "Unemployment turns on monetary eligibility (base-period earnings/weeks), "
                "a non-disqualifying separation, and being able and available; a quit 'with "
                "good cause' is fact-intensive and ambiguous."
            ),
            ProgramId.MEDICAID: (
                "Medicaid eligibility turns on citizenship, residency, and modified adjusted gross income (MAGI)."
            ),
            ProgramId.HOUSING: (
                "Housing voucher assistance turns on income limits (Area Median Income), citizenship, and local residency."
            ),
            ProgramId.APPEALS: (
                "Appeals eligibility turns on timeliness (filing within statutory window) and valid grounds of error."
            ),
        }

        # Canonical statutory definitions for n-gram boilerplate & legal accuracy
        self._statutory_definitions: dict[str, str] = {
            "bbce": "Broad-Based Categorical Eligibility under 7 CFR 273.2(j) which harmonizes asset tests with TANF-funded services.",
            "gross_income_standard": "Gross income eligibility standard equal to 130 percent of the federal poverty income guidelines under 7 CFR 273.9(a)(1).",
            "net_income_standard": "Net income eligibility standard equal to 100 percent of the federal poverty income guidelines under 7 CFR 273.9(a)(2).",
            "base_period": "The first four of the last five completed calendar quarters prior to the filing of the claim for unemployment benefits.",
            "quit_good_cause": "A voluntary separation from employment with real, substantial, and compelling cause of such nature as would cause a reasonable worker to quit.",
            "able_and_available": "Physically and mentally capable of performing suitable work and actively seeking work without unreasonable restrictions.",
            "standard_deduction": "Statutory monthly basic deduction subtracted from gross income pursuant to 7 CFR 273.9(d)(1).",
            "earned_income_deduction": "Twenty percent deduction applied to total earned income pursuant to 7 CFR 273.9(d)(2).",
        }

    def lookup_snap_deductions(self, household_size: int = 1) -> dict[str, Any]:
        """Lookup SNAP deduction schedule for given household size."""
        hh = max(1, household_size)
        std_ded = self._snap_deductions.standard_deduction.get(
            hh, self._snap_deductions.standard_deduction[8]
        )
        return {
            "household_size": hh,
            "standard_deduction": std_ded,
            "earned_income_deduction_pct": self._snap_deductions.earned_income_deduction_pct,
            "excess_shelter_cap": self._snap_deductions.excess_shelter_cap_non_elderly,
            "sua_table": dict(self._snap_deductions.standard_utility_allowances),
            "homeless_shelter_deduction": self._snap_deductions.homeless_shelter_deduction,
        }

    def lookup_snap_bbce(self, jurisdiction: str = "EX") -> SNAPBBCERules:
        """Lookup Broad-Based Categorical Eligibility configuration for jurisdiction."""
        return self._snap_bbce.get(jurisdiction, self._snap_bbce["EX"])

    def lookup_unemployment_monetary(self, jurisdiction: str = "EX") -> UnemploymentMonetaryRules:
        """Lookup Unemployment Insurance monetary thresholds for jurisdiction."""
        return self._ui_monetary.get(jurisdiction, self._ui_monetary["EX"])

    def lookup_procedural_statute(
        self, program: ProgramId, jurisdiction: str = "EX"
    ) -> ProceduralStatute:
        """Lookup procedural deadlines, filing windows, and good-cause definitions."""
        key = (program, jurisdiction)
        if key in self._procedural:
            return self._procedural[key]
        return self._procedural.get((program, "EX")) or ProceduralStatute(
            program=program,
            jurisdiction=jurisdiction,
            initial_filing_window_days=30,
            appeal_filing_window_days=60,
            good_cause_reasons=("extenuating_circumstances",),
            citation="Statutory Procedural Code",
        )

    def get_program_rationale(self, program: ProgramId | str, jurisdiction: str = "EX") -> str:
        """Retrieve canonical rationale string for program."""
        if isinstance(program, str):
            try:
                prog_enum = ProgramId(program.lower().strip())
            except ValueError:
                prog_enum = ProgramId.SNAP
        else:
            prog_enum = program
        return self._rationales.get(
            prog_enum, f"{prog_enum.value.upper()} statutory eligibility assessment."
        )

    def lookup_definition(self, key: str) -> str | None:
        """Lookup canonical legal definition for a key."""
        return self._statutory_definitions.get(key.lower().strip())

    def all_definitions(self) -> dict[str, str]:
        """Return all legal definitions in wiki."""
        return dict(self._statutory_definitions)


_GLOBAL_WIKI = StatutoryWiki()


def get_statutory_wiki() -> StatutoryWiki:
    """Return the global sovereign statutory knowledge wiki instance."""
    return _GLOBAL_WIKI
