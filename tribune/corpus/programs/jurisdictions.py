"""Jurisdiction profiles: the parameters every program rule set reads.

Two example jurisdictions ship so the Medicaid abstention machinery has something
real to stress against:

* ``EX`` — an example Medicaid-*expansion* state.
* ``NX`` — an example *non-expansion* state, where childless adults below the
  poverty line fall into the coverage gap (the canonical abstain-and-escalate
  case).

The numbers below are illustrative and modeled on 2024 federal figures (e.g. the
HHS poverty guidelines for the 48 contiguous states). They are NOT a substitute
for the current, official figures of any real jurisdiction. See LIMITATIONS.md.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from ...types import StrictModel

# HUD-style household-size adjustment factors applied to a 4-person AMI.
_AMI_FACTORS: dict[int, float] = {
    1: 0.70,
    2: 0.80,
    3: 0.90,
    4: 1.00,
    5: 1.08,
    6: 1.16,
    7: 1.24,
    8: 1.32,
}


class JurisdictionProfile(StrictModel):
    """Tunable thresholds for one jurisdiction. Frozen and strict by inheritance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    name: str
    medicaid_expansion: bool

    # Federal Poverty Level (annual), 48 contiguous states basis.
    fpl_base_annual: float = 15060.0
    fpl_increment_annual: float = 5380.0

    # SNAP
    snap_gross_income_pct: float = 1.30
    snap_net_income_pct: float = 1.00
    snap_asset_limit: float = 2750.0
    snap_bbce_waives_assets: bool = True

    # Medicaid (as a fraction of FPL)
    medicaid_magi_adult_pct: float = 1.38
    medicaid_parent_pct_nonexpansion: float = 0.40
    medicaid_pregnant_pct: float = 2.13
    medicaid_child_pct: float = 2.55

    # Housing Choice Voucher / public housing
    ami_4person: float = 100000.0
    ami_factors: dict[int, float] = Field(default_factory=lambda: dict(_AMI_FACTORS))
    housing_very_low_pct: float = 0.50
    housing_low_pct: float = 0.80

    # Unemployment insurance (example state code values)
    ui_min_base_period_earnings: float = 5000.0
    ui_min_weeks_worked: int = 1

    # Appeals
    appeal_window_days: int = 90

    def fpl_annual(self, household_size: int) -> float:
        hh = max(1, household_size)
        return self.fpl_base_annual + (hh - 1) * self.fpl_increment_annual

    def fpl_monthly(self, household_size: int) -> float:
        return self.fpl_annual(household_size) / 12.0

    def _ami_factor(self, household_size: int) -> float:
        if household_size in self.ami_factors:
            return self.ami_factors[household_size]
        # Beyond the table, HUD adds 8% of the 4-person AMI per additional person.
        return 1.0 + 0.08 * (household_size - 4)

    def ami_limit_annual(self, household_size: int, pct: float) -> float:
        return self.ami_4person * self._ami_factor(household_size) * pct


_PROFILES: dict[str, JurisdictionProfile] = {
    "EX": JurisdictionProfile(code="EX", name="Example (expansion state)", medicaid_expansion=True),
    "NX": JurisdictionProfile(
        code="NX",
        name="Example (non-expansion state)",
        medicaid_expansion=False,
    ),
}


def get_profile(code: str) -> JurisdictionProfile:
    if code not in _PROFILES:
        raise KeyError(f"unknown jurisdiction '{code}'; known: {sorted(_PROFILES)}")
    return _PROFILES[code]


def known_jurisdictions() -> list[str]:
    return sorted(_PROFILES)
