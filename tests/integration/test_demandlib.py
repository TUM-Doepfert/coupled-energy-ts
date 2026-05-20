"""Integration test: DemandlibElectricityProvider synthesises one BDEW H0 profile.

Exercises EnTiSe + demandlib on a single annual demand. demandlib bundles
the BDEW standard load profile shapes; no network is required.
"""
from __future__ import annotations

import pytest

from src.providers import (
    DemandlibElectricityProvider,
    ElectricityProvider,
    ElectricitySchema,
)

pytestmark = pytest.mark.integration


def test_demandlib_produces_one_profile():
    """One annual demand → one profile with the canonical schema."""
    provider = DemandlibElectricityProvider(
        annual_demands_kwh=[4500.0],
        profile_type="h0",
        freq="1h",
    )
    assert isinstance(provider, ElectricityProvider)

    df = provider.get_profiles(year=2010)
    for col in ElectricitySchema.REQUIRED:
        assert col in df.columns
    assert df["profile_id"].nunique() == 1
    # Full year hourly = 8760 rows. Allow a small slack for DST handling.
    assert 8700 <= len(df) <= 8800, len(df)
    # BDEW H0 is non-negative and bounded for 4500 kWh/year (peak ~1.5 kW).
    power = df["electricity_demand"]
    assert (power >= 0).all()
    assert power.max() < 5000.0, power.max()
