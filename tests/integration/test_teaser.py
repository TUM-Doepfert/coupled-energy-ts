"""Integration test: TEASERArchetypeProvider builds one archetype.

Exercises the TEASER + TABULA-DE wrapper end-to-end on a single synthetic
archetype-input row. No network: TABULA-DE construction data ships with
the ``teaser`` package.

This is the only test that verifies the headline "archetypes are computed
from a national typology" path; the offline suite uses ``FileArchetype-
Provider`` instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.providers import ArchetypeProvider, ArchetypeSchema, TEASERArchetypeProvider

pytestmark = pytest.mark.integration


def test_teaser_builds_one_archetype(tmp_path: Path):
    """A 1-row archetype-input CSV produces a 1-row archetype DataFrame."""
    csv = tmp_path / "archetypes.csv"
    pd.DataFrame([{
        "archetype_id": 1,
        "construction_year": 1995,
        "area_m2": 120.0,
        "n_floors": 2,
        "height_floor_m": 2.5,
    }]).to_csv(csv, index=False)

    provider = TEASERArchetypeProvider(
        archetypes_csv=csv,
        construction_data="tabula_de_standard",
        geometry_data="tabula_de_single_family_house",
    )
    assert isinstance(provider, ArchetypeProvider)

    df = provider.get_archetypes()
    for col in ArchetypeSchema.REQUIRED_1R1C:
        assert col in df.columns, f"TEASER output missing {col}"
    assert len(df) == 1
    row = df.iloc[0]
    # Sanity bounds — single-family detached, 1995 construction:
    # R ~ 1e-4..1e-2 K/W, C ~ 1e7..1e9 J/K. Wide bounds so the test
    # doesn't break if TEASER tweaks its TABULA factors.
    assert 1e-5 < row["thermal_resistance"] < 1e-1, row["thermal_resistance"]
    assert 1e6 < row["thermal_capacitance"] < 1e10, row["thermal_capacitance"]
