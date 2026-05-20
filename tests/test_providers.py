"""Smoke tests for the four provider families.

Verifies (a) Protocol compliance via runtime_checkable isinstance and
(b) that each shipped CSV/parquet provider returns a DataFrame matching
the canonical schema in ``src/providers/base.py``.

TEASERArchetypeProvider, OpenMeteoWeatherProvider, DemandlibElectricity-
Provider, and PyLPGElectricityProvider are not exercised — they require
network access or heavy optional packages and are out of scope for the
offline test suite.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.providers import (
    ArchetypeProvider,
    ArchetypeSchema,
    CSVArchetypeProvider,  # legacy alias under test
    CSVElectricityProvider,  # legacy alias under test
    CSVWeatherProvider,  # legacy alias under test
    DirectoryElectricityProvider,
    ElectricityProvider,
    ElectricitySchema,
    FileArchetypeProvider,
    FileOccupancyProvider,
    FileWeatherProvider,
    GeoMAOccupancyProvider,
    OccupancyProvider,
    OccupancySchema,
    ParquetElectricityProvider,
    ParquetOccupancyProvider,  # legacy alias under test
    WeatherProvider,
    WeatherSchema,
)

# ── Protocol compliance ───────────────────────────────────────────────────

def test_file_archetype_satisfies_protocol(archetype_csv_path: Path):
    p = FileArchetypeProvider(path=archetype_csv_path)
    assert isinstance(p, ArchetypeProvider)


def test_directory_electricity_satisfies_protocol(electricity_dir: Path):
    p = DirectoryElectricityProvider(input_dir=electricity_dir, resolution="1h")
    assert isinstance(p, ElectricityProvider)


def test_parquet_electricity_satisfies_protocol(electricity_long_parquet: Path):
    p = ParquetElectricityProvider(path=electricity_long_parquet)
    assert isinstance(p, ElectricityProvider)


def test_geoma_occupancy_satisfies_protocol():
    p = GeoMAOccupancyProvider(alpha=0.05)
    assert isinstance(p, OccupancyProvider)


def test_file_occupancy_satisfies_protocol(tmp_path: Path):
    # Build a trivial occupancy parquet so the provider has something to read.
    df = pd.DataFrame({
        "timestamp": pd.date_range("2010-01-01", periods=24, freq="1h"),
        "profile_id": 1,
        "occupied": [0, 1] * 12,
    })
    pq = tmp_path / "O.parquet"
    df.to_parquet(pq, index=False)
    p = FileOccupancyProvider(path=pq)
    assert isinstance(p, OccupancyProvider)


def test_file_weather_satisfies_protocol(weather_setup: dict[str, Path]):
    p = FileWeatherProvider(
        cache_dir=weather_setup["cache_dir"],
        locations_csv=weather_setup["locations_csv"],
    )
    assert isinstance(p, WeatherProvider)


# ── Schema return ─────────────────────────────────────────────────────────

def test_file_archetype_returns_canonical_schema(archetype_csv_path: Path):
    df = FileArchetypeProvider(path=archetype_csv_path).get_archetypes()
    for col in ArchetypeSchema.REQUIRED_1R1C:
        assert col in df.columns, f"missing required archetype column: {col}"
    assert len(df) >= 1


def test_directory_electricity_returns_canonical_schema(electricity_dir: Path):
    df = DirectoryElectricityProvider(input_dir=electricity_dir, resolution="1h").get_profiles(year=2010)
    for col in ElectricitySchema.REQUIRED:
        assert col in df.columns, f"missing required electricity column: {col}"
    assert df["profile_id"].nunique() == 2


def test_parquet_electricity_returns_canonical_schema(electricity_long_parquet: Path):
    df = ParquetElectricityProvider(path=electricity_long_parquet).get_profiles(year=2010)
    for col in ElectricitySchema.REQUIRED:
        assert col in df.columns
    assert df["profile_id"].nunique() == 2


def test_geoma_occupancy_returns_canonical_schema(electricity_long_parquet: Path):
    elec = pd.read_parquet(electricity_long_parquet)
    df = GeoMAOccupancyProvider(alpha=0.05).get_occupancy(elec)
    for col in OccupancySchema.REQUIRED:
        assert col in df.columns
    # Occupancy must be binary 0/1.
    assert set(df["occupied"].unique()).issubset({0, 1})


def test_file_weather_returns_canonical_schema(weather_setup: dict[str, Path]):
    provider = FileWeatherProvider(
        cache_dir=weather_setup["cache_dir"],
        locations_csv=weather_setup["locations_csv"],
    )
    locations = provider.list_locations()
    assert {"location_id", "latitude", "longitude"}.issubset(locations.columns)
    assert len(locations) == 2

    df = provider.get_weather(location_id=1, year=2010)
    for col in WeatherSchema.REQUIRED:
        assert col in df.columns


def test_file_archetype_rejects_missing_columns(tmp_path: Path):
    """The base validate_schema helper must trip on an incomplete CSV."""
    bad = tmp_path / "bad_archetypes.csv"
    pd.DataFrame([{"archetype_id": 1}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        FileArchetypeProvider(path=bad).get_archetypes()


# ── Legacy aliases still resolve ──────────────────────────────────────────

def test_legacy_aliases_point_at_new_classes():
    """Power users importing the old names must still get the right classes."""
    assert CSVArchetypeProvider is FileArchetypeProvider
    assert CSVElectricityProvider is DirectoryElectricityProvider
    assert ParquetOccupancyProvider is FileOccupancyProvider
    assert CSVWeatherProvider is FileWeatherProvider
