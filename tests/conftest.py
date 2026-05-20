"""Shared fixtures for the offline test suite.

All fixtures synthesise tiny inputs in pytest's ``tmp_path`` so the suite
runs offline (no Open-Meteo, no NREL EULP, no HTW Berlin profiles).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Integration opt-in ────────────────────────────────────────────────────
#
# Integration tests (under tests/integration/) exercise the heavy / network
# providers — TEASER + TABULA-DE, demandlib, Open-Meteo — plus full YAML
# dispatch round-trips. They're slower and a few hit the network, so they
# are skipped by default and opt-in via `pytest --run-integration`.

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (TEASER, demandlib, Open-Meteo, YAML dispatch)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: heavier / network tests skipped unless --run-integration is given",
    )
    config.addinivalue_line(
        "markers",
        "network: integration test that requires outbound HTTP (e.g. Open-Meteo)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# ── Time index ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def year() -> int:
    return 2010


@pytest.fixture(scope="session")
def hourly_index(year: int) -> pd.DatetimeIndex:
    """Full-year hourly UTC index — same shape the pipeline expects."""
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h", tz="UTC")


# ── Archetypes ────────────────────────────────────────────────────────────

@pytest.fixture
def archetype_csv_path(tmp_path: Path) -> Path:
    """A 1-archetype CSV that satisfies ArchetypeSchema.REQUIRED_1R1C."""
    df = pd.DataFrame([{
        "archetype_id": 1,
        "construction_year": 1995,
        "area_m2": 120.0,
        "n_floors": 2,
        "height_floor_m": 2.5,
        "thermal_resistance": 0.005,    # K/W (typical SFH order of magnitude)
        "thermal_capacitance": 5.0e7,   # J/K
        "window_area_total_m2": 20.0,   # split N/E/S/W at sim time
    }])
    p = tmp_path / "archetypes.csv"
    df.to_csv(p, index=False)
    return p


# ── Electricity ───────────────────────────────────────────────────────────

@pytest.fixture
def electricity_dir(tmp_path: Path, hourly_index: pd.DatetimeIndex) -> Path:
    """Directory with two HTW-style one-column CSVs."""
    rng = np.random.default_rng(seed=42)
    out_dir = tmp_path / "electricity"
    out_dir.mkdir()
    n = len(hourly_index)
    for stem, mean_w in (("3000", 300.0), ("4500", 450.0)):
        # Simple diurnal-ish synthetic profile so GeoMA has signal to chew on.
        hours = hourly_index.hour.values
        diurnal = 1.0 + 0.6 * np.sin(2 * np.pi * (hours - 6) / 24)
        noise = rng.normal(0, 0.05, n)
        power_w = mean_w * np.maximum(diurnal + noise, 0.05)
        df = pd.DataFrame({"power_W": power_w}, index=hourly_index)
        df.index.name = "timestamp"
        df.to_csv(out_dir / f"{stem}.csv")
    return out_dir


@pytest.fixture
def electricity_long_parquet(tmp_path: Path, hourly_index: pd.DatetimeIndex) -> Path:
    """Pre-built long-form electricity parquet matching ElectricitySchema."""
    rng = np.random.default_rng(seed=7)
    frames = []
    for pid in (1, 2):
        frames.append(pd.DataFrame({
            "timestamp": hourly_index,
            "profile_id": pid,
            "electricity_demand": rng.uniform(100, 500, len(hourly_index)),
        }))
    df = pd.concat(frames, ignore_index=True)
    p = tmp_path / "E.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def occupancy_long_parquet(tmp_path: Path, hourly_index: pd.DatetimeIndex) -> Path:
    """Pre-built long-form occupancy parquet matching OccupancySchema."""
    rng = np.random.default_rng(seed=3)
    frames = []
    for pid in (1, 2):
        frames.append(pd.DataFrame({
            "timestamp": hourly_index,
            "profile_id": pid,
            "occupied": rng.integers(0, 2, len(hourly_index)),
        }))
    df = pd.concat(frames, ignore_index=True)
    p = tmp_path / "O.parquet"
    df.to_parquet(p, index=False)
    return p


# ── Weather ───────────────────────────────────────────────────────────────

@pytest.fixture
def weather_setup(tmp_path: Path, hourly_index: pd.DatetimeIndex) -> dict[str, Path]:
    """Two location parquets + a location_mapping.csv. Drop-in for CSVWeatherProvider."""
    cache_dir = tmp_path / "weather"
    cache_dir.mkdir()

    locations = pd.DataFrame([
        {"location_id": 1, "latitude": 52.52, "longitude": 13.40},  # Berlin
        {"location_id": 2, "latitude": 48.13, "longitude": 11.58},  # Munich
    ])
    locations_csv = tmp_path / "location_mapping.csv"
    locations.to_csv(locations_csv, index=False)

    n = len(hourly_index)
    rng = np.random.default_rng(seed=1)
    for _, row in locations.iterrows():
        loc_id = int(row["location_id"])
        ghi = np.maximum(
            300.0 * np.sin(2 * np.pi * hourly_index.hour.values / 24), 0.0
        )
        df = pd.DataFrame({
            "timestamp": hourly_index,
            "location_id": loc_id,
            "air_temperature": 10.0 + 8.0 * np.sin(2 * np.pi * np.arange(n) / 8760),
            "global_horizontal_irradiance": ghi,
            # Synthetic beam/diffuse split: ~70% beam at midday under the
            # diurnal sine, with the remainder treated as diffuse. Exact
            # values don't matter for the offline tests — only schema.
            "direct_normal_irradiance": 0.7 * ghi,
            "diffuse_horizontal_irradiance": 0.3 * ghi,
            "wind_speed": rng.uniform(1.0, 5.0, n),
            "relative_humidity": rng.uniform(40.0, 90.0, n),
        })
        df.to_parquet(cache_dir / f"loc{loc_id:04d}.parquet", index=False)

    return {"cache_dir": cache_dir, "locations_csv": locations_csv}
