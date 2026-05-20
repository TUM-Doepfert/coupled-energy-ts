"""Integration tests for YAML provider-combination dispatch.

Each test writes a synthetic YAML config and runs the orchestrator with
``skip_simulation=True``. Together they cover the main combinations the
README and config docs advertise:

  - new key names: file / directory / geoma / file
  - long-form electricity parquet:           file / parquet / geoma / file
  - pre-computed occupancy:                  file / directory / file / file
  - legacy aliases:                          csv  / csv       / parquet / csv

The fixtures live in the parent ``tests/conftest.py`` (synthetic
archetypes, electricity directory + parquet, occupancy parquet, weather
setup). All tests are offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline import run as run_pipeline
from src.providers import (
    ArchetypeSchema,
    ElectricitySchema,
    OccupancySchema,
)

pytestmark = pytest.mark.integration


def _write_config(tmp_path: Path, cfg: dict) -> Path:
    out = tmp_path / "config.yml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "year": 2010,
        "output_dir": str(tmp_path / "output"),
        "thermal_model": "r1c1",
        "locations": "all",
        "simulation": {
            "heating_setpoint_C": 20.0, "cooling_setpoint_C": 26.0,
            "inhabitants": 2, "gains_per_person_W": 65,
            "resolution": "60min", "overwrite": False,
        },
        "n_jobs": 1,
    }


def _assert_schema(output_dir: Path):
    b = pd.read_parquet(output_dir / "B.parquet")
    for col in ArchetypeSchema.REQUIRED_1R1C:
        assert col in b.columns

    e = pd.read_parquet(output_dir / "E.parquet")
    for col in ElectricitySchema.REQUIRED:
        assert col in e.columns
    assert e["profile_id"].nunique() == 2

    o = pd.read_parquet(output_dir / "O.parquet")
    for col in OccupancySchema.REQUIRED:
        assert col in o.columns
    assert set(o["occupied"].unique()).issubset({0, 1})

    mapping = pd.read_csv(output_dir / "location_mapping.csv")
    assert len(mapping) == 2


# ── New key names ─────────────────────────────────────────────────────────

def test_file_directory_geoma_file(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    weather_setup: dict[str, Path],
):
    """Canonical synthetic combo: file archetype + directory of CSVs + GeoMA + file weather."""
    cfg = _base_cfg(tmp_path) | {
        "archetype_provider": {"type": "file", "path": str(archetype_csv_path)},
        "electricity_provider": {
            "type": "directory", "input_dir": str(electricity_dir), "resolution": "1h",
        },
        "occupancy_provider": {"type": "geoma", "alpha": 0.05},
        "weather_provider": {
            "type": "file",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
    }
    run_pipeline(_write_config(tmp_path, cfg), skip_simulation=True)
    _assert_schema(Path(cfg["output_dir"]))


def test_long_form_parquet_electricity(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_long_parquet: Path,
    weather_setup: dict[str, Path],
):
    """ParquetElectricityProvider as the electricity source."""
    cfg = _base_cfg(tmp_path) | {
        "archetype_provider": {"type": "file", "path": str(archetype_csv_path)},
        "electricity_provider": {"type": "parquet", "path": str(electricity_long_parquet)},
        "occupancy_provider": {"type": "geoma", "alpha": 0.05},
        "weather_provider": {
            "type": "file",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
    }
    run_pipeline(_write_config(tmp_path, cfg), skip_simulation=True)
    _assert_schema(Path(cfg["output_dir"]))


def test_precomputed_file_occupancy(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    occupancy_long_parquet: Path,
    weather_setup: dict[str, Path],
):
    """FileOccupancyProvider reads a pre-computed occupancy parquet (skips GeoMA)."""
    cfg = _base_cfg(tmp_path) | {
        "archetype_provider": {"type": "file", "path": str(archetype_csv_path)},
        "electricity_provider": {
            "type": "directory", "input_dir": str(electricity_dir), "resolution": "1h",
        },
        "occupancy_provider": {"type": "file", "path": str(occupancy_long_parquet)},
        "weather_provider": {
            "type": "file",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
    }
    run_pipeline(_write_config(tmp_path, cfg), skip_simulation=True)
    _assert_schema(Path(cfg["output_dir"]))


# ── Legacy aliases must keep working ──────────────────────────────────────

def test_legacy_csv_parquet_aliases(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    occupancy_long_parquet: Path,
    weather_setup: dict[str, Path],
):
    """Pre-rename YAML — `csv` + `csv` + `parquet` + `csv` — must still dispatch."""
    cfg = _base_cfg(tmp_path) | {
        "archetype_provider": {"type": "csv", "path": str(archetype_csv_path)},
        "electricity_provider": {
            "type": "csv", "input_dir": str(electricity_dir), "resolution": "1h",
        },
        "occupancy_provider": {"type": "parquet", "path": str(occupancy_long_parquet)},
        "weather_provider": {
            "type": "csv",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
    }
    run_pipeline(_write_config(tmp_path, cfg), skip_simulation=True)
    _assert_schema(Path(cfg["output_dir"]))


# ── Error surface ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("section,bad_type,error_hint", [
    ("archetype_provider",  "nope", "archetype_provider.type"),
    ("electricity_provider", "csv_directory", "electricity_provider.type"),
    ("occupancy_provider",  "yaml",   "occupancy_provider.type"),
    ("weather_provider",    "parquet", "weather_provider.type"),
])
def test_unknown_type_raises_clear_error(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    weather_setup: dict[str, Path],
    section: str,
    bad_type: str,
    error_hint: str,
):
    """Unknown provider types must raise a ValueError naming the section."""
    cfg = _base_cfg(tmp_path) | {
        "archetype_provider": {"type": "file", "path": str(archetype_csv_path)},
        "electricity_provider": {
            "type": "directory", "input_dir": str(electricity_dir), "resolution": "1h",
        },
        "occupancy_provider": {"type": "geoma", "alpha": 0.05},
        "weather_provider": {
            "type": "file",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
    }
    cfg[section] = {"type": bad_type}
    with pytest.raises(ValueError, match=error_hint):
        run_pipeline(_write_config(tmp_path, cfg), skip_simulation=True)
