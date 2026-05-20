"""End-to-end smoke test of the orchestrator.

Drives ``src.pipeline.run`` on a synthetic 2-location, 1-archetype,
2-profile setup using the file-based providers (CSV archetype, CSV
electricity, GeoMA occupancy, CSV weather). The simulation step is
skipped because it depends on EnTiSe + Open-Meteo-shaped solar inputs
that the offline suite cannot easily synthesise; the goal here is to
verify orchestration, schema compliance, and idempotency wiring.

The test asserts that B.parquet, E.parquet, O.parquet, and
location_mapping.csv are produced with the canonical schema and the
expected row counts.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.pipeline import run as run_pipeline
from src.providers import ArchetypeSchema, ElectricitySchema, OccupancySchema


def _write_config(
    tmp_path: Path,
    archetype_csv: Path,
    electricity_dir: Path,
    weather_setup: dict[str, Path],
) -> Path:
    output_dir = tmp_path / "output"
    cfg = {
        "year": 2010,
        "output_dir": str(output_dir),
        "thermal_model": "r1c1",
        "locations": "all",
        "archetype_provider": {
            "type": "file",
            "path": str(archetype_csv),
        },
        "electricity_provider": {
            "type": "directory",
            "input_dir": str(electricity_dir),
            "resolution": "1h",
        },
        "occupancy_provider": {
            "type": "geoma",
            "alpha": 0.05,
        },
        "weather_provider": {
            "type": "file",
            "weather_dir": str(weather_setup["cache_dir"]),
            "location_mapping": str(weather_setup["locations_csv"]),
        },
        "simulation": {
            "heating_setpoint_C": 20.0,
            "cooling_setpoint_C": 26.0,
            "inhabitants": 2,
            "gains_per_person_W": 80,
            "ach_model": "rule_based",
            "resolution": "60min",
            "overwrite": False,
        },
        "n_jobs": 1,
    }
    p = tmp_path / "synthetic.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_pipeline_smoke_run(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    weather_setup: dict[str, Path],
):
    config_path = _write_config(tmp_path, archetype_csv_path, electricity_dir, weather_setup)

    result = run_pipeline(config_path, skip_simulation=True)

    output_dir = Path(result.output_dir)

    # 1. Archetypes
    b_path = output_dir / "B.parquet"
    assert b_path.exists()
    b_df = pd.read_parquet(b_path)
    for col in ArchetypeSchema.REQUIRED_1R1C:
        assert col in b_df.columns
    assert len(b_df) == 1

    # 2. Electricity (2 profiles × 8760 hours = 17520, allow ±1 for DST edge)
    e_path = output_dir / "E.parquet"
    assert e_path.exists()
    e_df = pd.read_parquet(e_path)
    for col in ElectricitySchema.REQUIRED:
        assert col in e_df.columns
    assert e_df["profile_id"].nunique() == 2
    assert 17500 <= len(e_df) <= 17540

    # 3. Occupancy — same shape as electricity, schema compliant.
    o_path = output_dir / "O.parquet"
    assert o_path.exists()
    o_df = pd.read_parquet(o_path)
    for col in OccupancySchema.REQUIRED:
        assert col in o_df.columns
    assert o_df["profile_id"].nunique() == 2
    assert set(o_df["occupied"].unique()).issubset({0, 1})

    # 4. Weather mapping written with both locations.
    mapping_path = output_dir / "location_mapping.csv"
    assert mapping_path.exists()
    mapping = pd.read_csv(mapping_path)
    assert {"location_id", "latitude", "longitude"}.issubset(mapping.columns)
    assert len(mapping) == 2


def test_pipeline_idempotency(
    tmp_path: Path,
    archetype_csv_path: Path,
    electricity_dir: Path,
    weather_setup: dict[str, Path],
):
    """Running twice with the same outputs should short-circuit B/E/O."""
    config_path = _write_config(tmp_path, archetype_csv_path, electricity_dir, weather_setup)

    first = run_pipeline(config_path, skip_simulation=True)
    second = run_pipeline(config_path, skip_simulation=True)

    by_name_first = {s.name: s for s in first.steps}
    by_name_second = {s.name: s for s in second.steps}

    for step in ("archetypes", "electricity", "occupancy"):
        assert by_name_first[step].skipped is False
        assert by_name_second[step].skipped is True
