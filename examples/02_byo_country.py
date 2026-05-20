"""Bring-your-own-country example: build the dataset from your own files.

Demonstrates the *file-only* path through the pipeline. Useful when:

  - You have national archetypes from another source (EUReCA, EnergyPlus,
    hand-curated literature values) — no TEASER/TABULA needed.
  - You have already-measured electricity profiles in canonical form.
  - You have weather files (TMY, ERA5 extracts, station obs) you want
    to feed in instead of fetching from Open-Meteo.

This script writes a tiny synthetic dataset to ``examples/_byo_demo/``
and runs the pipeline end-to-end (B / E / O steps; weather and sim are
skipped because the demo doesn't need them). It is the smallest end-to-
end example you can run with no network, no TEASER, and no TABULA.

Run with:
    uv run python examples/02_byo_country.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

# Make `src.*` importable when running from the examples/ folder.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.resolve()))

from src.pipeline import run  # noqa: E402


ROOT = HERE / "_byo_demo"


def _make_archetypes_parquet(path: Path) -> None:
    """Two SFH archetypes with hand-curated 1R1C parameters."""
    df = pd.DataFrame(
        {
            "archetype_id": [1, 2],
            "construction_year": [1980, 2015],
            "area_m2": [120.0, 150.0],
            "n_floors": [2, 2],
            "height_floor_m": [2.5, 2.6],
            # Pre-1990 envelope vs. modern construction (illustrative).
            "thermal_resistance": [0.0030, 0.0080],
            "thermal_capacitance": [3.0e7, 5.0e7],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_electricity_parquet(path: Path, year: int = 2024) -> None:
    """Two synthetic profiles: a workday-shaped and a flat household."""
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    t = idx.hour + idx.minute / 60.0

    base_a = 200 + 600 * np.exp(-((t - 19) ** 2) / 4)  # evening peak
    base_b = np.full_like(base_a, 350.0)               # flat
    noise = rng.normal(0, 30, len(idx))

    df = pd.concat(
        [
            pd.DataFrame({"timestamp": idx, "profile_id": 1, "electricity_demand": base_a + noise}),
            pd.DataFrame({"timestamp": idx, "profile_id": 2, "electricity_demand": base_b + noise}),
        ],
        ignore_index=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_config(path: Path, archetypes_path: Path, electricity_path: Path,
                  output_dir: Path, year: int = 2024) -> None:
    """Tiny YAML pointing at the synthetic files. CSV providers everywhere
    so this example needs neither TEASER nor Open-Meteo.
    """
    cfg = {
        "year": year,
        "output_dir": str(output_dir),
        "thermal_model": "r1c1",
        "locations": "all",
        "archetype_provider": {
            "type": "csv",
            "path": str(archetypes_path),
        },
        "electricity_provider": {
            "type": "parquet",
            "path": str(electricity_path),
        },
        "occupancy_provider": {
            "type": "geoma",
            "alpha": 0.05,
        },
        # No real weather provider; the example skips weather fetch.
        "weather_provider": {
            "type": "csv",
            "weather_dir": str(output_dir / "weather"),  # unused
            "location_mapping": str(output_dir / "location_mapping.csv"),
        },
        "simulation": {
            "heating_setpoint_C": 20.0,
            "cooling_setpoint_C": 26.0,
            "inhabitants": 2,
            "gains_per_person_W": 65,
            "resolution": "60min",
            "overwrite": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _make_locations_csv(path: Path) -> None:
    """One dummy location so the location_mapping step has something to write."""
    pd.DataFrame(
        {"location_id": [1], "latitude": [50.0], "longitude": [10.0]}
    ).to_csv(path, index=False)


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    archetypes = ROOT / "input" / "archetypes.parquet"
    electricity = ROOT / "input" / "E.parquet"
    config = ROOT / "config.yml"
    output_dir = ROOT / "output"
    locations_csv = output_dir / "location_mapping.csv"

    _make_archetypes_parquet(archetypes)
    _make_electricity_parquet(electricity)
    output_dir.mkdir(parents=True, exist_ok=True)
    _make_locations_csv(locations_csv)
    _write_config(config, archetypes, electricity, output_dir)

    print(f"Demo files written to {ROOT}")
    print(f"Running pipeline (B/E/O only — weather + sim skipped)…")
    result = run(
        config,
        overwrite=False,
        skip_weather_fetch=True,
        skip_simulation=True,
    )
    print(result)
    print()
    print(f"Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
