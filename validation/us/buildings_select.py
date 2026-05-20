"""Step 1: select qualified buildings per county, write per-county metadata CSVs.

For each of the 6 counties in COUNTIES, filter the global ResStock metadata
to Single-Family Detached buildings with cooling and heating. Output one CSV
per county (small, reviewable) listing the qualified bldg_ids and key
attributes used by downstream scripts.

Run from the project root:
    uv run python validation/us/buildings_select.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES, METADATA_URL


METADATA_COLUMNS_KEEP = [
    # Identity / climate
    "in.county", "in.state", "in.building_america_climate_zone",
    # Building type & geometry
    "in.geometry_building_type_recs", "in.geometry_floor_area",
    "in.geometry_stories", "in.geometry_wall_type",
    "in.geometry_attic_type", "in.geometry_foundation_type",
    "in.window_areas", "in.windows",
    # HVAC
    "in.hvac_cooling_type", "in.hvac_cooling_efficiency",
    "in.hvac_heating_type", "in.hvac_heating_type_and_fuel",
    "in.hvac_heating_efficiency", "in.heating_fuel",
    # Setpoints + setback schedules (needed for US per-building)
    "in.heating_setpoint", "in.heating_setpoint_has_offset",
    "in.heating_setpoint_offset_magnitude", "in.heating_setpoint_offset_period",
    "in.cooling_setpoint", "in.cooling_setpoint_has_offset",
    "in.cooling_setpoint_offset_magnitude", "in.cooling_setpoint_offset_period",
    # Envelope properties
    "in.infiltration",
    "in.insulation_wall", "in.insulation_ceiling", "in.insulation_floor",
    "in.insulation_foundation_wall",
    # Weather location (per-building, used for solar gain calcs)
    "in.weather_file_city",
    "in.weather_file_latitude", "in.weather_file_longitude",
    # Vintage / size / occupancy
    "in.vintage", "in.sqft", "in.occupants",
]


def download_metadata(out_path: Path) -> None:
    if out_path.exists():
        print(f"Metadata already at {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {METADATA_URL}")
    urllib.request.urlretrieve(METADATA_URL, out_path)
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({sz_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path,
        default=Path("validation/us/data/metadata.parquet"))
    parser.add_argument("--out-dir", type=Path,
        default=Path("validation/us/data"))
    parser.add_argument("--metadata-fallback", type=Path, default=None,
        help="Optional path to a local metadata.parquet (if download fails).")
    args = parser.parse_args()

    if args.metadata_fallback and args.metadata_fallback.exists() and not args.metadata.exists():
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(args.metadata_fallback, args.metadata)
        print(f"Copied metadata from {args.metadata_fallback}")
    if not args.metadata.exists():
        download_metadata(args.metadata)

    print(f"Reading {args.metadata}")
    cols = METADATA_COLUMNS_KEEP
    md = pd.read_parquet(args.metadata, columns=cols)
    md["bldg_id"] = md.index.astype(int)
    print(f"  {len(md):,} total ResStock buildings")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for county_id, state, zone, expected_n, label in COUNTIES:
        sub = md[md["in.county"] == county_id]
        sfh = sub[
            (sub["in.geometry_building_type_recs"] == "Single-Family Detached")
            & (sub["in.hvac_cooling_type"] != "None")
            & (sub["in.heating_fuel"] != "None")
        ].copy()
        # Cap to expected_n with a deterministic seeded sample if the
        # county has substantially more qualified buildings than the
        # cohort target. Keeps download volume bounded for sensitivity-
        # test counties that are otherwise much larger than the
        # canonical 6 (e.g. G5300330 with 842 qualified SFH).
        if len(sfh) > expected_n + 10:
            sfh = sfh.sample(n=expected_n, random_state=0).sort_index()
        zone_safe = zone.replace(" ", "_")
        out_path = args.out_dir / f"{county_id}_{zone_safe}_qualified.csv"
        sfh.to_csv(out_path, index=False)
        actual_n = len(sfh)
        flag = "OK" if abs(actual_n - expected_n) <= 2 else f"WARN: expected {expected_n}"
        print(f"  {county_id:9s} {state:3s} {zone:14s}: {actual_n:4d} qualified SFH  ({flag}) -> {out_path.name}")
        summary_rows.append({
            "county_id": county_id, "state": state, "climate_zone": zone,
            "label": label, "n_qualified": actual_n,
            "csv_path": str(out_path),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "selection_summary.csv", index=False)
    total = summary["n_qualified"].sum()
    print(f"\nTotal qualified SFH across {len(COUNTIES)} counties: {total}")
    print(f"Selection summary: {args.out_dir / 'selection_summary.csv'}")


if __name__ == "__main__":
    main()
