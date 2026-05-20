"""Diagnostic helper. Not part of the published pipeline.

Find alternative counties in a given climate zone for sensitivity checks.

Generalises find_marine_counties.py. Reads metadata.parquet, filters to:
  - in.geometry_building_type_recs == "Single-Family Detached"
  - in.hvac_cooling_type != "None"
  - in.heating_fuel != "None"
  - in.building_america_climate_zone == <zone>

Then groups by county and prints the top counties by qualified-SFH count.
The top three (excluding the county already in COUNTIES for that zone)
are recommended as sensitivity-test additions.

Usage:
    uv run python validation/us/find_zone_counties.py --zone Cold
    uv run python validation/us/find_zone_counties.py --zone "Mixed-Humid"
    uv run python validation/us/find_zone_counties.py --zone "Hot-Dry"

Recognised zones (Building America climate zones in EULP):
    Marine, Cold, Mixed-Humid, Mixed-Dry, Very Cold, Hot-Humid, Hot-Dry,
    Subarctic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "validation" / "demand" / "data" / "metadata.parquet"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True,
                   help="Building America climate zone, e.g. 'Cold', "
                        "'Marine', 'Mixed-Humid', 'Hot-Dry'.")
    p.add_argument("--top", type=int, default=15,
                   help="Number of top counties to display (default 15).")
    p.add_argument("--n-suggest", type=int, default=3,
                   help="Number of sensitivity-test additions to recommend.")
    args = p.parse_args()

    if not META.exists():
        sys.exit(f"{META} not found. Run select_buildings.py first to "
                 "download the metadata cache.")

    cols = [
        "in.geometry_building_type_recs",
        "in.hvac_cooling_type",
        "in.heating_fuel",
        "in.building_america_climate_zone",
        "in.county", "in.state",
    ]
    df = pd.read_parquet(META, columns=cols).reset_index()

    # Identify counties already in COUNTIES for this zone (to mark them
    # in the output). Match on the canonical zone label OR the
    # short-form variants we use in sensitivity entries (e.g. Marine vs
    # Marine-WA2). The match logic is "starts with the zone string".
    existing_in_zone = {
        cid for (cid, _state, z, *_rest) in COUNTIES
        if z == args.zone or z.startswith(args.zone + "-")
    }

    qual = df[
        (df["in.geometry_building_type_recs"] == "Single-Family Detached")
        & (df["in.hvac_cooling_type"] != "None")
        & (df["in.heating_fuel"] != "None")
        & (df["in.building_america_climate_zone"] == args.zone)
    ].copy()

    if qual.empty:
        sys.exit(f"No qualified buildings found for zone={args.zone!r}. "
                 "Check spelling — try one of: "
                 "Marine, Cold, Mixed-Humid, Mixed-Dry, Very Cold, "
                 "Hot-Humid, Hot-Dry, Subarctic.")

    by_county = (qual.groupby(["in.county", "in.state"])
                       .size()
                       .reset_index(name="n_qualified")
                       .sort_values("n_qualified", ascending=False))

    print(f"\nZone {args.zone!r}: qualified-SFH counts per county "
          f"(top {args.top}):")
    print(f"{'rank':>4}  {'county_id':<10} {'state':<3} {'n_qualified':>11}")
    print("-" * 40)
    for rank, (_, row) in enumerate(by_county.head(args.top).iterrows(), start=1):
        marker = "  <-- already in COUNTIES" if row["in.county"] in existing_in_zone else ""
        print(f"{rank:>4}  {row['in.county']:<10} {row['in.state']:<3} "
              f"{row['n_qualified']:>11}{marker}")

    print()
    print(f"Suggested {args.n_suggest} sensitivity-test additions "
          f"(highest counts not already in COUNTIES):")
    candidates = (by_county[~by_county["in.county"].isin(existing_in_zone)]
                  .head(args.n_suggest))
    safe = args.zone.replace(" ", "_")
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        # Build a unique zone label like 'Cold-MI2' so the figure
        # renderer doesn't collide on dict keys.
        unique_label = f"{safe}-{row['in.state']}{i + 1}"
        print(f"  ({row['in.county']!r}, {row['in.state']!r}, "
              f"{unique_label!r}, 100, "
              f"'{row['in.state']} #{i + 1} ({args.zone})'),")


if __name__ == "__main__":
    main()
