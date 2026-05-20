"""US per-building validation-set overview map.

Renders one map showing every county in COUNTIES.py with markers sized
by qualified-SFH count and coloured by climate zone. Intended for the
Validation section as a one-glance "where the per-building tests
happened" reference, replacing the older single-zone Marine map.

Sources used:
  * ``validation/us/COUNTIES.py``  — the canonical 6-zone list
    (any sensitivity additions are picked up automatically; pass
    ``--zones-only "Marine,Cold,..."`` to restrict).
  * ``validation/us/data/<county>_<zone>_qualified.csv`` —
    produced by ``buildings_select.py``; we read the per-building
    ``in.weather_file_latitude`` / ``in.weather_file_longitude``
    columns and reduce to one centroid + count per county.

Output:
  * ``img/us_validation_map.png`` and ``.pdf``

Dependencies:
  Cartopy is preferred for state outlines and shaded relief. If
  unavailable, the script falls back to a plain lat/lon scatter
  with manual axis bounds; install Cartopy with
  ``uv add cartopy`` (or ``pip install cartopy``) for the full map.

Usage:
    uv run python validation/us/validation_map.py
    uv run python validation/us/validation_map.py --no-counts
    uv run python validation/us/validation_map.py \\
        --out-fig img/validation_map_with_sensitivity.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES  # noqa: E402

mpl.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# Climate-zone palette. Order matters for legend readability —
# coldest → hottest top-to-bottom.
ZONE_ORDER = ["Very Cold", "Cold", "Marine", "Mixed-Humid",
              "Mixed-Dry", "Hot-Humid", "Hot-Dry"]
ZONE_COLOR = {
    "Very Cold":   "#2c3e8c",
    "Cold":        "#4a86c5",
    "Marine":      "#3aa17e",
    "Mixed-Humid": "#d4a73c",
    "Mixed-Dry":   "#b5995a",
    "Hot-Humid":   "#c0392b",
    "Hot-Dry":     "#8b3a3a",
}

# Paths to the county shapefile + IECC zone-per-county mapping used for
# the choropleth background. Both are generated/cached under
# validation/us/data/.
COUNTY_SHAPEFILE = (
    "validation/us/data/shapefiles/cb_2022_us_county_5m.shp"
)
COUNTY_ZONE_CSV = "validation/us/data/county_zone.csv"


def collect_county_centroids(data_dir: Path,
                              entries: list[tuple]) -> pd.DataFrame:
    """For each (county_id, state, zone, ...) entry, read the qualified
    CSV and reduce to one centroid + count row."""
    rows = []
    for cid, state, zone, _exp, label in entries:
        zone_safe = zone.replace(" ", "_")
        csv = data_dir / f"{cid}_{zone_safe}_qualified.csv"
        if not csv.exists():
            print(f"[validation_map] skip {cid} {zone}: {csv.name} missing")
            continue
        df = pd.read_csv(
            csv,
            usecols=["bldg_id",
                      "in.weather_file_latitude",
                      "in.weather_file_longitude"],
        )
        rows.append({
            "county_id": cid,
            "state": state,
            "zone": zone,
            "label": label,
            "n": len(df),
            "lat": float(df["in.weather_file_latitude"].mean()),
            "lon": float(df["in.weather_file_longitude"].mean()),
        })
    return pd.DataFrame(rows)


def _try_cartopy_axes(fig):
    """Return a Cartopy GeoAxes if Cartopy is installed, else None.

    Plain Cartopy import is gated behind a try block so the script
    still runs (with a degraded fallback map) if Cartopy is not
    available in the user's environment.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        return None, None, None
    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-124, -67, 24.5, 49.5], crs=ccrs.PlateCarree())
    return ax, ccrs, cfeature


def _add_basemap(ax, ccrs, cfeature) -> None:
    """Light shaded-relief basemap with state lines and coastlines."""
    ax.add_feature(cfeature.LAND, facecolor="#f5f1e8", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe2ec", edgecolor="none")
    ax.add_feature(cfeature.STATES, edgecolor="#888", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#555", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, edgecolor="#777", linewidth=0.6)


def _add_zone_choropleth(ax, ccrs) -> bool:
    """Light shaded county polygons coloured by IECC climate zone.

    Reads the county-FIPS shapefile (US Census 5m generalized) and
    the county_zone.csv lookup, joins them on 5-digit FIPS, and fills
    each county polygon with the zone's colour at low alpha so the
    overlaid scatter markers still pop. Returns True if the layer
    was drawn.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("[validation_map] geopandas not available; skipping zone choropleth")
        return False
    shp = ROOT / COUNTY_SHAPEFILE
    csv = ROOT / COUNTY_ZONE_CSV
    if not shp.exists() or not csv.exists():
        print(f"[validation_map] choropleth data missing ({shp.name} or "
              f"{csv.name}); skipping zone background")
        return False
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    gdf["fips"] = gdf["STATEFP"].astype(str) + gdf["COUNTYFP"].astype(str)
    zones = pd.read_csv(csv, dtype=str)
    zones["fips"] = zones["county_id"].str[1:3] + zones["county_id"].str[4:7]
    merged = gdf.merge(zones[["fips", "zone"]], on="fips", how="inner")
    # Limit to contiguous US (state FIPS <= 56, exclude 02 AK and 15 HI)
    merged = merged[~merged["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]
    for zone, group in merged.groupby("zone"):
        color = ZONE_COLOR.get(zone, "#888888")
        ax.add_geometries(
            group.geometry, crs=ccrs.PlateCarree(),
            facecolor=color, edgecolor="none", alpha=0.18, zorder=1,
        )
    return True



def _scatter_points(ax, centroids: pd.DataFrame, ccrs,
                     show_counts: bool = False) -> None:
    """Plot one uniformly-sized marker per county, coloured by climate
    zone, with a state-abbreviation label next to each marker."""
    transform = (ccrs.PlateCarree() if ccrs is not None else None)
    plot_kwargs = {"transform": transform} if transform is not None else {}

    # Uniform marker — county counts (all ~100) carry no useful signal.
    marker_size = 200.0

    for zone in ZONE_ORDER:
        sub = centroids[centroids["zone"] == zone]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"],
                    s=marker_size,
                    color=ZONE_COLOR[zone],
                    edgecolors="white", linewidths=1.2,
                    alpha=0.95, label=zone, zorder=5,
                    **plot_kwargs)
        for _, row in sub.iterrows():
            # State abbreviation right of each marker
            ax.text(row["lon"] + 1.2, row["lat"], row["state"],
                     fontsize=10, ha="left", va="center",
                     color="black", weight="bold", zorder=6,
                     **plot_kwargs)
    # Add zone-extra markers (sensitivity zones starting with "Marine-",
    # "Cold-" etc.) using the parent zone's colour but a lighter face.
    for _, row in centroids.iterrows():
        if row["zone"] in ZONE_ORDER:
            continue
        parent = next((z for z in ZONE_ORDER
                       if row["zone"].startswith(z + "-")), None)
        face = ZONE_COLOR.get(parent, "#888888")
        ax.scatter([row["lon"]], [row["lat"]],
                    s=sizes.loc[row.name] * 0.7,
                    facecolors="none", edgecolors=face,
                    linewidths=1.6, alpha=0.9, zorder=4,
                    **plot_kwargs)


def render(centroids: pd.DataFrame, out_path: Path,
            show_counts: bool = True,
            no_state_lines: bool = False) -> None:
    if centroids.empty:
        sys.exit("No counties to plot. Did you run buildings_select.py?")

    fig = plt.figure(figsize=(10, 5.5))
    ax, ccrs, cfeature = _try_cartopy_axes(fig)
    using_cartopy = ax is not None

    if not using_cartopy:
        print("[validation_map] Cartopy not available; falling back to "
              "plain lat/lon scatter. Install with 'uv add cartopy' "
              "for state outlines and shaded relief.")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-125, -67)
        ax.set_ylim(24, 50)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, ls=":", alpha=0.4)
    elif not no_state_lines:
        _add_basemap(ax, ccrs, cfeature)
        _add_zone_choropleth(ax, ccrs)

    _scatter_points(ax, centroids, ccrs, show_counts=show_counts)

    # Legend: include every canonical zone whose colour appears on the
    # figure — either as a validation marker or in the choropleth
    # backdrop. Hides zones that are entirely absent.
    in_choropleth = set()
    csv = ROOT / COUNTY_ZONE_CSV
    if csv.exists():
        in_choropleth = set(pd.read_csv(csv)["zone"].unique())
    legend_zones = [z for z in ZONE_ORDER
                    if (centroids["zone"] == z).any()
                    or any(centroids["zone"].str.startswith(z + "-"))
                    or z in in_choropleth]
    handles = [mpatches.Patch(color=ZONE_COLOR[z], label=z) for z in legend_zones]
    if handles:
        ax.legend(handles=handles, loc="lower left", framealpha=0.95,
                   title="Climate zone")

    # No in-figure title — the LaTeX caption carries that role.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200,
                 bbox_inches="tight" if using_cartopy else None)
    fig.savefig(out_path.with_suffix(".pdf"),
                 bbox_inches="tight" if using_cartopy else None)
    plt.close(fig)
    print(f"[validation_map] -> {out_path} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "validation" / "demand" / "data")
    p.add_argument("--out-fig", type=Path,
                   default=ROOT / "img" / "validation_map.png")
    p.add_argument("--zones-only", type=str, default="",
                   help="Comma-separated subset of zones to render "
                        "(default: all in COUNTIES.py).")
    p.add_argument("--no-counts", action="store_true",
                   help="Hide per-county building-count labels.")
    p.add_argument("--no-state-lines", action="store_true",
                   help="Skip state outlines (faster render, plain map).")
    args = p.parse_args()

    entries = COUNTIES
    if args.zones_only.strip():
        wanted = {z.strip() for z in args.zones_only.split(",")}
        entries = [e for e in COUNTIES if e[2] in wanted
                   or any(e[2].startswith(z + "-") for z in wanted)]
        if not entries:
            sys.exit(f"No counties match --zones-only={args.zones_only!r}")

    centroids = collect_county_centroids(args.data_dir, entries)
    print(centroids[["county_id", "state", "zone", "n",
                      "lat", "lon"]].to_string(index=False))

    render(centroids, args.out_fig,
           show_counts=not args.no_counts,
           no_state_lines=args.no_state_lines)


if __name__ == "__main__":
    main()
