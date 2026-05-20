"""Render paper Figure 3: BKG 10 km grid centroids covering Germany.

Loads the official 10 km BKG grid from
``input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg``, computes the centroid of
each cell, and renders a static map showing the grid polygons with
their centroids on top of an OpenStreetMap basemap.

Three outputs are written to ``img/``:

- ``cells_centroids_map.png`` (raster, for paper preview)
- ``cells_centroids_map.pdf`` (vector, for paper inclusion)
- ``cells_centroids_map.html`` (interactive, Folium/Leaflet)

The 4,045 centroids are the geographic reference points for the
per-location weather time series in the published Germany 2010 dataset.

Usage:
    uv run python figures/figure_germany_map.py
    uv run python figures/figure_germany_map.py --no-interactive
"""
from __future__ import annotations

import argparse
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_GRID = ROOT / "input" / "grid" / "DE_Grid_ETRS89-UTM32_10km.gpkg"
DEFAULT_OUTPUT_DIR = ROOT / "img"


def load_grid(file_path: Path) -> gpd.GeoDataFrame:
    """Load the BKG grid from a GeoPackage file."""
    return gpd.read_file(file_path)


def compute_centroids(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of ``grid`` with each polygon replaced by its centroid."""
    centroids = grid.copy()
    centroids["geometry"] = centroids.centroid
    return centroids


def render_interactive_map(grid: gpd.GeoDataFrame,
                            centroids: gpd.GeoDataFrame,
                            out_path: Path) -> None:
    """Save an interactive Folium/Leaflet map with an OSM basemap.

    Both the grid polygons and centroid markers are added as toggleable
    layers. The result is a single self-contained HTML file.
    """
    grid_wgs84 = grid.to_crs(epsg=4326)
    centroids_wgs84 = centroids.to_crs(epsg=4326)

    m = grid_wgs84.explore(
        name="Cells",
        style_kwds={"fillOpacity": 0.1, "color": "black", "weight": 1},
    )
    centroids_wgs84.explore(
        m=m,
        name="Centroids",
        marker_type="circle",
        color="purple",
        marker_kwds={"radius": 1, "fill": True, "fillOpacity": 0.8},
    )
    m.save(str(out_path))
    print(f"[germany_map] interactive -> {out_path}")


def render_static_map(grid: gpd.GeoDataFrame,
                       centroids: gpd.GeoDataFrame,
                       png_path: Path,
                       pdf_path: Path) -> None:
    """Save a static map with an OpenStreetMap basemap via contextily.

    The grid and centroids are reprojected to Web Mercator (EPSG:3857)
    so contextily's tile providers can attach correctly.
    """
    grid_3857 = grid.to_crs(epsg=3857)
    centroids_3857 = centroids.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    grid_3857.plot(ax=ax, facecolor="none", edgecolor="black",
                    linewidth=0.5, alpha=0.8)
    centroids_3857.plot(ax=ax, color="purple", markersize=2, alpha=0.9)

    minx, miny, maxx, maxy = grid_3857.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                    crs=grid_3857.crs)
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[germany_map] static   -> {png_path}")
    print(f"[germany_map] static   -> {pdf_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID,
                   help=f"Path to the BKG grid GeoPackage "
                        f"(default: {DEFAULT_GRID})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory for the rendered figures "
                        f"(default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip the Folium interactive HTML output.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[germany_map] loading grid from {args.grid} ...")
    grid = load_grid(args.grid)
    centroids = compute_centroids(grid)
    print(f"[germany_map] {len(grid)} cells, {len(centroids)} centroids")

    render_static_map(grid, centroids,
                       args.out_dir / "cells_centroids_map.png",
                       args.out_dir / "cells_centroids_map.pdf")
    if not args.no_interactive:
        render_interactive_map(grid, centroids,
                                args.out_dir / "cells_centroids_map.html")


if __name__ == "__main__":
    main()
