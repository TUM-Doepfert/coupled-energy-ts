"""Module: variance_attribution.py

Attribute the spread in the released heating and cooling demand series to its
three input dimensions: spatial location (climate), building archetype
(envelope), and electricity profile (occupancy and internal gains).

Answers the reviewer question of which input parameters are primarily
responsible for the variability visible in the per-profile fan of
``figures/figure_building_diversity_ts.py``.

Two complementary views are produced:

1. A three-way variance decomposition of annual demand over a stratified
   sample of locations, all 11 archetypes and all 74 profiles. Because the
   released dataset is a full factorial, the main-effect sums of squares are
   exact and the remainder is interaction.

2. The within-cell profile spread, holding location and archetype fixed. This
   is the quantity the figure shows directly.
"""

from __future__ import annotations

__author__ = "Markus Doepfert"
__credits__ = ["Markus Doepfert"]
__license__ = "MIT"
__maintainer__ = "Markus Doepfert"
__email__ = "markus.doepfert@tum.de"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_ARCHETYPES = 11


def _stratified_locations(mapping_csv: Path, n: int) -> list[int]:
    """Pick ``n`` locations spread over the latitude range, so the sample
    spans the German climate gradient rather than clustering."""
    m = pd.read_csv(mapping_csv).sort_values("latitude").reset_index(drop=True)
    idx = np.linspace(0, len(m) - 1, n).round().astype(int)
    return m.loc[idx, "location_id"].tolist()


def collect(hc_root: Path, locations: list[int]) -> pd.DataFrame:
    """Annual heating and cooling energy in kWh for every
    (location, archetype, profile) triple in the sample."""
    rows = []
    for loc in locations:
        for arch in range(1, N_ARCHETYPES + 1):
            f = hc_root / f"loc{loc:04d}" / f"hc_arch{arch:02d}.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(f, columns=["profile_id", "q_heat_w", "q_cool_w"])
            g = d.groupby("profile_id")[["q_heat_w", "q_cool_w"]].sum() / 1000.0
            g = g.reset_index().rename(columns={"q_heat_w": "heat_kwh",
                                                "q_cool_w": "cool_kwh"})
            g["location_id"] = loc
            g["archetype_id"] = arch
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def decompose(df: pd.DataFrame, value: str) -> pd.Series:
    """Main-effect share of total variance for a balanced full factorial.

    SS_factor is the between-group sum of squares of the factor's marginal
    means, scaled by the number of observations per level. What the three
    main effects do not explain is interaction between them.
    """
    v = df[value].to_numpy(float)
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    if ss_total <= 0:
        return pd.Series(dtype=float)
    out = {}
    n = len(df)
    for factor in ["location_id", "archetype_id", "profile_id"]:
        means = df.groupby(factor)[value].mean()
        counts = df.groupby(factor)[value].size()
        out[factor] = float((counts * (means - grand) ** 2).sum() / ss_total)
    out["interaction_and_residual"] = 1.0 - sum(out.values())
    out["_n_observations"] = n
    out["_mean_kwh"] = grand
    out["_cv"] = v.std() / grand if grand else np.nan
    return pd.Series(out)


def within_cell(df: pd.DataFrame, value: str) -> pd.Series:
    """Spread across the 74 profiles with location and archetype held fixed,
    summarised over all sampled cells. This is what the figure shows."""
    g = df.groupby(["location_id", "archetype_id"])[value]
    med = g.median()
    iqr = g.quantile(0.75) - g.quantile(0.25)
    rng = g.max() - g.min()
    keep = med > 0
    return pd.Series({
        "median_iqr_over_median": float((iqr[keep] / med[keep]).median()),
        "median_range_over_median": float((rng[keep] / med[keep]).median()),
        "n_cells": int(keep.sum()),
    })


LOAD_BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
LOAD_LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]


def load_level_spread(hc_root: Path, locations: list[int],
                      archetypes: tuple[int, ...] = (1, 6, 11),
                      min_active_hours: int = 200) -> dict[str, pd.DataFrame]:
    """Cross-profile spread as a function of load level, per (location,
    archetype) cell.

    This is the situation Figure 4 shows: location and archetype fixed, so the
    only varying input is the electricity profile. For each cell the hourly
    cross-profile IQR is divided by the cross-profile median, active hours are
    binned by load relative to that cell's own peak, and the median taken
    within each bin. Reported across cells so the result does not rest on a
    single cell, which is not representative.
    """
    acc: dict[str, list[pd.Series]] = {"q_heat_w": [], "q_cool_w": []}
    for loc in locations:
        for arch in archetypes:
            f = hc_root / f"loc{loc:04d}" / f"hc_arch{arch:02d}.parquet"
            if not f.exists():
                continue
            d = pd.read_parquet(
                f, columns=["timestamp", "profile_id", "q_heat_w", "q_cool_w"])
            for col in ("q_heat_w", "q_cool_w"):
                if d[col].sum() <= 0:
                    continue
                w = d.pivot_table(index="timestamp", columns="profile_id",
                                  values=col)
                med = w.median(axis=1)
                on = med > 0
                if int(on.sum()) < min_active_hours:
                    continue
                q = med[on]
                iqr = (w.quantile(0.75, axis=1) - w.quantile(0.25, axis=1))[on]
                b = pd.cut(q / q.max(), LOAD_BINS, labels=LOAD_LABELS)
                ser = (iqr / q).groupby(b, observed=False).median()
                ser.name = (loc, arch)
                acc[col].append(ser)
    out = {}
    for col, frames in acc.items():
        if not frames:
            continue
        a = pd.DataFrame(frames).T
        out[col] = pd.DataFrame({
            "median": a.median(axis=1),
            "p10": a.quantile(0.10, axis=1),
            "p90": a.quantile(0.90, axis=1),
            "n_cells": a.notna().sum(axis=1),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--n-locations", type=int, default=40)
    p.add_argument("--n-cell-locations", type=int, default=20,
                   help="locations used for the per-cell load-level view")
    p.add_argument("--cell-archetypes", type=int, nargs="+", default=[1, 6, 11])
    p.add_argument("--out-csv", type=Path,
                   default=Path("validation/germany/data/variance_attribution.csv"))
    a = p.parse_args()

    locs = _stratified_locations(a.output_dir / "location_mapping.csv", a.n_locations)
    print(f"[sample] {len(locs)} locations x {N_ARCHETYPES} archetypes x 74 profiles")
    df = collect(a.output_dir / "HC", locs)
    print(f"[sample] {len(df):,} (location, archetype, profile) triples")

    res = {}
    for value in ["heat_kwh", "cool_kwh"]:
        sub = df if value == "heat_kwh" else df[df.groupby(
            ["location_id", "archetype_id"])["cool_kwh"].transform("median") > 0]
        res[value] = decompose(sub, value)
        res[value] = pd.concat([res[value], within_cell(sub, value)])
    out = pd.DataFrame(res)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== variance decomposition, annual demand ===")
    print(out.to_string())

    ll_locs = locs[:: max(1, len(locs) // a.n_cell_locations)][:a.n_cell_locations]
    print(f"\n=== load-level cross-profile spread, {len(ll_locs)} locations "
          f"x archetypes {a.cell_archetypes} ===")
    spread = load_level_spread(a.output_dir / "HC", ll_locs,
                               tuple(a.cell_archetypes))
    for col, tab in spread.items():
        print(f"\n{col}")
        print(tab.to_string())
        tab.to_csv(a.out_csv.with_name(
            a.out_csv.stem + f"_load_level_{col}.csv"))


if __name__ == "__main__":
    main()
