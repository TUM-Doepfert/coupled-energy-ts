"""Render paper Figure 2: GeoMA occupancy-inference example.

Loads one HTW Berlin profile from ``input/electricity/60min``, computes
the geometric moving-average baseline ``g_t``, detects occupancy as
``p_t > g_t``, and applies the paper's night-rule extension. Plots a
two-panel figure: top is the demand and the baseline; bottom is the
binary occupancy with the night-rule extension shown in a distinct
style so the reader can see what the rule adds on top of raw GeoMA.

Saves PGF (for ``\\input{}`` into the paper), PDF and PNG previews to
``img/`` by default, matching the pattern of
``figure_building_diversity_ts.py``.

The GeoMA recursion and night rule are implemented inline (~10 lines
each) so this script is self-contained and does not depend on EnTiSe
or on the rest of the pipeline. The math matches Methods §2: GeoMA
recursion ``g_{t+1} = alpha * p_t + (1 - alpha) * g_t``, occupied iff
``p_t > g_t``, then if any hour between 21:00 and 23:59 of day d shows
occupancy, force occupancy on hours [00:00, 09:00) of day d+1.

Usage:
    python figures/figure_geoma_example.py
    python figures/figure_geoma_example.py --profile 2630.csv
    python figures/figure_geoma_example.py --alpha 0.05 --window-days 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paper-matched font sizes (tuned for full text-width rendering in
# Nature Sci-Data layout).
mpl.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  11,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize":  9,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_INPUT_DIR  = ROOT / "input" / "electricity" / "60min"
DEFAULT_OUTPUT_DIR = ROOT / "img"
DEFAULT_PROFILE    = "4695.csv"
DEFAULT_ALPHA      = 0.05
DEFAULT_WINDOW_DAYS = 4

# --------------------------------------------------------------------- #
# GeoMA + night rule
# --------------------------------------------------------------------- #

def geoma_baseline(p: pd.Series, alpha: float) -> pd.Series:
    """Recursive smoother: g_{t+1} = alpha * p_t + (1 - alpha) * g_t."""
    arr = p.to_numpy(dtype=float)
    g = np.empty_like(arr)
    g[0] = arr[0]
    for t in range(1, arr.size):
        g[t] = alpha * arr[t - 1] + (1.0 - alpha) * g[t - 1]
    return pd.Series(g, index=p.index, name="baseline_g_t")


def occupancy_geoma(p: pd.Series, g: pd.Series) -> pd.Series:
    """Binary occupancy: 1 where instantaneous demand exceeds baseline."""
    return (p.to_numpy() > g.to_numpy()).astype("int8")


def apply_night_rule(occ: pd.Series,
                     evening_hours: tuple[int, int] = (21, 23),
                     extend_hours:  tuple[int, int] = (0, 9),
                     min_evening_hits: int = 1) -> pd.Series:
    """Extend evening occupancy into the next morning.

    If at least ``min_evening_hits`` of the hours
    ``[evening_hours[0], evening_hours[1]]`` on day d show occupancy,
    force occupancy on day d+1 between
    ``[extend_hours[0], extend_hours[1])``.
    """
    result = occ.copy()
    idx = pd.DatetimeIndex(result.index)
    in_evening = (idx.hour >= evening_hours[0]) & (idx.hour <= evening_hours[1])
    evening = result[in_evening]
    for day, day_values in evening.groupby(evening.index.date):
        if int(day_values.sum()) >= min_evening_hits:
            day_ts = pd.Timestamp(day, tz=idx.tz)
            start = day_ts + pd.Timedelta(days=1, hours=extend_hours[0])
            end   = day_ts + pd.Timedelta(days=1, hours=extend_hours[1])
            result.loc[(idx >= start) & (idx < end)] = 1
    return result


# --------------------------------------------------------------------- #
# IO + window selection
# --------------------------------------------------------------------- #

def load_profile(path: Path) -> pd.DataFrame:
    """Read one HTW Berlin CSV; values are in Wh.

    The CSVs span CET/CEST transitions with mixed offsets, so the index
    is coerced to UTC on read (matches src/preprocessing.py).
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = ["demand_wh"]
    df["demand_kw"] = df["demand_wh"] / 1000.0
    return df


def pick_illustrative_window(occ_geoma: pd.Series,
                             occ_with_rule: pd.Series,
                             window_days: int) -> pd.DatetimeIndex:
    """Pick a multi-day window where the night rule fires at least once.

    Strategy: count, per day, how many hours the night rule adds on top
    of raw GeoMA, then centre the window on the day with the largest
    addition.
    """
    added = (occ_with_rule.astype(int) - occ_geoma.astype(int)).clip(lower=0)
    per_day = added.resample("1D").sum()
    if (per_day > 0).any():
        centre = per_day.idxmax()
    else:
        centre = occ_geoma.index[len(occ_geoma) // 2]
    centre = pd.Timestamp(centre).tz_convert(occ_geoma.index.tz).normalize()
    half = window_days // 2
    start = centre - pd.Timedelta(days=half)
    end   = start  + pd.Timedelta(days=window_days)
    mask  = (occ_geoma.index >= start) & (occ_geoma.index < end)
    return occ_geoma.index[mask]


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #

def render(input_dir: Path,
           profile_csv: str,
           alpha: float,
           window_days: int,
           out_dir: Path,
           min_evening_hits: int = 1) -> None:

    profile = load_profile(input_dir / profile_csv)

    # GeoMA + occupancy
    p_wh = profile["demand_wh"]
    g_wh = geoma_baseline(p_wh, alpha)
    occ_geoma = pd.Series(occupancy_geoma(p_wh, g_wh),
                          index=p_wh.index, name="occ_geoma")
    occ_final = apply_night_rule(occ_geoma,
                                 min_evening_hits=min_evening_hits)

    # Illustrative multi-day window
    win = pick_illustrative_window(occ_geoma, occ_final, window_days)
    p_kw  = profile["demand_kw"].loc[win]
    g_kw  = (g_wh / 1000.0).loc[win]
    # Compute the contiguous 9-hour night-rule envelope on the full series,
    # then slice to the display window. This makes each firing show up as
    # one solid block instead of being broken into strips wherever GeoMA
    # independently hit a morning hour.
    envelope_full = pd.Series(0, index=occ_geoma.index, dtype=int)
    idx_full = pd.DatetimeIndex(occ_geoma.index)
    in_evening_full = (idx_full.hour >= 21) & (idx_full.hour <= 23)
    evening_full = occ_geoma[in_evening_full]
    for day, day_values in evening_full.groupby(evening_full.index.date):
        if int(day_values.sum()) >= min_evening_hits:
            day_ts = pd.Timestamp(day, tz=idx_full.tz)
            start = day_ts + pd.Timedelta(days=1)
            end   = start + pd.Timedelta(hours=9)
            envelope_full.loc[(idx_full >= start) & (idx_full < end)] = 1

    occ_g = occ_geoma.loc[win].astype(int)
    occ_f = occ_final.loc[win].astype(int)
    envelope = envelope_full.loc[win]

    # ---- single-panel figure ---- #
    fig, ax = plt.subplots(figsize=(8.0, 3.2))

    ymax = float(max(p_kw.max(), g_kw.max())) * 1.06

    # Layer 1 (behind): full 9-hour night-rule envelope as one block per firing
    ax.fill_between(envelope.index, 0, ymax, where=envelope.values == 1,
                    step="post",
                    facecolor="#f6c971", alpha=0.45,
                    hatch="///",
                    edgecolor="#a35e00", linewidth=0.0,
                    label="Night-rule extension")
    # Layer 2 (above): GeoMA detections sit on top
    ax.fill_between(occ_g.index, 0, ymax, where=occ_g.values == 1,
                    step="post",
                    color="#2c7d2c", alpha=0.30,
                    label="GeoMA detected",
                    linewidth=0)

    # Demand and baseline on top
    ax.plot(p_kw.index, p_kw.values,
            color="#1f4e79", lw=1.0, label=r"Demand $p_t$", zorder=3)
    ax.plot(g_kw.index, g_kw.values,
            color="#cc7a00", lw=1.4, linestyle="--",
            label=r"Baseline $g_t$ (moving average)", zorder=3)

    ax.set_ylim(0, ymax)
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Date")
    # Legend below the axes in one row so it does not obscure the data
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -0.22),
              ncol=4,
              frameon=False,
              handlelength=2.2,
              columnspacing=1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "geoma"
    pdf  = out_dir / f"{stem}.pdf"
    pgf  = out_dir / f"{stem}.pgf"
    png  = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pgf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {pgf}\nwrote {png}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--profile",    default=DEFAULT_PROFILE,
                        help="CSV filename in --input-dir, e.g. 1398.csv")
    parser.add_argument("--alpha",      type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--window-days", type=int,  default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--out-dir",    type=Path,  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-evening-hits", type=int, default=1,
                        help="Min hours of detected occupancy in 21:00-23:00 "
                             "before the night rule fires (paper text: >=1).")
    args = parser.parse_args()
    render(args.input_dir, args.profile, args.alpha,
           args.window_days, args.out_dir, args.min_evening_hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
