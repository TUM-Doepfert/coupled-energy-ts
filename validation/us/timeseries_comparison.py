"""Time-series overlay for the paper (US validation) — distributional version.

For each climate zone, computes the across-building 25th/50th/75th
percentiles at every hour for both the 1R1C simulation and the EULP
reference, and shows them as filled bands with median lines. This
removes the "single representative building" framing concern: the
figure displays the full per-building spread rather than one example.

Run after simulate.py + building_comparison.py:
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as _mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES


def load_weather_simple(weather_csv: Path) -> pd.DataFrame:
    w = pd.read_csv(weather_csv)
    ts = pd.to_datetime(w["date_time"]) - pd.Timedelta(hours=1)
    return pd.DataFrame({
        "datetime": ts,
        "T_out": pd.to_numeric(w["Dry Bulb Temperature [°C]"]).astype(float),
    }).set_index("datetime")


def _drop_low_signal(metrics_df: pd.DataFrame, zone: str,
                     min_real_mwh: float = 1.0) -> list[int]:
    """List of bldg_ids for this zone with annual real heating > min_real_mwh."""
    sub = metrics_df[(metrics_df["zone"] == zone)
                     & (metrics_df["energy_real_kwh_heat"] > min_real_mwh * 1000.0)]
    return sub["bldg_id"].astype(int).tolist()


def load_zone_panel(rp: Path, sp: Path, bldg_ids: list[int]):
    """Read real + sim parquet for a zone, restrict to informative buildings,
    pivot to hour-by-building matrices, and return (real_h, real_c, sim_h, sim_c)
    each as a (hours x buildings) DataFrame in kW."""
    real = pd.read_parquet(rp, columns=["timestamp", "bldg_id",
                                         "q_heat_w", "q_cool_w"])
    sim = pd.read_parquet(sp, columns=["timestamp", "bldg_id",
                                        "q_heat_w_sim", "q_cool_w_sim"])
    real = real[real["bldg_id"].isin(bldg_ids)]
    sim = sim[sim["bldg_id"].isin(bldg_ids)]
    for df in (real, sim):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    real = real.pivot_table(index="timestamp", columns="bldg_id",
                            values=["q_heat_w", "q_cool_w"], aggfunc="mean")
    sim = sim.pivot_table(index="timestamp", columns="bldg_id",
                          values=["q_heat_w_sim", "q_cool_w_sim"], aggfunc="mean")
    # Resample to hourly so sim (15-min source) and real (hourly) align
    real = real.resample("1h").mean()
    sim = sim.resample("1h").mean()
    real_h = real["q_heat_w"] / 1000.0
    real_c = real["q_cool_w"] / 1000.0
    sim_h = sim["q_heat_w_sim"] / 1000.0
    sim_c = sim["q_cool_w_sim"] / 1000.0
    return real_h, real_c, sim_h, sim_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("validation/us/data"))
    parser.add_argument("--metrics-csv", type=Path,
                        default=Path("validation/us/data/building_comparison_metrics.csv"))
    parser.add_argument("--out-fig", type=Path,
                        default=Path("img/us_timeseries_comparison.png"))
    args = parser.parse_args()

    if not args.metrics_csv.exists():
        sys.exit(f"Missing {args.metrics_csv}; run building_comparison.py first.")
    metrics = pd.read_csv(args.metrics_csv)

    proc = args.data_dir / "processed"
    sim_dir = args.data_dir / "sim"
    weather_dir = args.data_dir / "weather"

    _mpl.rcParams.update({
        "font.size":       16,
        "axes.titlesize":  18,
        "axes.labelsize":  16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    # Collect right-axis (T_out) handles + global min/max so we can unify
    # the temperature axis across all panels in a second pass.
    twin_axes = []
    t_out_min, t_out_max = +1e9, -1e9
    kw_max_pos, kw_max_neg = 0.0, 0.0  # left-axis bounds (heating+, cooling-)
    x_min, x_max = None, None

    for ax_idx, (cid, state, zone, _, _) in enumerate(COUNTIES):
        ax = axes[ax_idx]
        zone_safe = zone.replace(" ", "_")
        rp = proc / f"{cid}_{zone_safe}.parquet"
        sp = sim_dir / f"{cid}_{zone_safe}.parquet"
        wp = weather_dir / f"{cid}_2018.csv"
        if not (rp.exists() and sp.exists() and wp.exists()):
            ax.set_visible(False); continue

        bldg_ids = _drop_low_signal(metrics, zone)
        if len(bldg_ids) < 3:
            print(f"  {zone}: skipped, only {len(bldg_ids)} informative buildings")
            ax.set_visible(False); continue

        weather = load_weather_simple(wp)
        real_h, real_c, sim_h, sim_c = load_zone_panel(rp, sp, bldg_ids)
        # Make all four matrices share the same time index for clean percentile arithmetic
        idx = real_h.index.intersection(sim_h.index)
        real_h = real_h.loc[idx]; real_c = real_c.loc[idx]
        sim_h  = sim_h.loc[idx];  sim_c  = sim_c.loc[idx]

        def pcts(df):
            return (df.quantile(0.25, axis=1),
                    df.quantile(0.50, axis=1),
                    df.quantile(0.75, axis=1))
        rh25, rh50, rh75 = pcts(real_h)
        rc25, rc50, rc75 = pcts(real_c)
        sh25, sh50, sh75 = pcts(sim_h)
        sc25, sc50, sc75 = pcts(sim_c)

        # ---- Plot bands ----
        # EULP heating (grey, solid). Single "Reference" legend entry; the
        # cooling reference line below shares the same style and is not
        # re-labelled.
        ax.fill_between(idx, rh25, rh75, color="#a6a6a6", alpha=0.2, linewidth=0)
        ax.plot(idx, rh50, color="#555555", lw=0.7, alpha=0.9, label="Reference")
        # Sim heating (red, dashed)
        ax.fill_between(idx, sh25, sh75, color="#c0392b", alpha=0.2, linewidth=0)
        ax.plot(idx, sh50, color="#9c2718", lw=0.8, ls="-", alpha=0.95,
                label="Simulation heating")
        # EULP cooling, mirrored below zero (grey, solid; no separate legend entry)
        ax.fill_between(idx, -rc75, -rc25, color="#a6a6a6", alpha=0.2, linewidth=0)
        ax.plot(idx, -rc50, color="#555555", lw=0.7, alpha=0.9)
        # Sim cooling, mirrored below zero (blue, dashed)
        ax.fill_between(idx, -sc75, -sc25, color="#1565a8", alpha=0.2, linewidth=0)
        ax.plot(idx, -sc50, color="#0e436b", lw=0.8, ls="-", alpha=0.95,
                label="Simulation cooling")
        ax.axhline(0, color="black", lw=0.5, alpha=0.6)
        # Track global kW range across zones to unify the left axis later.
        # Track x-axis bounds so we can tighten the plot to the data span.
        if x_min is None or idx.min() < x_min: x_min = idx.min()
        if x_max is None or idx.max() > x_max: x_max = idx.max()
        kw_max_pos = max(kw_max_pos, float(max(sh75.max(), rh75.max())))
        kw_max_neg = max(kw_max_neg, float(max(sc75.max(), rc75.max())))
        ax.set_ylabel(f"Power (kW)")
        from matplotlib.ticker import AutoMinorLocator
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, axis="y", which="major", linestyle=":",
                alpha=0.6, linewidth=0.7)
        ax.grid(True, axis="y", which="minor", linestyle=":",
                alpha=0.6, linewidth=0.5)
        ax.text(0.005, 0.95,
                f"{zone}",
                transform=ax.transAxes, va="top", fontsize=14,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          boxstyle="round,pad=0.25"))

        ax2 = ax.twinx()
        ax2.plot(weather.index, weather["T_out"],
                 color="#2e8b57", lw=0.4, alpha=0.45)
        ax2.set_ylabel("T$_{out}$ (°C)", color="#2e8b57")
        ax2.tick_params(axis="y", colors="#2e8b57")
        twin_axes.append(ax2)
        t_out_min = min(t_out_min, float(weather["T_out"].min()))
        t_out_max = max(t_out_max, float(weather["T_out"].max()))

        # Per-panel legend removed; a shared legend is placed below the figure.


        print(f"  {zone:12s}  n={len(bldg_ids):3d}  "
              f"heat sim/real median@max-day={sh50.max()/max(rh50.max(),0.01):.2f}")

    # Unify the left-side kW axis across all panels so heating-vs-cooling
    # magnitudes are directly comparable between climate zones.
    kw_hi = kw_max_pos * 1.08
    kw_lo = -kw_max_neg * 1.08
    for ax in axes:
        ax.set_ylim(kw_lo, kw_hi)

    # Unify the right-side outdoor-temperature axis across all panels
    if twin_axes:
        global_lo = t_out_min - 3
        global_hi = t_out_max + 3
        for tax in twin_axes:
            tax.set_ylim(global_lo, global_hi)
    # Tight x-axis: start at the first data timestamp, end at the last
    if x_min is not None and x_max is not None:
        for ax in axes:
            ax.set_xlim(x_min, x_max)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    # Shared legend below the panel grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"\nFigure saved -> {args.out_fig}")


if __name__ == "__main__":
    main()
