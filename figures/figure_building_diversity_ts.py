"""Building diversity time-series figure for the paper.

For one (location, archetype) we plot all 74 Q_heat (or Q_cool) traces
as low-alpha thin lines and the cross-profile median as a thick line.
Two stacked panels: a heating-active week and a cooling-active week,
chosen so that household-level decision-disagreement is maximised.

Why this picture: the methodology claim is that coupling per-household
electricity -> GeoMA occupancy -> internal gains -> thermal demand makes
different households at the same (location, archetype) produce
*different* heating/cooling curves. The fan-out of the thin lines at
the start and end of each thermal event is exactly that effect made
visible.

Outputs both PNG (for the GitHub README) and PDF (for camera-ready
inclusion in the paper).

Usage:
    python figures/figure_building_diversity_ts.py
    python figures/figure_building_diversity_ts.py --location-id 372 --archetype-id 11
    python figures/figure_building_diversity_ts.py --seed 42
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

# Camera-ready font sizes — tuned to be legible when the figure is
# rendered at one-column width in a Sci-Data-style layout.
mpl.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  12,
    "axes.labelsize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype":    42,   # embed TrueType so paper PDF is selectable
    "ps.fonttype":     42,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def find_location_with_cooling(
    output_dir: Path,
    archetype_id: int,
    min_cooling_kwh_yr: float = 5.0,
    max_files_to_scan: int = 200,
    seed: int = 7,
) -> int:
    # Deterministic given (output_dir contents, archetype_id, seed). Pass
    # a different ``seed`` to walk the candidate list in a different order
    # and land on a different building.
    """Pick a location_id at the requested archetype with non-trivial
    summer cooling demand."""
    hc_root = output_dir / "HC"
    if not hc_root.is_dir():
        raise FileNotFoundError(
            f"{hc_root} not found. Run "
            f"'coupled-energy-ts run config/germany_2010.yml' first."
        )
    rng = np.random.default_rng(seed)
    candidate_locs = sorted(int(p.name.replace("loc", ""))
                             for p in hc_root.glob("loc*"))
    rng.shuffle(candidate_locs)

    scanned = 0
    for loc in candidate_locs:
        f = hc_root / f"loc{loc:04d}" / f"hc_arch{archetype_id:02d}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["profile_id", "q_cool_w"])
        scanned += 1
        ann = df.groupby("profile_id")["q_cool_w"].sum() / 1000.0
        if ann.max() >= min_cooling_kwh_yr:
            print(f"[hero] picked loc{loc:04d} × archetype {archetype_id}, "
                  f"max cooling = {ann.max():.1f} kWh/yr")
            return loc
        if scanned >= max_files_to_scan:
            raise RuntimeError(
                f"No location with cooling > {min_cooling_kwh_yr} kWh/yr at "
                f"archetype {archetype_id}. Lower --min-cooling-kwh or "
                f"override with --location-id."
            )
    raise RuntimeError("Exhausted candidates without finding meaningful cooling.")


def load_all_profiles(
    output_dir: Path,
    weather_dir: Path,
    location_id: int,
    archetype_id: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    hc_path = output_dir / "HC" / f"loc{location_id:04d}" / f"hc_arch{archetype_id:02d}.parquet"
    w_path = weather_dir / f"loc{location_id:04d}.parquet"
    for p in (hc_path, w_path):
        if not p.exists():
            raise FileNotFoundError(p)

    hc = pd.read_parquet(hc_path)
    hc["timestamp"] = pd.to_datetime(hc["timestamp"], utc=True)

    w = pd.read_parquet(w_path)
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
    t_out = w.set_index("timestamp")["air_temperature"].sort_index()

    target = t_out.index
    q_heat = (hc.pivot_table(index="timestamp", columns="profile_id",
                              values="q_heat_w", aggfunc="mean")
                .sort_index() / 1000.0)
    q_cool = (hc.pivot_table(index="timestamp", columns="profile_id",
                              values="q_cool_w", aggfunc="mean")
                .sort_index() / 1000.0)
    q_heat = q_heat.reindex(target, method="nearest", tolerance=pd.Timedelta("30min"))
    q_cool = q_cool.reindex(target, method="nearest", tolerance=pd.Timedelta("30min"))
    return t_out, q_heat, q_cool


def pick_extreme_weeks(t_out: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Legacy: coldest and hottest 7-day rolling-mean windows. Kept for
    callers that want extreme-T weeks; the figure now uses the spread-
    maximising selection below instead."""
    daily = t_out.resample("D").mean()
    weekly = daily.rolling(7, min_periods=7).mean()
    winter_end = weekly.idxmin()
    summer_end = weekly.idxmax()
    winter_start = (winter_end - pd.Timedelta(days=6)).normalize()
    summer_start = (summer_end - pd.Timedelta(days=6)).normalize()
    return winter_start, summer_start


def pick_high_spread_weeks(
    q_heat: pd.DataFrame,
    q_cool: pd.DataFrame,
    active_kw: float = 0.1,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the heating week and cooling week where households disagree
    most about *whether* to fire the controller.

    For each timestamp we count the number of profiles in the "active"
    state (Q > active_kw). A hour where every profile is active or
    every profile is idle is uninformative — they all agree. A hour
    where (say) 30 of 74 profiles are active is a *decision-diverse*
    hour: same weather, same envelope, but internal gains tip the
    cooling/heating decision differently for different households.
    That's the visible footprint of the coupling.

    Selection criterion: rolling 7-day sum of decision-diverse hours.
    Cooling tends to land in marginal-warm weeks (some homes overheat,
    others don't); heating tends to land in shoulder-season weeks for
    the same reason. Both happen to coincide with where the eye sees
    the largest visual fan-out across the 74 thin lines.
    """
    n_profiles = q_heat.shape[1]

    def _best_week(q: pd.DataFrame) -> pd.Timestamp:
        on_count = (q > active_kw).sum(axis=1)
        # symmetric "disagreement" score: 0 when all on or all off,
        # 1 when exactly half are on.
        disagreement = 1.0 - ((on_count - n_profiles / 2.0).abs()
                                / (n_profiles / 2.0))
        roll = disagreement.rolling("7D", min_periods=120).sum()
        end = roll.idxmax()
        return (end - pd.Timedelta(days=6)).normalize()

    return _best_week(q_heat), _best_week(q_cool)


def render_panel(ax, q_slice: pd.DataFrame, t_out_slice: pd.Series,
                  *, mode: str) -> None:
    idx = q_slice.index
    if mode == "heating":
        line_color, median_color = "#c0392b", "#7b241c"
    else:
        line_color, median_color = "#1565a8", "#0e3e6f"

    # All 74 profiles as thin low-alpha lines
    for pid in q_slice.columns:
        ax.plot(idx, q_slice[pid].values,
                color=line_color, lw=0.4, alpha=0.18)

    # Median as a single thick line
    ax.plot(idx, q_slice.median(axis=1).values,
            color=median_color, lw=2.2)

    ax.set_ylim(bottom=0)
    ax.set_ylabel(("Q$_{heat}$" if mode == "heating" else "Q$_{cool}$") + " [kW]")
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)

    # T_out twin axis, thin trace
    ax_t = ax.twinx()
    ax_t.plot(idx, t_out_slice.values,
              color="#2e8b57", lw=0.9, alpha=0.7)
    ax_t.set_ylabel("T$_{out}$ [°C]", color="#2e8b57")
    ax_t.tick_params(axis="y", labelcolor="#2e8b57")

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))


def render(t_out: pd.Series, q_heat: pd.DataFrame, q_cool: pd.DataFrame,
           out_path: Path, location_id: int, archetype_id: int) -> None:
    heat_start, cool_start = pick_high_spread_weeks(q_heat, q_cool)
    winter_start = heat_start
    summer_start = cool_start
    winter_end = winter_start + pd.Timedelta(days=7)
    summer_end = summer_start + pd.Timedelta(days=7)
    print(f"[hero] high-heating-spread week starts {winter_start.date()}, "
          f"high-cooling-spread week starts {summer_start.date()}")

    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(11, 6.4))
    render_panel(ax_w,
                  q_heat.loc[winter_start:winter_end],
                  t_out.loc[winter_start:winter_end],
                  mode="heating")
    render_panel(ax_s,
                  q_cool.loc[summer_start:summer_end],
                  t_out.loc[summer_start:summer_end],
                  mode="cooling")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save both PNG (raster, for README / preview) and PDF (vector,
    # camera-ready for the paper).
    fig.savefig(out_path, dpi=200)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[building_diversity] figures -> {out_path}, {pdf_path}  "
          f"(loc{location_id:04d} × archetype {archetype_id}, "
          f"all {q_heat.shape[1]} profiles)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--weather-dir", type=Path, default=Path("input/weather/2010"))
    p.add_argument("--location-id", type=int, default=None)
    p.add_argument("--archetype-id", type=int, default=11,
                   help="Default 11 = most-modern TABULA-DE archetype, where "
                        "internal-gain coupling has the largest relative effect.")
    p.add_argument("--out-fig", type=Path,
                   default=Path("img/building_diversity_ts.png"),
                   help="Path of the PNG output. A matching .pdf is written "
                        "alongside it for camera-ready paper inclusion.")
    p.add_argument("--min-cooling-kwh", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7,
                   help="Seed for the location-shuffle in the auto-picker. "
                        "Change to land on a different building.")
    args = p.parse_args()

    if args.location_id is None:
        loc = find_location_with_cooling(
            args.output_dir, args.archetype_id, args.min_cooling_kwh,
            seed=args.seed,
        )
    else:
        loc = args.location_id

    t_out, q_heat, q_cool = load_all_profiles(
        args.output_dir, args.weather_dir, loc, args.archetype_id
    )
    render(t_out, q_heat, q_cool, args.out_fig, loc, args.archetype_id)


if __name__ == "__main__":
    main()
