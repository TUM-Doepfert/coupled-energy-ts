"""Per-zone load-diversity figure for the paper.

The paper does NOT use Winter et al.'s peak simultaneity factor g(n)
because it collapses a year of dynamics into a single number dominated
by the coldest hour, where any 74-household stock saturates near 1.
Instead, we compare the load-duration properties of the NREL EULP
stock (their full ResStock simulator) against our pipeline run on the
same buildings (US per-building inputs: matched setpoints, occupants,
geometry, weather). Both populations contain the same buildings; the
only difference is the simulation methodology.

The figure is a 6x2 per-zone grid:

    rows = climate zone (Marine, Cold, Mixed-Humid, Very Cold,
            Hot-Humid, Hot-Dry — same order as COUNTIES)
    left column  = heating load-duration curve
    right column = cooling load-duration curve

Per-zone is the right denominator: NREL and our pipeline see the same
weather year and the same building stock per zone, so the
sum-of-individual-peaks normalisation has the same physical meaning on
both sides. Aggregating across climate zones depresses the curves by
a constant factor (each zone's peaks happen on different days, so the
numerator at any hour can never hit the cross-zone sum-of-peaks) — that
artefact is the only reason the cross-zone aggregate looked so much
worse than these per-zone curves.

The supplementary active-share figure is rendered separately by
``--share`` (or both with ``--share --ldc``).

Outputs both PNG and PDF.

Usage:
    python validation/us/diversity_comparison.py
    python validation/us/diversity_comparison.py --share
    python validation/us/diversity_comparison.py --no-cooling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from COUNTIES import COUNTIES  # noqa: E402

COLS = {
    "heat": {"nrel": "q_heat_w", "sim": "q_heat_w_sim"},
    "cool": {"nrel": "q_cool_w", "sim": "q_cool_w_sim"},
}
ACTIVE_KW = 0.1


def _wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot timestamp×bldg_id wide; convert W to kW."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return (df.pivot_table(index="timestamp", columns="bldg_id",
                            values=value_col, aggfunc="mean")
              / 1000.0).sort_index()


def load_pair(county_id: str, zone: str, data_dir: Path,
              modes: tuple[str, ...] = ("heat", "cool")
              ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None:
    """Return ``{mode: (nrel_wide, sim_wide)}`` for the requested modes."""
    zone_safe = zone.replace(" ", "_")
    nrel_pq = data_dir / "processed" / f"{county_id}_{zone_safe}.parquet"
    sim_pq = data_dir / "sim" / f"{county_id}_{zone_safe}.parquet"
    if not (nrel_pq.exists() and sim_pq.exists()):
        return None
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mode in modes:
        nrel_col = COLS[mode]["nrel"]
        sim_col = COLS[mode]["sim"]
        try:
            n = pd.read_parquet(nrel_pq, columns=["timestamp", "bldg_id", nrel_col])
            s = pd.read_parquet(sim_pq, columns=["timestamp", "bldg_id", sim_col])
        except (KeyError, ValueError):
            print(f"  [{county_id} {zone}] mode={mode}: column missing, skipping")
            continue
        n_w = _wide(n, nrel_col)
        s_w = _wide(s, sim_col)
        common_b = sorted(set(n_w.columns) & set(s_w.columns))
        common_t = n_w.index.intersection(s_w.index)
        if not common_b or len(common_t) == 0:
            continue
        out[mode] = (n_w.loc[common_t, common_b], s_w.loc[common_t, common_b])
    return out or None


def aggregate_pairs(
    data_dir: Path, modes: tuple[str, ...] = ("heat", "cool"),
) -> dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Load every (county, zone) pair into ``{mode: {zone: (nrel, sim)}}``."""
    by_mode: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {
        m: {} for m in modes
    }
    for cid, _state, zone, _exp, _label in COUNTIES:
        per_mode = load_pair(cid, zone, data_dir, modes=modes)
        if per_mode is None:
            print(f"[load_diversity] skip {cid} {zone}: data missing")
            continue
        for m, (n_w, s_w) in per_mode.items():
            by_mode[m][zone] = (n_w, s_w)
            print(f"  {zone:<14} mode={m:<4} buildings={n_w.shape[1]:>4}  "
                  f"timestamps={n_w.shape[0]}")
    return by_mode


def panel_ldc(ax, nrel_wide: pd.DataFrame, sim_wide: pd.DataFrame,
              mode: str, *, show_legend: bool = False,
              show_xlabel: bool = True, show_ylabel: bool = True,
              normalize: str = "sum-of-peaks") -> None:
    """Aggregate-normalised LDC: sum across buildings each hour, divide
    by the chosen denominator, sort descending.

    Both curves are normalised by the **same** denominator, derived from
    the NREL reference. This preserves energy-magnitude information: if
    the two curves overlap, the two stocks deliver the same total annual
    energy; a vertically lower simulator curve means lower delivered
    energy, not just a different shape.

    ``normalize``:
      - ``"aggregate-peak"``: shared denominator = max_t agg_nrel(t).
        Reference anchored at 1.0; simulator's vertical position
        encodes its peak-aggregate ratio against the reference.
      - ``"sum-of-peaks"`` (default): shared denominator =
        Σ_i max_t P_nrel,i(t). Recovers the Winter-style
        "are peaks synchronized?" interpretation while keeping
        magnitudes comparable between simulators.
    """

    def _curve(wide: pd.DataFrame, denom: float) -> np.ndarray:
        agg = wide.sum(axis=1, min_count=1).dropna()
        return np.sort(agg.values / max(denom, 1e-9))[::-1]

    if normalize == "sum-of-peaks":
        denom = float(nrel_wide.max(axis=0).sum())
    else:  # aggregate-peak
        denom = float(nrel_wide.sum(axis=1, min_count=1).dropna().max())

    n_curve = _curve(nrel_wide, denom)
    s_curve = _curve(sim_wide, denom)
    n = len(n_curve)
    x = np.arange(n) / max(n - 1, 1)

    sim_color = "#c0392b" if mode == "heat" else "#1565a8"
    ax.plot(x, n_curve, color="#555555", lw=1.6, label="Reference")
    ax.plot(x, s_curve, color=sim_color, lw=1.6, ls="--",
            label=f"Sim {'heating' if mode == 'heat' else 'cooling'}")
    ax.set_xlim(0, 1)
    y_top = max(1.0, float(n_curve.max()), float(s_curve.max())) * 1.05
    ax.set_ylim(0, y_top)
    if show_xlabel:
        label_long = "heating" if mode == "heat" else "cooling"
        ax.set_xlabel("Time fraction")
    if show_ylabel:
        ax.set_ylabel(
            "Aggregate / Max" if normalize == "aggregate-peak"
            else "Aggregate / Σ peaks"
        )
    ax.grid(True, ls=":", alpha=0.4)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9)


def _active_share_values(wide: pd.DataFrame,
                          condition_on_active: bool = True) -> np.ndarray:
    n_b = wide.shape[1]
    if n_b == 0:
        return np.array([])
    active = (wide > ACTIVE_KW).sum(axis=1)
    share = (active / n_b).dropna().values
    if condition_on_active:
        share = share[share > 0]
    return share


def _share_ymax(by_mode: dict, mode: str, bins: np.ndarray,
                 condition_on_active: bool = True) -> float:
    """Maximum fraction-of-hours per share bin across all zones for the mode."""
    y = 0.0
    for zone, (n_w, s_w) in by_mode.get(mode, {}).items():
        for w in (n_w, s_w):
            share = _active_share_values(w, condition_on_active)
            if share.size == 0:
                continue
            counts, _ = np.histogram(share, bins=bins)
            frac = counts / share.size
            y = max(y, float(frac.max()))
    return y * 1.05  # 5% headroom


def panel_active_share(ax, nrel_wide: pd.DataFrame, sim_wide: pd.DataFrame,
                        mode: str, *, show_legend: bool = False,
                        show_xlabel: bool = True, show_ylabel: bool = True,
                        condition_on_active: bool = True,
                        y_max: float | None = None) -> None:
    """Distribution of fraction-of-buildings-active across the year.

    Bar height = fraction of hours falling in the corresponding active-share
    bin (each bar normalised by the total number of qualifying hours, so the
    bars sum to 1 across the full [0, 1] range).

    When ``condition_on_active=True`` (default), hours where the entire
    stock is inactive (share = 0, e.g. summer for heating) are filtered
    out before binning. This isolates the *diversity* signal from the
    weather-driven on/off envelope: the remaining hours are exactly
    those where the population is partially active, and the spread of
    the share distribution measures cross-building heterogeneity.

    ``y_max`` (optional) forces a common upper y-limit. Used by the
    combined renderer to lock all heating coincidence panels to one
    scale and all cooling coincidence panels to another.
    """
    n_share = _active_share_values(nrel_wide, condition_on_active)
    s_share = _active_share_values(sim_wide, condition_on_active)

    bins = np.linspace(0.0, 1.0, 41)
    sim_color = "#c0392b" if mode == "heat" else "#1565a8"
    n_weights = np.full_like(n_share, 1.0 / n_share.size) if n_share.size else None
    s_weights = np.full_like(s_share, 1.0 / s_share.size) if s_share.size else None
    ax.hist(n_share, bins=bins, weights=n_weights, alpha=0.55,
            color="#a6a6a6", edgecolor="#555555", linewidth=0.4,
            label="Reference")
    ax.hist(s_share, bins=bins, weights=s_weights, alpha=0.35,
            color=sim_color, edgecolor=sim_color, linewidth=0.4,
            label=f"Sim {'heating' if mode == 'heat' else 'cooling'}")
    ax.set_xlim(0, 1)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    if show_xlabel:
        ax.set_xlabel("Active fraction")
    if show_ylabel:
        ax.set_ylabel("Fraction of hours")
    ax.grid(True, ls=":", alpha=0.4)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9)


def render_ldc_per_zone(by_mode, out_path: Path,
                         include_cooling: bool = True,
                         normalize: str = "sum-of-peaks") -> None:
    """6 rows × 2 cols. Each row a zone; left = heating, right = cooling.
    If --no-cooling, single column.
    """
    zones = [z for _cid, _st, z, _e, _l in COUNTIES]
    zones_with_data = [
        z for z in zones
        if z in by_mode.get("heat", {})
        or (include_cooling and z in by_mode.get("cool", {}))
    ]
    if not zones_with_data:
        sys.exit("No data to render.")

    n_cols = 2 if include_cooling else 1
    fig, axes = plt.subplots(
        len(zones_with_data), n_cols,
        figsize=(5.5 * n_cols, 2.6 * len(zones_with_data)),
        squeeze=False,
    )
    for row, zone in enumerate(zones_with_data):
        is_last = row == len(zones_with_data) - 1
        if zone in by_mode.get("heat", {}):
            n_w, s_w = by_mode["heat"][zone]
            panel_ldc(axes[row][0], n_w, s_w, "heat",
                       show_legend=(row == 0),
                       show_xlabel=is_last,
                       show_ylabel=True,
                       normalize=normalize)
        else:
            axes[row][0].set_visible(False)
        axes[row][0].set_title(f"{zone} — heating", loc="left")

        if include_cooling:
            if zone in by_mode.get("cool", {}):
                n_w, s_w = by_mode["cool"][zone]
                panel_ldc(axes[row][1], n_w, s_w, "cool",
                           show_legend=(row == 0),
                           show_xlabel=is_last,
                           show_ylabel=False,
                           normalize=normalize)
            else:
                axes[row][1].set_visible(False)
            axes[row][1].set_title(f"{zone} — cooling", loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[load_diversity LDC per-zone] -> {out_path} and {out_path.with_suffix('.pdf')}")


def render_active_share(by_mode, out_path: Path,
                         include_cooling: bool = True) -> None:
    """Aggregate (across-zone) active-share histogram. 1 row × 2 cols
    (heating + cooling). This is the methodology fingerprint plot —
    aggregating is fine here because the histogram of an active-fraction
    is dimensionless and zone-mixing washes out cleanly.
    """
    modes = ["heat"] + (["cool"] if include_cooling else [])
    fig, axes = plt.subplots(1, len(modes),
                              figsize=(6.5 * len(modes), 4.5),
                              squeeze=False)
    for col, m in enumerate(modes):
        per_zone = by_mode.get(m, {})
        if not per_zone:
            axes[0][col].set_visible(False)
            continue
        # Concatenate horizontally to one wide frame across zones
        n_frames = [n_w.rename(columns={c: f"{z}_{c}" for c in n_w.columns})
                     for z, (n_w, _) in per_zone.items()]
        s_frames = [s_w.rename(columns={c: f"{z}_{c}" for c in s_w.columns})
                     for z, (_, s_w) in per_zone.items()]
        n_all = pd.concat(n_frames, axis=1, join="outer")
        s_all = pd.concat(s_frames, axis=1, join="outer")
        panel_active_share(axes[0][col], n_all, s_all, m, show_legend=True)
        title = "Heating" if m == "heat" else "Cooling"
        axes[0][col].set_title(f"{title} — simultaneous-activity distribution",
                                loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[load_diversity active-share] -> {out_path} and {out_path.with_suffix('.pdf')}")





def render_combined(by_mode, out_path: Path,
                    include_cooling: bool = True,
                    normalize: str = "sum-of-peaks") -> None:
    """Combined per-zone figure: LDC (aggregate shape) + active-share
    (weather-independent diversity test) side by side for each mode.

    Layout: 6 rows × 4 cols when cooling is included, else 6 × 2.
        cols = (heat LDC, heat active-share,
                cool LDC, cool active-share)
    """
    zones = [z for _cid, _st, z, _e, _l in COUNTIES]
    zones_with_data = [
        z for z in zones
        if z in by_mode.get("heat", {})
        or (include_cooling and z in by_mode.get("cool", {}))
    ]
    if not zones_with_data:
        sys.exit("No data to render.")

    modes = ["heat"] + (["cool"] if include_cooling else [])
    n_cols = 2 * len(modes)
    fig, axes = plt.subplots(
        len(zones_with_data), n_cols,
        figsize=(3.4 * n_cols, 2.6 * len(zones_with_data)),
        squeeze=False,
    )

    # Precompute a per-mode y-limit so all coincidence panels of the same
    # mode (heating across all zones, cooling across all zones) share a
    # common vertical scale and remain visually comparable.
    share_bins = np.linspace(0.0, 1.0, 41)
    y_max_share = {mode: _share_ymax(by_mode, mode, share_bins) for mode in modes}

    for row, zone in enumerate(zones_with_data):
        is_last = row == len(zones_with_data) - 1
        for m_idx, mode in enumerate(modes):
            ldc_ax = axes[row][m_idx * 2]
            sh_ax  = axes[row][m_idx * 2 + 1]
            label_long = "heating" if mode == "heat" else "cooling"
            ldc_ax.set_title(f"{zone} — {label_long.capitalize()} LDC", loc="left",
                             fontsize=11, pad=4)
            sh_ax.set_title(f"{zone} — {label_long.capitalize()} Coincidence",
                            loc="left", fontsize=11, pad=4)

            if zone in by_mode.get(mode, {}):
                n_w, s_w = by_mode[mode][zone]
                panel_ldc(ldc_ax, n_w, s_w, mode,
                           show_legend=False,
                           show_xlabel=is_last,
                           show_ylabel=(m_idx == 0),
                           normalize=normalize)
                panel_active_share(sh_ax, n_w, s_w, mode,
                                   show_legend=False,
                                   show_xlabel=is_last,
                                   show_ylabel=(m_idx == 0),
                                   y_max=y_max_share[mode])
            else:
                ldc_ax.set_visible(False)
                sh_ax.set_visible(False)

    # Single shared legend below the panel grid (3 entries).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="#555555", lw=2.0, label="Reference"),
        Line2D([0], [0], color="#c0392b", lw=2.0, ls="--", label="Simulation heating"),
        Line2D([0], [0], color="#1565a8", lw=2.0, ls="--", label="Simulation cooling"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005),
               ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[load_diversity combined LDC+share] -> {out_path}"
          f" and {out_path.with_suffix('.pdf')}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "validation" / "us" / "data")
    p.add_argument("--out", type=Path,
                   default=Path("img/us_diversity_comparison.png"),
                   help="Output path for the LDC per-zone grid.")
    p.add_argument("--share-out", type=Path,
                   default=Path("img/us_diversity_comparison_share.png"),
                   help="Output path for the active-share figure.")
    p.add_argument("--ldc", action="store_true",
                   help="Render the legacy per-zone LDC grid only.")
    p.add_argument("--combined", action="store_true",
                   help="Render the combined LDC + active-share per-zone grid (default).")
    p.add_argument("--share", action="store_true",
                   help="Also render the cross-zone active-share figure.")
    p.add_argument("--no-cooling", action="store_true",
                   help="Render heating only (omit cooling column / panel).")
    p.add_argument("--normalize", choices=["aggregate-peak", "sum-of-peaks"],
                   default="sum-of-peaks",
                   help="LDC denominator. 'sum-of-peaks' (default) is the "
                        "Winter-style synchronization metric (sum of per-"
                        "building annual peaks), sensitive to setback "
                        "recovery. 'aggregate-peak' is the standard utility "
                        "LDC anchored at 1.0 - both curves compared by SHAPE "
                        "only, robust to per-building peak inflation from "
                        "morning recovery spikes.")
    args = p.parse_args()

    # Default behaviour: combined LDC + active-share per zone
    do_combined = args.combined or (not args.ldc and not args.share)
    do_ldc = args.ldc
    do_share = args.share

    include_cooling = not args.no_cooling
    modes = ("heat", "cool") if include_cooling else ("heat",)
    print("[load_diversity] loading per-zone data ...")
    by_mode = aggregate_pairs(args.data_dir, modes=modes)
    if not any(by_mode.values()):
        sys.exit("No data found; run validation/us/runme.py first.")

    if do_combined:
        render_combined(by_mode, args.out, include_cooling=include_cooling,
                        normalize=args.normalize)
    if do_ldc:
        render_ldc_per_zone(by_mode, args.out, include_cooling=include_cooling,
                             normalize=args.normalize)
    if do_share:
        render_active_share(by_mode, args.share_out, include_cooling=include_cooling)


if __name__ == "__main__":
    main()
