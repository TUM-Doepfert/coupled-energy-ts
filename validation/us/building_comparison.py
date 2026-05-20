"""US per-building comparison: per-building RMSE + correlation + per-m^2 scatter.

Reads:
  - data/processed/<county>_<zone>.parquet      : NREL EULP truth
  - data/sim/<county>_<zone>.parquet     : our R1C1 simulation
  - data/<county>_<zone>_qualified.csv          : per-building metadata
                                                  (sqft, vintage, infiltration...)

Outputs:
  - <out-fig>                       : 2x6 panel of per-m^2 log-log scatter
  - <out-csv>                       : per-building metrics (long format)
  - data/building_comparison_outliers.csv        : worst N buildings per zone with their
                                      key inputs, for diagnosis or filtering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES

SQFT_TO_M2 = 0.092903


def per_building_metrics(real_df, sim_df, var):
    real_col = var
    sim_col = var + "_sim"
    rows = []
    for bid, real_b in real_df.groupby("bldg_id"):
        sim_b = sim_df[sim_df["bldg_id"] == bid]
        if sim_b.empty:
            continue
        rb = real_b.copy()
        sb = sim_b.copy()
        rb["timestamp"] = pd.to_datetime(rb["timestamp"])
        sb["timestamp"] = pd.to_datetime(sb["timestamp"])
        if rb["timestamp"].dt.tz is not None:
            rb["timestamp"] = rb["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        if sb["timestamp"].dt.tz is not None:
            sb["timestamp"] = sb["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        merged = pd.merge(
            rb[["timestamp", real_col]],
            sb[["timestamp", sim_col]],
            on="timestamp", how="inner",
        )
        if len(merged) < 100:
            continue
        r_arr = merged[real_col].to_numpy(dtype=np.float64)
        s_arr = merged[sim_col].to_numpy(dtype=np.float64)
        if r_arr.std() < 1e-6 and s_arr.std() < 1e-6:
            pearson = np.nan
        else:
            try:
                pearson, _ = pearsonr(r_arr, s_arr)
            except Exception:
                pearson = np.nan
        rmse = float(np.sqrt(np.mean((r_arr - s_arr) ** 2)))
        rows.append({
            "bldg_id": bid,
            f"rmse_{var}": rmse,
            f"r_{var}": pearson,
            f"energy_real_kwh": r_arr.sum() * 0.25 / 1000,
            f"energy_sim_kwh": s_arr.sum() * 0.25 / 1000,
            f"peak_real_w": r_arr.max(),
            f"peak_sim_w": s_arr.max(),
        })
    return pd.DataFrame(rows)


def _load_qualified(data_dir, cid, zone_safe):
    csv = data_dir / f"{cid}_{zone_safe}_qualified.csv"
    if not csv.exists():
        return None
    keep = ["bldg_id", "in.sqft", "in.geometry_stories", "in.vintage",
            "in.infiltration", "in.heating_setpoint", "in.cooling_setpoint",
            "in.hvac_heating_type_and_fuel"]
    df = pd.read_csv(csv, usecols=lambda c: c in keep)
    df["area_m2"] = df["in.sqft"].astype(float) * SQFT_TO_M2
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
        default=Path("validation/us/data"))
    parser.add_argument("--out-fig", type=Path,
        default=Path("img/us_building_comparison.png"))
    parser.add_argument("--out-csv", type=Path,
        default=Path("validation/us/data/building_comparison_metrics.csv"))
    parser.add_argument("--outlier-csv", type=Path,
        default=Path("validation/us/data/building_comparison_outliers.csv"))
    parser.add_argument("--outlier-ratio", type=float, default=3.0,
        help="Flag a building as outlier if max(sim/real, real/sim) > this.")
    parser.add_argument("--top-n-outliers", type=int, default=10)
    parser.add_argument("--min-real-mwh", type=float, default=1.0,
        help="Buildings with real annual energy < this MWh are excluded "
             "from per-variable metrics (heating or cooling) as "
             "non-informative low-signal cases.")
    args = parser.parse_args()

    proc_dir = args.data_dir / "processed"
    sim_dir = args.data_dir / "sim"

    all_metrics = []
    # Font bump applied before the figure is created so all later
    # elements pick the larger sizes up. Figsize kept at the original
    # 18x7 — the issue was only relative font size at print scale.
    import matplotlib as _mpl
    _mpl.rcParams.update({
        "font.size":       14,
        "axes.titlesize":  16,
        "axes.labelsize":  14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))

    for col_idx, (cid, state, zone, _, _) in enumerate(COUNTIES):
        zone_safe = zone.replace(" ", "_")
        real_pq = proc_dir / f"{cid}_{zone_safe}.parquet"
        sim_pq  = sim_dir / f"{cid}_{zone_safe}.parquet"
        if not (real_pq.exists() and sim_pq.exists()):
            print(f"[skip] {cid}: real or sim missing")
            for r in (0, 1):
                axes[r, col_idx].set_visible(False)
            continue

        real_df = pd.read_parquet(real_pq)
        sim_df  = pd.read_parquet(sim_pq)
        meta = _load_qualified(args.data_dir, cid, zone_safe)

        m_h = per_building_metrics(real_df, sim_df, "q_heat_w")
        m_c = per_building_metrics(real_df, sim_df, "q_cool_w")
        m = m_h.merge(m_c[["bldg_id","rmse_q_cool_w","r_q_cool_w",
                          "energy_real_kwh","energy_sim_kwh",
                          "peak_real_w","peak_sim_w"]],
                      on="bldg_id", suffixes=("_heat","_cool"))
        if meta is not None:
            m = m.merge(meta, on="bldg_id", how="left")
        else:
            m["area_m2"] = np.nan
        m["county"] = cid; m["zone"] = zone

        # per-m^2 normalization
        m["heat_real_kwh_m2"] = m["energy_real_kwh_heat"] / m["area_m2"]
        m["heat_sim_kwh_m2"]  = m["energy_sim_kwh_heat"]  / m["area_m2"]
        m["cool_real_kwh_m2"] = m["energy_real_kwh_cool"] / m["area_m2"]
        m["cool_sim_kwh_m2"]  = m["energy_sim_kwh_cool"]  / m["area_m2"]
        # ratio (avoid div-by-zero): max(sim/real, real/sim) on annual energy
        eps = 1.0  # 1 kWh/yr floor to avoid blowup on near-zero buildings
        m["heat_ratio"] = np.maximum(
            m["energy_sim_kwh_heat"] / np.maximum(m["energy_real_kwh_heat"], eps),
            m["energy_real_kwh_heat"] / np.maximum(m["energy_sim_kwh_heat"], eps),
        )
        m["cool_ratio"] = np.maximum(
            m["energy_sim_kwh_cool"] / np.maximum(m["energy_real_kwh_cool"], eps),
            m["energy_real_kwh_cool"] / np.maximum(m["energy_sim_kwh_cool"], eps),
        )
        all_metrics.append(m)

        # Mark low-real-signal buildings for exclusion from per-variable metrics
        m["low_signal_heat"] = m["energy_real_kwh_heat"] < args.min_real_mwh * 1000
        m["low_signal_cool"] = m["energy_real_kwh_cool"] < args.min_real_mwh * 1000

        # ── Plots: log-log, per m^2 ──
        for row_idx, (var, color, ylabel, xlabel) in enumerate([
            ("heat", "#c0392b", "Simulation (kWh/m$^{2}$)", "Reference (kWh/m$^{2}$)"),
            ("cool", "#1565a8", "Simulation (kWh/m$^{2}$)", "Reference (kWh/m$^{2}$)"),
        ]):
            ax = axes[row_idx, col_idx]
            mask_low = m[f"low_signal_{var}"]
            x = m[f"{var}_real_kwh_m2"].clip(lower=0.5)
            y = m[f"{var}_sim_kwh_m2"].clip(lower=0.5)
            # Plot only informative buildings; low-signal ones are flagged
            # in the metrics CSV and described in the caption.
            ax.scatter(x[~mask_low], y[~mask_low], s=18, alpha=0.75, c=color)
            lo, hi = 0.5, max(1000, x.max(), y.max()) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            # Force a tick at every decade so 10^0 .. 10^3 are all visible.
            from matplotlib.ticker import LogLocator
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            if row_idx == 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(zone, fontsize=16, pad=8)
            # Stats annotation: median over informative buildings only
            metric_col = {'heat':'heat_w','cool':'cool_w'}[var]
            valid = m[~mask_low]
            r_med = valid[f"r_q_{metric_col}"].median()
            rmse_med = valid[f"rmse_q_{metric_col}"].median() / 1000
            rmse_max = valid[f"rmse_q_{metric_col}"].max() / 1000 if len(valid) else float('nan')
            n_out = ((m[f"{var}_ratio"] > args.outlier_ratio) & (~mask_low)).sum()
            ax.text(0.04, 0.96,
                    f"RMSE = {rmse_med:.1f} kW\n$\\tilde r$ = {r_med:.2f}",
                    transform=ax.transAxes, va="top", fontsize=14,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                              boxstyle="round,pad=0.35"))
    # Suptitle removed; caption serves that role for the paper.
    # One shared legend for the low-signal grey markers.
    # Bottom legend removed — row layout + colour code already convey heating/cooling.
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"Figure -> {args.out_fig}")

    if all_metrics:
        df = pd.concat(all_metrics, ignore_index=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Metrics -> {args.out_csv}")

        # ── Outlier dump ──
        outliers_h = df[df["heat_ratio"] > args.outlier_ratio].nlargest(
            args.top_n_outliers * 6, "heat_ratio")
        outliers_c = df[df["cool_ratio"] > args.outlier_ratio].nlargest(
            args.top_n_outliers * 6, "cool_ratio")
        out_cols = ["zone","county","bldg_id","in.vintage","in.sqft","area_m2",
                    "in.geometry_stories","in.infiltration",
                    "in.heating_setpoint","in.hvac_heating_type_and_fuel",
                    "energy_real_kwh_heat","energy_sim_kwh_heat","heat_ratio",
                    "energy_real_kwh_cool","energy_sim_kwh_cool","cool_ratio",
                    "heat_real_kwh_m2","heat_sim_kwh_m2",
                    "cool_real_kwh_m2","cool_sim_kwh_m2",
                    "rmse_q_heat_w","rmse_q_cool_w","r_q_heat_w","r_q_cool_w"]
        out_cols = [c for c in out_cols if c in df.columns]
        outliers = (
            pd.concat([outliers_h, outliers_c]).drop_duplicates("bldg_id")
            .sort_values(["zone","heat_ratio"], ascending=[True, False])
        )
        outliers[out_cols].to_csv(args.outlier_csv, index=False)
        print(f"Outliers (>{args.outlier_ratio}× sim/real, top per zone) -> "
              f"{args.outlier_csv}  ({len(outliers)} rows)")

        # ── Summary tables ──
        print(f"\n=== Per-zone heating diagnostics (excluding real<{args.min_real_mwh:g} MWh/yr) ===")
        df_h = df[~df["low_signal_heat"]]
        n_dropped_h = df.groupby("zone").apply(
            lambda g: int(g["low_signal_heat"].sum()))
        agg_h = df_h.groupby("zone").agg(
            n_used=("bldg_id","count"),
            heat_r_med=("r_q_heat_w","median"),
            heat_rmse_med_kw=("rmse_q_heat_w", lambda x: x.median()/1000),
            heat_rmse_mean_kw=("rmse_q_heat_w", lambda x: x.mean()/1000),
            heat_rmse_max_kw=("rmse_q_heat_w", lambda x: x.max()/1000),
            heat_real_kwh_m2_med=("heat_real_kwh_m2","median"),
            heat_sim_kwh_m2_med=("heat_sim_kwh_m2","median"),
            heat_outliers=(
                "heat_ratio", lambda x: int((x > args.outlier_ratio).sum())
            ),
        ).round(2)
        agg_h["n_dropped_lowsig"] = n_dropped_h
        print(agg_h.to_string())

        print(f"\n=== Per-zone cooling diagnostics (excluding real<{args.min_real_mwh:g} MWh/yr) ===")
        df_c = df[~df["low_signal_cool"]]
        n_dropped_c = df.groupby("zone").apply(
            lambda g: int(g["low_signal_cool"].sum()))
        agg_c = df_c.groupby("zone").agg(
            n_used=("bldg_id","count"),
            cool_r_med=("r_q_cool_w","median"),
            cool_rmse_med_kw=("rmse_q_cool_w", lambda x: x.median()/1000),
            cool_rmse_mean_kw=("rmse_q_cool_w", lambda x: x.mean()/1000),
            cool_rmse_max_kw=("rmse_q_cool_w", lambda x: x.max()/1000),
            cool_real_kwh_m2_med=("cool_real_kwh_m2","median"),
            cool_sim_kwh_m2_med=("cool_sim_kwh_m2","median"),
            cool_outliers=(
                "cool_ratio", lambda x: int((x > args.outlier_ratio).sum())
            ),
        ).round(2)
        agg_c["n_dropped_lowsig"] = n_dropped_c
        print(agg_c.to_string())


if __name__ == "__main__":
    main()
