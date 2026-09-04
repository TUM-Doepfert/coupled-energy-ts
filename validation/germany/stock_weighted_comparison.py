"""Stock-weighted ABSOLUTE magnitude comparison against When2Heat.

Why this exists
---------------
``demand_comparison.py`` compares peak-NORMALISED shapes, because the raw
aggregate weights all 11 TABULA vintages and all 4 045 grid cells equally.
Reviewers 2 and 3 both objected that no absolute-magnitude claim can rest
on that. This script consumes the factorised output of
``aggregate_by_archetype.py`` and applies the archetype stock mix, so the
national series carries physical MW.

    Q(t) = N_total * sum_a f_a * A_a(t)

``A_a(t)`` is the spatially averaged demand of one dwelling of archetype
``a`` (W); ``f_a`` is that vintage's share of the SFH stock; ``N_total``
is the number of German single-family dwellings.

Spatial weights stay UNIFORM here: per-cell building counts are a separate
sourcing problem and are not resolved. So this run removes the archetype-mix
bias only.

Note on ``demand_comparison.compute_metrics``: it peak-normalises BOTH
series before taking the ratio of sums, so its ``energy_ratio_norm`` is a
peakiness statistic, not an energy bias. The absolute ratio here is
computed on unnormalised series and is a different quantity.

Usage
-----
    uv run python validation/germany/stock_weighted_comparison.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.germany.demand_comparison import load_when2heat  # noqa: E402

N_TOTAL_DEFAULT = 9.976e6          # German single-family dwellings


def load_archetype_weights(path: Path) -> pd.Series:
    """Read ``archetype_id -> share`` from the TABULA weights CSV.

    Historical note: an earlier revision of this CSV left the comma inside
    the ``source`` field unquoted, so most rows carried 8 fields against a
    7-name header. ``pd.read_csv`` absorbed that by promoting the first
    column to the index, shifting every column by one and turning ``share``
    into text - an all-NaN weight vector with no error raised. The file has
    since been corrected (quoted fields, ``year_range`` split into
    ``year_from``/``year_to``), so a plain read is correct again. The
    sum-to-one assertion stays: it is what would have caught the old bug.
    """
    df = pd.read_csv(path)
    missing = {"archetype_id", "share"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    s = df.set_index("archetype_id")["share"].astype(float).sort_index()
    s.name = "share"
    if not np.isclose(s.sum(), 1.0, atol=1e-4):
        raise ValueError(f"{path}: shares sum to {s.sum():.6f}, expected 1.0")
    return s


def metrics_absolute(ours: pd.Series, theirs: pd.Series) -> dict:
    """Unnormalised comparison metrics.

    * ``energy_ratio`` - sum(ours)/sum(theirs) on the RAW series. This is the
      energy bias, and is exactly what ``compute_metrics.energy_ratio_norm``
      is not.
    * Pearson r is scale-invariant, so it matches the normalised figure.
    * nRMSE is reported against both the When2Heat mean and its peak, since
      "normalised RMSE" is ambiguous and the two differ by ~3x here.
    """
    common = ours.index.intersection(theirs.index)
    if len(common) == 0:
        raise ValueError("No timestamp overlap between pipeline and When2Heat.")
    o, t = ours.loc[common], theirs.loc[common]
    o_d, t_d = o.resample("D").mean(), t.resample("D").mean()
    rmse_h = float(np.sqrt(((o - t) ** 2).mean()))
    rmse_d = float(np.sqrt(((o_d - t_d) ** 2).mean()))
    return {
        "n_hours": int(len(common)),
        "annual_ours_twh": float(o.sum()) / 1e6,        # MWh -> TWh
        "annual_when2heat_twh": float(t.sum()) / 1e6,
        "energy_ratio": float(o.sum() / t.sum()),
        "pearson_r_daily": float(pearsonr(o_d.values, t_d.values)[0]),
        "pearson_r_hourly": float(pearsonr(o.values, t.values)[0]),
        "nrmse_hourly_vs_mean": rmse_h / float(t.mean()),
        "nrmse_hourly_vs_peak": rmse_h / float(t.max()),
        "nrmse_daily_vs_mean": rmse_d / float(t_d.mean()),
        "peak_ours_mw": float(o.max()),
        "peak_when2heat_mw": float(t.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Path("validation/germany/data")
    p.add_argument("--data-dir", type=Path, default=d)
    p.add_argument("--b-parquet", type=Path, default=Path("output/B.parquet"))
    p.add_argument("--when2heat-csv", type=Path, default=d / "when2heat.csv")
    p.add_argument("--n-total", type=float, default=N_TOTAL_DEFAULT)
    args = p.parse_args()

    series = pd.read_parquet(args.data_dir / "archetype_series.parquet")
    annual = pd.read_parquet(args.data_dir / "annual_energy_by_loc_arch.parquet")
    b = pd.read_parquet(args.b_parquet)[["archetype_id", "area_m2"]]
    w = load_archetype_weights(args.data_dir / "archetype_weights_tabula_de.csv")

    arch_ids = sorted(int(c[4:6]) for c in series.columns if c.endswith("_uniform"))
    A = series[[f"arch{a:02d}_uniform" for a in arch_ids]].copy()
    A.columns = arch_ids
    print(f"[stock] A_a(t): {A.shape[0]} timestamps x {A.shape[1]} archetypes, W/dwelling")

    f = w.reindex(arch_ids).astype(float)
    if f.isna().any():
        raise ValueError(f"no stock share for archetypes {list(f.index[f.isna()])}")
    print(f"[stock] archetype shares sum to {f.sum():.6f}")
    f_uniform = pd.Series(1.0 / len(arch_ids), index=arch_ids)

    # ---------------- Task 2: per-archetype specific demand -----------------
    area = b.set_index("archetype_id")["area_m2"]
    ann = annual.merge(b, on="archetype_id", how="left")
    if ann["area_m2"].isna().any():
        raise ValueError("archetype_id present in HC tree but missing from B.parquet")
    ann["kwh_per_m2"] = ann["annual_kwh_per_dwelling"] / ann["area_m2"]
    spec = (ann.groupby("archetype_id")["kwh_per_m2"]
              .agg(["median", "min", "max", "count"]).round(1))
    spec.insert(0, "area_m2", area.reindex(spec.index).values)
    print("\n[stock] specific demand kWh/(m2 a) across locations:")
    print(spec.to_string())

    # ---------------- national series, MW -----------------------------------
    def national(fw: pd.Series) -> pd.Series:
        s = (A * fw).sum(axis=1) * args.n_total / 1e6
        return s.sort_index()

    q_weighted = national(f)
    q_uniform = national(f_uniform)

    # ---------------- Task 3: When2Heat --------------------------------------
    theirs, col = load_when2heat(args.when2heat_csv, country="DE", year=2010,
                                 column="DE_heat_demand_space_SFH")
    print(f"[stock] When2Heat column {col}: {len(theirs)} h, "
          f"{theirs.sum()/1e6:.1f} TWh, peak {theirs.max():.0f} MW")

    m_w = metrics_absolute(q_weighted, theirs)
    m_u = metrics_absolute(q_uniform, theirs)

    # ---------------- implied stock aggregates --------------------------------
    e_a = A.sum(axis=0) / 1000.0                       # kWh/a per dwelling, per arch

    def implied(fw: pd.Series) -> dict:
        energy_kwh = args.n_total * float((fw * e_a).sum())
        floor_m2 = args.n_total * float((fw * area.reindex(arch_ids)).sum())
        return {"twh": energy_kwh / 1e9,
                "floor_area_bn_m2": floor_m2 / 1e9,
                "specific_kwh_m2": energy_kwh / floor_m2}

    imp_w, imp_u = implied(f), implied(f_uniform)

    print("\n[stock] implied stock aggregates:")
    for name, imp in (("weighted", imp_w), ("uniform", imp_u)):
        print(f"  {name:9s} {imp['twh']:7.1f} TWh/a   "
              f"{imp['floor_area_bn_m2']:.3f} bn m2   "
              f"{imp['specific_kwh_m2']:6.1f} kWh/(m2 a)")
    print(f"\n[stock] weighting effect on national total: "
          f"{100 * (q_weighted.sum() / q_uniform.sum() - 1):+.1f} %")

    print("\n[stock] absolute metrics vs When2Heat:")
    keys = ["annual_ours_twh", "annual_when2heat_twh", "energy_ratio",
            "pearson_r_daily", "pearson_r_hourly", "nrmse_hourly_vs_mean",
            "nrmse_hourly_vs_peak", "nrmse_daily_vs_mean",
            "peak_ours_mw", "peak_when2heat_mw"]
    print(f"  {'metric':24s} {'weighted':>14s} {'uniform':>14s}")
    for k in keys:
        print(f"  {k:24s} {m_w[k]:14.4f} {m_u[k]:14.4f}")

    spec.to_csv(args.data_dir / "specific_demand_by_archetype.csv")
    pd.DataFrame({"weighted": m_w, "uniform": m_u}).to_csv(
        args.data_dir / "stock_weighted_metrics.csv")
    pd.DataFrame({"weighted_mw": q_weighted, "uniform_mw": q_uniform,
                  "when2heat_mw": theirs.reindex(q_weighted.index)}
                 ).to_parquet(args.data_dir / "national_series_mw.parquet")
    print(f"\n[stock] wrote specific_demand_by_archetype.csv, "
          f"stock_weighted_metrics.csv, national_series_mw.parquet -> {args.data_dir}")


if __name__ == "__main__":
    main()
