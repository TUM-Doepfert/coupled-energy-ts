"""Preprocess raw HTW Berlin three-phase electricity measurements into
the hourly per-household profiles consumed by the pipeline as Record E.

The HTW Berlin dataset (Tjaden et al.) provides minute-resolution active-
power measurements split across three phases per household. This script:

1. Loads ``PL1.csv``, ``PL2.csv``, and ``PL3.csv`` from
   ``input/electricity/raw/``. Each input has no header and each column
   is one household.
2. Reindexes the three phases to a full-year 2010 minute index in the
   ``Europe/Berlin`` timezone.
3. Sums the three phases per household to recover total active power.
4. Resamples to the target temporal resolution (default 60-minute mean).
5. Computes the annual energy demand in kWh for each household.
6. Writes one CSV per household to
   ``input/electricity/{resolution}/{annual_demand_kwh}.csv``.

The hourly outputs of this script are what the pipeline reads via the
HTW electricity provider. Users who supply their own metered residential
electricity data do not need to run this script: they can drop their
preformatted CSVs directly into ``input/electricity/{resolution}/`` and
the pipeline picks them up. This script documents how the published
Germany 2010 Record E was produced from the raw HTW Berlin download.

Usage:
    uv run python scripts/preprocess_electricity_htw.py
    uv run python scripts/preprocess_electricity_htw.py --resolution 15min
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_RAW_DIR = ROOT / "input" / "electricity" / "raw"
DEFAULT_OUT_BASE = ROOT / "input" / "electricity"
DEFAULT_RESOLUTION = "60min"
DEFAULT_YEAR = 2010


def load_three_phases(raw_dir: Path, year: int) -> pd.DataFrame:
    """Load PL1, PL2, PL3 CSVs and return their element-wise sum.

    Each input CSV has no header and one column per household. All three
    phases are reindexed to a shared minute timestamp index for the given
    year in the ``Europe/Berlin`` timezone, then summed to recover the
    total active power per household.
    """
    full_index = pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31 23:59:00",
        freq="1min",
        tz="Europe/Berlin",
    )
    phases: dict[str, pd.DataFrame] = {}
    for phase in ("PL1", "PL2", "PL3"):
        df = pd.read_csv(raw_dir / f"{phase}.csv", header=None)
        df = df.set_index(full_index)
        phases[phase] = df
    return phases["PL1"] + phases["PL2"] + phases["PL3"]


def resample_and_save(p_total_w: pd.DataFrame,
                       out_dir: Path,
                       resolution: str) -> None:
    """Resample minute-resolution total power to ``resolution`` and write
    one CSV per household, named by integer annual demand in kWh.

    Annual demand is computed as the sum of resampled mean power values
    scaled by the resampling-step duration in hours, divided by 1000.
    """
    p_resampled_w = (
        p_total_w.resample(resolution).mean().round().astype(int)
    )

    hours_per_step = pd.Timedelta(resolution) / pd.Timedelta("1h")
    demand_kwh = (
        p_resampled_w.sum() * hours_per_step / 1e3
    ).round().astype(int)
    p_resampled_w.columns = demand_kwh

    out_dir.mkdir(parents=True, exist_ok=True)
    for col in p_resampled_w.columns:
        path = out_dir / f"{col}.csv"
        p_resampled_w[col].to_csv(path, index=True)
    print(f"[preprocess_electricity_htw] wrote {len(p_resampled_w.columns)} "
          f"profiles to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help=f"Directory containing PL1.csv, PL2.csv, PL3.csv "
                        f"(default: {DEFAULT_RAW_DIR})")
    p.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE,
                   help=f"Output base directory. The resolution string is "
                        f"appended as a subdirectory "
                        f"(default: {DEFAULT_OUT_BASE})")
    p.add_argument("--resolution", default=DEFAULT_RESOLUTION,
                   help=f"Resampling resolution as a pandas frequency string, "
                        f"e.g. '15min' or '60min' "
                        f"(default: {DEFAULT_RESOLUTION})")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR,
                   help=f"Calendar year of the raw HTW Berlin data "
                        f"(default: {DEFAULT_YEAR})")
    args = p.parse_args()

    print(f"[preprocess_electricity_htw] loading three phases from "
          f"{args.raw_dir} ...")
    p_total_w = load_three_phases(args.raw_dir, args.year)
    print(f"[preprocess_electricity_htw] {p_total_w.shape[1]} households, "
          f"{p_total_w.shape[0]} minute timestamps")

    out_dir = args.out_base / args.resolution
    resample_and_save(p_total_w, out_dir, args.resolution)


if __name__ == "__main__":
    main()
