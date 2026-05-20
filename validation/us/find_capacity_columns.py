"""Diagnostic helper. Not part of the published pipeline.

Quick scan: list every metadata/qualified-CSV column that might hold
HVAC capacity, with example values, so we can wire the right field into
SimInputs for the US per-building capacity-cap update.

Usage:
    uv run python validation/us/find_capacity_columns.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "demand" / "data"

KEYWORDS = ("capac", "btu", "size", "kw", "ton", "watt", "nominal", "rated")


def show_columns(df: pd.DataFrame, source: str) -> None:
    cols = [c for c in df.columns if any(k in c.lower() for k in KEYWORDS)]
    print(f"\n[{source}]  matched {len(cols)} columns out of {len(df.columns)}:")
    for c in cols:
        sample = df[c].dropna()
        ex = sample.iloc[0] if len(sample) else "<all NaN>"
        nuniq = sample.nunique() if len(sample) else 0
        print(f"  {c:<55s} ex={ex!r:<35s} uniq={nuniq}")
    if not cols:
        print("  (no matches)")


def main() -> None:
    md = DATA / "metadata.parquet"
    if md.exists():
        df = pd.read_parquet(md).reset_index()
        show_columns(df, f"metadata.parquet  ({len(df)} rows, {len(df.columns)} cols)")
    else:
        print(f"[skip] {md} not found")

    for csv in sorted(DATA.glob("*_qualified.csv")):
        df = pd.read_csv(csv)
        show_columns(df, f"{csv.name}  ({len(df)} rows, {len(df.columns)} cols)")
        # only need one — they all share schema
        break


if __name__ == "__main__":
    main()
