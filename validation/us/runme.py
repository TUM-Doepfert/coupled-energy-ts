"""US per-building validation — single-script orchestrator.

Runs all four steps in order:

    1. buildings_select.py            metadata download + qualified-SFH selection
    2. download_data.py               parallel download of EULP per-building parquets
    3. convert.py                     NREL EULP -> our schema
    4a. building_simulate.py            re-sim each EULP building with our pipeline
    4b. building_comparison.py             RMSE / Pearson r per building
    4c. timeseries_comparison.py  6-panel paper figure

Each step is idempotent on a per-output basis. Steps 1-3 build a ~12 GB
local cache under ``data/``; pass ``--skip-download`` to reuse it.

Usage:
    cd validation/us
    python runme.py [--skip-download]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

STEPS_PRE = [
    "buildings_select.py",
    "download_data.py",
    "convert.py",
]

STEPS_TIER1A = [
    "building_simulate.py",
    "building_comparison.py",
    "timeseries_comparison.py",
    "validation_map.py",
]


def _run(script: str, *extra_args: str) -> int:
    print()
    print("=" * 70)
    print(f"  $ python {script}")
    print("=" * 70)
    return subprocess.call([sys.executable, str(HERE / script), *extra_args], cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip steps 1-3 (assumes data/ cache already populated).")
    args = ap.parse_args()

    if not args.skip_download:
        for s in STEPS_PRE:
            if _run(s):
                sys.exit(f"step '{s}' failed")
    else:
        print("[demand] --skip-download: reusing existing data/ cache.")

    for s in STEPS_TIER1A:
        # building_simulate needs --county all to cover every zone
        extra = ("--county", "all") if s == "building_simulate.py" else ()
        if _run(s, *extra):
            sys.exit(f"step '{s}' failed")


if __name__ == "__main__":
    main()
