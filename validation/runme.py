"""Top-level validation orchestrator.

Convenience wrapper that runs the two per-tier entry points in order:

  US tier (per-building + load-diversity, NREL EULP)
      validation/us/runme.py
      -> img/us_building_comparison.{png,pdf}
         img/us_timeseries_comparison.{png,pdf}
         img/us_validation_map.{png,pdf}

  Germany tier (When2Heat aggregate cross-validation)
      validation/germany/demand_comparison.py
      -> img/de_demand_comparison.{png,pdf}

The load-diversity figure (img/load_diversity.{png,pdf}) is NOT run by
default — it needs the German pipeline outputs under ../output/. Invoke
it directly with:

      uv run python validation/us/diversity_comparison.py

Usage
-----
    cd validation
    python runme.py                     # both tiers in order
    python runme.py --only us
    python runme.py --only germany
    python runme.py --skip-download     # forwarded to us/runme.py

Prerequisites
-------------
  - US tier needs ~12 GB of NREL EULP data; the first run downloads it.
  - Germany tier needs the German pipeline outputs in ../output/ and
    the When2Heat CSV at germany/data/when2heat.csv. See
    ../README.md for the canonical pipeline run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
US = HERE / "us"
GERMANY = HERE / "germany"


def _run_script(script: Path, *args: str) -> int:
    cmd = [sys.executable, str(script), *args]
    print()
    print("=" * 70)
    print(f"  $ {' '.join(cmd)}")
    print("=" * 70)
    return subprocess.call(cmd, cwd=HERE.parent)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", choices=("us", "germany"),
                   help="Run only the named tier and skip the other.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the NREL select+download+convert prep in the US tier.")
    args = p.parse_args()

    if args.only is None or args.only == "us":
        extra = ("--skip-download",) if args.skip_download else ()
        if rc := _run_script(US / "runme.py", *extra):
            sys.exit(f"us runme failed with code {rc}")

    if args.only is None or args.only == "germany":
        if rc := _run_script(GERMANY / "demand_comparison.py"):
            sys.exit(f"germany demand_comparison failed with code {rc}")


if __name__ == "__main__":
    main()
