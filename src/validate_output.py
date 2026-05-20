"""Sanity-check the H/C output Parquet files.

Runs a battery of structural and physical checks on the generated dataset.
By default checks all files found under output/HC/; use --sample N to check
only N randomly selected files for a quick smoke test.

Usage:
    coupled-energy-ts validate-output
    coupled-energy-ts validate-output --sample 20
    coupled-energy-ts validate-output --output-dir output --verbose
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

# Expected schema
EXPECTED_COLUMNS = ["timestamp", "profile_id", "q_heat_w", "q_cool_w"]
# Derive the expected profile_id set from E.parquet at runtime so the
# script generalises to any country instantiation, not just Germany 2010.
EXPECTED_PROFILE_IDS: set[int] | None = None  # populated by _expected_profile_ids()

def _expected_profile_ids(electricity_parquet: Path) -> set[int]:
    """Read the released electricity record and return the set of profile_ids.

    Generalises the legacy hard-coded ``set(range(1, 75))`` so the validator
    works for any country/run, not only the Germany 2010 reference dataset.
    """
    import pandas as _pd
    return set(int(x) for x in _pd.read_parquet(electricity_parquet,
                                                 columns=["profile_id"])["profile_id"].unique())


EXPECTED_ROWS_60MIN = 8760 * 74                    # hourly year × 74 profiles
EXPECTED_ROWS_15MIN = 35040 * 74                   # 15-min year × 74 profiles

# Seasonal checks (Northern Hemisphere, month numbers)
WINTER_MONTHS = {12, 1, 2}
SUMMER_MONTHS = {6, 7, 8}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = PASS if ok else (WARN if warn_only else FAIL)
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok or warn_only


def validate_file(path: Path, verbose: bool = False) -> tuple[int, int]:
    """Validate one HC Parquet file. Returns (passes, failures)."""
    passes = failures = 0

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [{FAIL}] Could not read file: {e}")
        return 0, 1

    def record(ok: bool, warn_only: bool = False) -> None:
        nonlocal passes, failures
        if ok or warn_only:
            passes += 1
        else:
            failures += 1

    # 1. Schema
    ok = list(df.columns) == EXPECTED_COLUMNS
    record(check("columns", ok, f"got {df.columns.tolist()}" if not ok else ""))

    # 2. Profile IDs
    expected_ids = EXPECTED_PROFILE_IDS if EXPECTED_PROFILE_IDS is not None else set()
    actual_ids = set(int(x) for x in df["profile_id"].unique())
    ok = actual_ids == expected_ids
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    record(check(f"profile_ids match E.parquet ({len(expected_ids)} expected)", ok,
                 f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}" if not ok else ""))

    # 3. Row count (allow 60-min or 15-min)
    n = len(df)
    ok_rows = n in (EXPECTED_ROWS_60MIN, EXPECTED_ROWS_15MIN)
    record(check("row count", ok_rows,
                 f"{n} (expected {EXPECTED_ROWS_60MIN} or {EXPECTED_ROWS_15MIN})" if not ok_rows else f"{n}",
                 warn_only=not ok_rows))

    # 4. Non-negative values
    ok_heat = (df["q_heat_w"] >= 0).all()
    ok_cool = (df["q_cool_w"] >= 0).all()
    record(check("q_heat_w >= 0", ok_heat,
                 f"{(df['q_heat_w'] < 0).sum()} negative rows" if not ok_heat else ""))
    record(check("q_cool_w >= 0", ok_cool,
                 f"{(df['q_cool_w'] < 0).sum()} negative rows" if not ok_cool else ""))

    # 5. Mutual exclusion. Under R1C1 with a single fixed setpoint pair,
    # heating and cooling cannot be simultaneously positive at any hour.
    # Under R5C1/R7C2 with multi-node thermal mass or schedule-driven
    # runs, transient corner cases could in principle produce both > 0;
    # the check below is correct for the released R1C1 dataset and is a
    # diagnostic warning (rather than a hard error) for higher-fidelity
    # models.
    both = ((df["q_heat_w"] > 0) & (df["q_cool_w"] > 0)).sum()
    ok = both == 0
    record(check("no simultaneous heat+cool (R1C1 thermostat constraint)",
                 ok,
                 f"{both} hours with both > 0 — expected zero under R1C1"
                 if not ok else ""))

    # 6. Seasonal sense (heating dominant in winter, cooling plausible in summer)
    ts = pd.to_datetime(df["timestamp"])
    winter_mask = ts.dt.month.isin(WINTER_MONTHS)
    summer_mask = ts.dt.month.isin(SUMMER_MONTHS)
    mean_heat_winter = df.loc[winter_mask, "q_heat_w"].mean()
    mean_heat_summer = df.loc[summer_mask, "q_heat_w"].mean()
    ok_season = mean_heat_winter > mean_heat_summer
    record(check("winter heating > summer heating", ok_season,
                 f"winter={mean_heat_winter:.1f}W summer={mean_heat_summer:.1f}W" if verbose or not ok_season else ""))

    mean_cool_summer = df.loc[summer_mask, "q_cool_w"].mean()
    mean_cool_winter = df.loc[winter_mask, "q_cool_w"].mean()
    ok_cool_season = mean_cool_summer >= mean_cool_winter
    record(check("summer cooling >= winter cooling", ok_cool_season,
                 f"summer={mean_cool_summer:.1f}W winter={mean_cool_winter:.1f}W" if verbose or not ok_cool_season else ""))

    # 7. No all-zero file (at least some heating demand)
    any_heat = (df["q_heat_w"] > 0).any()
    record(check("has non-zero heating", any_heat, "all zeros — possible simulation failure"))

    # 8. Timestamp completeness (no duplicate timestamps per profile)
    dup = df.groupby("profile_id")["timestamp"].apply(lambda s: s.duplicated().sum())
    total_dups = int(dup.sum())
    ok = total_dups == 0
    record(check("no duplicate timestamps", ok, f"{total_dups} duplicates" if not ok else ""))

    return passes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate H/C output Parquet files.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--sample", type=int, default=None,
                        help="Check only N randomly selected files (default: all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample selection.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print seasonal statistics for every file.")
    args = parser.parse_args()
    run_validation(args)


def run_validation(args) -> None:
    """Programmatic entry point used by ``src/cli.py``."""
    global EXPECTED_PROFILE_IDS
    e_pq = args.output_dir / "E.parquet"
    if e_pq.exists():
        EXPECTED_PROFILE_IDS = _expected_profile_ids(e_pq)
        print(f"Expected profile_ids derived from {e_pq.name}: "
              f"{min(EXPECTED_PROFILE_IDS)}-{max(EXPECTED_PROFILE_IDS)} "
              f"({len(EXPECTED_PROFILE_IDS)} profiles)")
    else:
        print(f"Warning: {e_pq} not found; falling back to set(range(1, 75)).")
        EXPECTED_PROFILE_IDS = set(range(1, 75))

    hc_dir = args.output_dir / "HC"
    files = sorted(hc_dir.rglob("hc_arch*.parquet"))

    if not files:
        print(f"No HC Parquet files found under {hc_dir}")
        sys.exit(1)

    if args.sample is not None and args.sample < len(files):
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, args.sample))
        print(f"Checking {len(files)} randomly sampled files (seed={args.seed}).")
    else:
        print(f"Checking all {len(files)} files.")

    total_pass = total_fail = 0
    failed_files: list[Path] = []

    for path in files:
        print(f"\n{path.relative_to(args.output_dir)}")
        p, f = validate_file(path, verbose=args.verbose)
        total_pass += p
        total_fail += f
        if f:
            failed_files.append(path)

    print(f"\n{'='*60}")
    print(f"Files checked : {len(files)}")
    print(f"Checks passed : {total_pass}")
    print(f"Checks failed : {total_fail}")
    if failed_files:
        print(f"\nFiles with failures ({len(failed_files)}):")
        for p in failed_files:
            print(f"  {p}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
