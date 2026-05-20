"""Step 2: download per-building parquets and county weather CSVs from OEDI.

Reads the qualified-buildings CSVs produced by buildings_select.py, then
downloads each building's raw EULP parquet directly from the OEDI S3 bucket
(no Selenium scraping, no pagination — direct HTTPS GETs by predictable URL).

Parallelism: ThreadPoolExecutor with --workers (default 12). Network-bound,
threads are appropriate.

Run from the project root:
    uv run python validation/us/download_data.py
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES, HOUSEHOLD_URL_FMT, WEATHER_URL_FMT


def download_one(url: str, dest: Path, timeout: int = 60, retries: int = 3) -> tuple[Path, str]:
    if dest.exists() and dest.stat().st_size > 1024:
        return dest, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest, "ok"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    return dest, f"fail: {last_err}"


def download_county(
    county_id: str, state: str, zone: str,
    qualified_csv: Path, raw_dir: Path, weather_dir: Path,
    workers: int = 12,
) -> dict:
    sub = pd.read_csv(qualified_csv)
    bldg_ids = sub["bldg_id"].astype(int).tolist()
    out_county_dir = raw_dir / f"{county_id}_{zone.replace(' ', '_')}"
    out_county_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{county_id} {zone}] {len(bldg_ids)} buildings -> {out_county_dir}")
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for bid in bldg_ids:
            url = HOUSEHOLD_URL_FMT.format(state=state, bldg_id=bid)
            dest = out_county_dir / f"{bid}-0.parquet"
            futures[ex.submit(download_one, url, dest)] = bid
        for i, fut in enumerate(as_completed(futures), 1):
            bid = futures[fut]
            _, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  bldg {bid}: {status}")
            if i % 25 == 0 or i == len(bldg_ids):
                print(f"  {i}/{len(bldg_ids)} ({time.time()-t0:.0f}s)")

    # Weather
    w_url = WEATHER_URL_FMT.format(state=state, county_id=county_id)
    w_dest = weather_dir / f"{county_id}_2018.csv"
    _, w_status = download_one(w_url, w_dest)
    print(f"  weather {county_id}: {w_status}")
    return {
        "county_id": county_id, "zone": zone,
        "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail,
        "weather_status": w_status,
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
        default=Path("validation/us/data"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--counties", type=str, default=None,
        help="Comma-separated county IDs; default = all 6")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    weather_dir = args.data_dir / "weather"
    raw_dir.mkdir(parents=True, exist_ok=True)
    weather_dir.mkdir(parents=True, exist_ok=True)

    county_filter = set(args.counties.split(",")) if args.counties else None

    summary = []
    for county_id, state, zone, _expected, _label in COUNTIES:
        if county_filter and county_id not in county_filter:
            continue
        zone_safe = zone.replace(" ", "_")
        qcsv = args.data_dir / f"{county_id}_{zone_safe}_qualified.csv"
        if not qcsv.exists():
            print(f"[skip] {qcsv} not found - run buildings_select.py first")
            continue
        summary.append(download_county(
            county_id, state, zone, qcsv, raw_dir, weather_dir, args.workers
        ))

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(args.data_dir / "download_summary.csv", index=False)
        print("\n=== Download summary ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
