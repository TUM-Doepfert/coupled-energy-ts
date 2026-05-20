from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .pipeline import run as run_pipeline
from .preprocessing import build_electricity_record, build_occupancy_record
from .providers.archetype import TEASERArchetypeProvider
from .weather import (
    build_location_mapping,
    compact_existing_weather_parquets,
    convert_existing_weather_csvs,
    downsample_existing_weather_parquets_to_hourly,
    fetch_weather_files,
)


@dataclass(frozen=True)
class _CLIResult:
    rows: int
    output_path: Path


def _build_archetypes(archetypes_csv: Path, output_path: Path,
                      mapping_path: Path) -> _CLIResult:
    """CLI handler: generate B.parquet via TEASERArchetypeProvider and
    side-write the archetype_mapping.csv (verbatim copy of the input CSV)."""
    import pandas as pd
    provider = TEASERArchetypeProvider(archetypes_csv=archetypes_csv)
    df = provider.get_archetypes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(archetypes_csv).to_csv(mapping_path, index=False)
    return _CLIResult(rows=len(df), output_path=output_path)


def _build_archetype_mapping(archetypes_csv: Path,
                              output_path: Path) -> _CLIResult:
    """CLI handler: copy the archetype-input CSV to output/archetype_mapping.csv."""
    import pandas as pd
    df = pd.read_csv(archetypes_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return _CLIResult(rows=len(df), output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="coupled-energy-ts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    electricity = subparsers.add_parser("build-electricity", help="Build output/E.parquet from hourly CSV profiles.")
    electricity.add_argument("--input-dir", type=Path, default=Path("input/electricity/60min"))
    electricity.add_argument("--output", type=Path, default=Path("output/E.parquet"))
    electricity.add_argument("--mapping", type=Path, default=Path("output/profile_mapping.csv"))

    occupancy = subparsers.add_parser("build-occupancy", help="Build output/O.parquet from output/E.parquet.")
    occupancy.add_argument("--electricity", type=Path, default=Path("output/E.parquet"))
    occupancy.add_argument("--output", type=Path, default=Path("output/O.parquet"))
    occupancy.add_argument("--lambda-occ", type=float, default=0.05)
    occupancy.add_argument("--local-tz", type=str, default="Europe/Berlin",
        help="IANA timezone in which the paper night rule (21:00-23:59 -> 00:00-09:00) is evaluated.")

    weather = subparsers.add_parser("fetch-weather", help="Fetch weather Parquet files from Open-Meteo.")
    weather.add_argument("--grid", type=Path, default=Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"))
    weather.add_argument("--output-dir", type=Path, default=Path("input/weather/2010"))
    weather.add_argument("--year", type=int, default=2010)
    weather.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (default 1; raise carefully — free Open-Meteo enforces strict per-minute limits).")
    weather.add_argument("--max-retries", type=int, default=5, help="Retries per location on transient API errors with exponential backoff.")
    weather.add_argument("--limit", type=int, default=None)
    weather.add_argument("--overwrite", action="store_true")
    weather.add_argument("--interpolate-15min", action="store_true", help="Optionally upsample hourly Open-Meteo weather to 15-minute resolution.")
    weather.add_argument("--location-mapping", type=Path, default=Path("output/location_mapping.csv"))

    archetypes = subparsers.add_parser("build-archetypes", help="Build output/B.parquet from TEASER TABULA DE SFH archetypes.")
    archetypes.add_argument("--archetypes", type=Path, default=Path("input/archetypes.csv"))
    archetypes.add_argument("--output", type=Path, default=Path("output/B.parquet"))
    archetypes.add_argument("--archetype-mapping", type=Path, default=Path("output/archetype_mapping.csv"))

    archetype_mapping = subparsers.add_parser("build-archetype-mapping", help="Build output/archetype_mapping.csv from input/archetypes.csv.")
    archetype_mapping.add_argument("--archetypes", type=Path, default=Path("input/archetypes.csv"))
    archetype_mapping.add_argument("--output", type=Path, default=Path("output/archetype_mapping.csv"))

    location_mapping = subparsers.add_parser("build-location-mapping", help="Build output/location_mapping.csv from the German grid.")
    location_mapping.add_argument("--grid", type=Path, default=Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"))
    location_mapping.add_argument("--output", type=Path, default=Path("output/location_mapping.csv"))

    convert_weather = subparsers.add_parser("convert-weather-csvs", help="Convert existing coordinate-named weather CSVs to loc#### Parquet files.")
    convert_weather.add_argument("--csv-dir", type=Path, default=Path("input/weather/2010"))
    convert_weather.add_argument("--grid", type=Path, default=Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"))
    convert_weather.add_argument("--output-dir", type=Path, default=Path("input/weather/2010"))
    convert_weather.add_argument("--overwrite", action="store_true")
    convert_weather.add_argument("--interpolate-15min", action="store_true", help="Optionally upsample hourly weather CSVs to 15-minute resolution.")

    compact_weather = subparsers.add_parser("compact-weather", help="Rewrite existing loc#### weather Parquet files with compact dtypes and zstd compression.")
    compact_weather.add_argument("--weather-dir", type=Path, default=Path("input/weather/2010"))

    hourly_weather = subparsers.add_parser("weather-to-hourly", help="Rewrite existing loc#### weather Parquet files to hourly timestamps.")
    hourly_weather.add_argument("--weather-dir", type=Path, default=Path("input/weather/2010"))

    run_cmd = subparsers.add_parser("run", help="Run the full coupled-time-series pipeline from a YAML config.")
    run_cmd.add_argument("config", type=Path, help="Path to a YAML config (see config/germany_2010.yml).")
    run_cmd.add_argument("--overwrite", action="store_true", help="Re-run steps whose outputs already exist.")
    run_cmd.add_argument("--skip-weather-fetch", action="store_true", help="Skip Open-Meteo download (use cached files).")
    run_cmd.add_argument("--skip-simulation", action="store_true", help="Build B/E/O/W only; do not run the thermal simulation.")

    validate = subparsers.add_parser("validate-output", help="Sanity-check the H/C output parquet files (structural + physical checks).")
    validate.add_argument("--output-dir", type=Path, default=Path("output"))
    validate.add_argument("--sample", type=int, default=None, help="Check only N randomly-sampled files.")
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "build-electricity":
            result = build_electricity_record(args.input_dir, args.output, args.mapping)
        elif args.command == "build-archetype-mapping":
            result = _build_archetype_mapping(args.archetypes, args.output)
        elif args.command == "build-archetypes":
            result = _build_archetypes(args.archetypes, args.output, args.archetype_mapping)
        elif args.command == "build-occupancy":
            result = build_occupancy_record(args.electricity, args.output, args.lambda_occ, args.local_tz)
        elif args.command == "build-location-mapping":
            result = build_location_mapping(args.grid, args.output)
        elif args.command == "fetch-weather":
            result = fetch_weather_files(
                args.grid,
                args.output_dir,
                args.year,
                args.n_jobs,
                args.limit,
                args.overwrite,
                args.max_retries,
                args.interpolate_15min,
                args.location_mapping,
            )
        elif args.command == "convert-weather-csvs":
            result = convert_existing_weather_csvs(
                args.csv_dir,
                args.grid,
                args.output_dir,
                args.overwrite,
                args.interpolate_15min,
            )
        elif args.command == "compact-weather":
            result = compact_existing_weather_parquets(args.weather_dir)
        elif args.command == "weather-to-hourly":
            result = downsample_existing_weather_parquets_to_hourly(args.weather_dir)
        elif args.command == "run":
            result = run_pipeline(
                args.config,
                overwrite=args.overwrite,
                skip_weather_fetch=args.skip_weather_fetch,
                skip_simulation=args.skip_simulation,
            )
        elif args.command == "validate-output":
            from .validate_output import run_validation
            run_validation(args)
            return
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except RuntimeError as exc:
        import sys
        import traceback as _tb
        # Surface the full traceback under --verbose so weather-fetch /
        # EnTiSe failures can be diagnosed; print only the message
        # otherwise (avoid burying the user under a stack trace by default).
        if getattr(args, "verbose", False):
            _tb.print_exc(file=sys.stderr)
        parser.exit(status=1, message=f"error: {exc}\n")

    if args.command == "run":
        print(result)
    elif hasattr(result, "written"):
        print(f"Wrote {result.written} weather files, skipped {result.skipped}, output directory {result.output_dir}")
    elif hasattr(result, "profiles"):
        print(f"Wrote {result.rows:,} rows for {result.profiles} profiles to {result.output_path}")
    else:
        print(f"Wrote {result.rows:,} rows to {result.output_path}")


if __name__ == "__main__":
    main()
