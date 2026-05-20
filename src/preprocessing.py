from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
from entise.methods.occupancy import geoma


E_COLUMNS = ["timestamp", "profile_id", "electricity_demand"]
O_COLUMNS = ["timestamp", "profile_id", "occupied"]
PROFILE_MAPPING_COLUMNS = ["profile_id", "source_file", "annual_demand_kwh"]


@dataclass(frozen=True)
class BuildResult:
    rows: int
    profiles: int
    output_path: Path


def require_parquet_engine() -> None:
    if find_spec("pyarrow") is None and find_spec("fastparquet") is None:
        raise RuntimeError(
            "Writing Parquet requires pyarrow or fastparquet. Run `uv sync` after adding "
            "the dependency, or install `pyarrow` in the active environment."
        )


def read_electricity_csv(path: Path) -> pd.DataFrame:
    """Read one HTW Berlin profile CSV.

    The hourly (60min) source files span the DST transition with mixed
    offsets (CET +01:00 in winter, CEST +02:00 in summer), which trips
    ``parse_dates=True`` and yields an object-dtype index. We pass the
    timestamps through ``pd.to_datetime(..., utc=True)`` to coerce both
    offsets onto a uniform UTC index — this preserves the absolute
    instant of every observation and lets downstream code call
    ``pd.to_datetime`` on the column without errors.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] != 1:
        raise ValueError(f"Expected one power column in {path}, found {df.shape[1]}")
    df.index = pd.to_datetime(df.index, utc=True)

    return (
        df.rename(columns={df.columns[0]: "electricity_demand"})
        .rename_axis("timestamp")
        .reset_index()
        .loc[:, ["timestamp", "electricity_demand"]]
    )


def electricity_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.csv"), key=lambda p: (int(p.stem), p.name))
    if not files:
        raise FileNotFoundError(f"No CSV electricity profiles found in {input_dir}")
    return files


def build_electricity_record(
    input_dir: Path = Path("input/electricity/60min"),
    output_path: Path = Path("output/E.parquet"),
    mapping_path: Path = Path("output/profile_mapping.csv"),
) -> BuildResult:
    require_parquet_engine()

    frames = []
    mapping_rows = []
    for profile_id, path in enumerate(electricity_files(input_dir), start=1):
        profile = read_electricity_csv(path)
        profile.insert(1, "profile_id", profile_id)
        frames.append(profile)
        mapping_rows.append(
            {
                "profile_id": profile_id,
                "source_file": path.name,
                "annual_demand_kwh": int(path.stem),
            }
        )

    record = pd.concat(frames, ignore_index=True).loc[:, E_COLUMNS]
    mapping = pd.DataFrame(mapping_rows, columns=PROFILE_MAPPING_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    record.to_parquet(output_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    return BuildResult(rows=len(record), profiles=len(mapping), output_path=output_path)


DEFAULT_LOCAL_TZ = "Europe/Berlin"


def _apply_paper_night_rule(occupied: pd.Series,
                            local_tz: str = DEFAULT_LOCAL_TZ) -> pd.Series:
    """Extend evening occupancy into the next morning.

    Paper rule: if occupancy is detected for at least one hour between
    21:00 and 23:59 LOCAL TIME on day d, the household is assumed
    occupied from 00:00 until 09:00 LOCAL TIME of day d+1.

    The electricity readers coerce CSVs to a UTC index to handle DST,
    so the rule is evaluated by temporarily converting the series to
    ``local_tz``. The returned series carries the original (UTC) index
    so downstream pipeline steps remain timezone-consistent. Default
    ``local_tz="Europe/Berlin"`` matches the Germany 2010 reference
    dataset; pass a different IANA zone for other countries.

    The "one hour" threshold is converted into a number of samples based
    on the index step, so the rule fires correctly at hourly, 30-min,
    15-min, etc. resolutions. Previously a hard-coded ``>= 4`` was used,
    which silently disabled the rule at hourly resolution because three
    binary samples (hours 21, 22, 23) can never sum to 4.
    """
    idx_orig = pd.DatetimeIndex(occupied.index)
    if idx_orig.tz is None:
        idx_local = idx_orig.tz_localize("UTC").tz_convert(local_tz)
    else:
        idx_local = idx_orig.tz_convert(local_tz)

    result_local = pd.Series(occupied.values, index=idx_local).copy()

    # Number of samples that represents one hour at the current resolution.
    if len(idx_local) < 2:
        threshold = 1
    else:
        step_hours = (idx_local[1] - idx_local[0]).total_seconds() / 3600.0
        threshold = max(1, int(round(1.0 / step_hours))) if step_hours > 0 else 1

    evening = result_local[(idx_local.hour >= 21) & (idx_local.hour <= 23)]
    for day, day_values in evening.groupby(evening.index.date):
        if day_values.sum() >= threshold:
            start = pd.Timestamp(day, tz=local_tz) + pd.Timedelta(days=1)
            end = start + pd.Timedelta(hours=9)
            local_idx = result_local.index
            result_local.loc[(local_idx >= start) & (local_idx < end)] = 1

    # Restore the original (UTC) index so callers see a timezone-consistent series.
    return pd.Series(result_local.values, index=idx_orig).astype("int8")


def calculate_occupancy(profile: pd.DataFrame, lambda_occ: float = 0.05,
                        local_tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    electricity = (
        profile.set_index("timestamp")
        .rename(columns={"electricity_demand": Columns.POWER})
        .loc[:, [Columns.POWER]]
    )
    obj = {
        Objects.LAMBDA: lambda_occ,
        Objects.NIGHT_SCHEDULE: False,
        Objects.NIGHT_SCHEDULE_START: 21,
        Objects.NIGHT_SCHEDULE_END: 23,
    }
    raw = geoma.calculate_timeseries(obj, {Types.ELECTRICITY: electricity})
    occupied = _apply_paper_night_rule(raw[Objects.OCCUPANCY], local_tz=local_tz)
    return occupied.rename("occupied").rename_axis("timestamp").reset_index()


def build_occupancy_record(
    electricity_path: Path = Path("output/E.parquet"),
    output_path: Path = Path("output/O.parquet"),
    lambda_occ: float = 0.05,
    local_tz: str = DEFAULT_LOCAL_TZ,
) -> BuildResult:
    require_parquet_engine()

    electricity = pd.read_parquet(electricity_path)
    missing = set(E_COLUMNS) - set(electricity.columns)
    if missing:
        raise ValueError(f"{electricity_path} is missing required columns: {sorted(missing)}")

    frames = []
    for profile_id, profile in electricity.groupby("profile_id", sort=True):
        occupancy = calculate_occupancy(
            profile.loc[:, E_COLUMNS],
            lambda_occ=lambda_occ,
            local_tz=local_tz,
        )
        occupancy.insert(1, "profile_id", int(profile_id))
        frames.append(occupancy)

    record = pd.concat(frames, ignore_index=True).loc[:, O_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record.to_parquet(output_path, index=False)

    return BuildResult(rows=len(record), profiles=record["profile_id"].nunique(), output_path=output_path)
