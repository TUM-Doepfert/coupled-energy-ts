from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from entise.services.weather.openmeteo import OpenMeteoProvider
from joblib import Parallel, delayed
from tqdm import tqdm

from .preprocessing import require_parquet_engine


LOCATION_MAPPING_COLUMNS = ["location_id", "latitude", "longitude"]

WEATHER_COLUMNS = [
    "timestamp",
    "location_id",
    "air_temperature",
    "global_horizontal_irradiance",
    "direct_normal_irradiance",
    "diffuse_horizontal_irradiance",
    "wind_speed",
    "relative_humidity",
]
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "wind_speed_10m",
]
WEATHER_FLOAT_COLUMNS = [
    "air_temperature",
    "global_horizontal_irradiance",
    "direct_normal_irradiance",
    "diffuse_horizontal_irradiance",
    "wind_speed",
    "relative_humidity",
]
WEATHER_PARQUET_COMPRESSION = "zstd"
WEATHER_PARQUET_COMPRESSION_LEVEL = 9


@dataclass(frozen=True)
class Location:
    location_id: int
    latitude: float
    longitude: float


@dataclass(frozen=True)
class LocationMappingResult:
    rows: int
    output_path: Path


@dataclass(frozen=True)
class WeatherBuildResult:
    written: int
    skipped: int
    output_dir: Path


def load_grid_locations(grid_path: Path = Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg")) -> pd.DataFrame:
    grid = gpd.read_file(grid_path)
    centroids = grid.geometry.centroid
    points = gpd.GeoDataFrame(
        {"location_id": range(1, len(grid) + 1)},
        geometry=centroids,
        crs=grid.crs,
    ).to_crs("EPSG:4326")
    return pd.DataFrame(
        {
            "location_id": points["location_id"].astype(int),
            "latitude": points.geometry.y.astype(float),
            "longitude": points.geometry.x.astype(float),
        }
    )


def build_location_mapping(
    grid_path: Path = Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"),
    output_path: Path = Path("output/location_mapping.csv"),
) -> LocationMappingResult:
    locations = load_grid_locations(grid_path)[LOCATION_MAPPING_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    locations.to_csv(output_path, index=False)
    return LocationMappingResult(rows=len(locations), output_path=output_path)


def weather_output_path(output_dir: Path, location_id: int) -> Path:
    return output_dir / f"loc{location_id:04d}.parquet"


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        for column in columns:
            if column == candidate or column.startswith(candidate):
                return column
    raise ValueError(f"Could not find any of {candidates} in weather columns: {columns}")


def normalize_weather_frame(df: pd.DataFrame, location_id: int) -> pd.DataFrame:
    source = df.copy()
    columns = source.columns.tolist()

    timestamp_col = _find_column(columns, ("datetime", "timestamp"))
    air_temperature_col = _find_column(columns, ("air_temperature", "temperature_2m"))
    irradiance_col = _find_column(columns, ("global_horizontal_irradiance", "shortwave_radiation"))
    dni_col = _find_column(columns, ("direct_normal_irradiance",))
    dhi_col = _find_column(columns, ("diffuse_horizontal_irradiance", "diffuse_radiation"))
    wind_speed_col = _find_column(columns, ("wind_speed", "wind_speed_10m"))
    humidity_col = _find_column(columns, ("relative_humidity", "relative_humidity_2m"))

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(source[timestamp_col]),
            "location_id": int(location_id),
            "air_temperature": pd.to_numeric(source[air_temperature_col], errors="raise"),
            "global_horizontal_irradiance": pd.to_numeric(source[irradiance_col], errors="raise"),
            "direct_normal_irradiance": pd.to_numeric(source[dni_col], errors="raise"),
            "diffuse_horizontal_irradiance": pd.to_numeric(source[dhi_col], errors="raise"),
            "wind_speed": pd.to_numeric(source[wind_speed_col], errors="raise"),
            "relative_humidity": pd.to_numeric(source[humidity_col], errors="raise"),
        }
    )

    # Heuristic to detect 0–1 vs 0–100 RH: if the *typical* value is sub-2 the
    # column is almost certainly fractional. Mean is more robust than max
    # because legitimate near-zero RH outliers in arid climates can drag the
    # max above 1.5 without changing the underlying scale convention.
    if out["relative_humidity"].mean() < 1.5:
        out["relative_humidity"] *= 100.0

    return out.sort_values("timestamp").reset_index(drop=True)


def interpolate_weather_15min(df: pd.DataFrame) -> pd.DataFrame:
    values = df.set_index("timestamp").sort_index()
    location_id = int(values["location_id"].iloc[0])
    values = values.drop(columns=["location_id"])

    full_index = pd.date_range(values.index.min(), values.index.max() + pd.Timedelta(minutes=45), freq="15min")
    values = values.reindex(full_index).interpolate(method="time").ffill().bfill()
    values.insert(0, "location_id", location_id)
    return values.rename_axis("timestamp").reset_index().loc[:, WEATHER_COLUMNS]


def keep_hourly_weather(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.loc[pd.to_datetime(df["timestamp"]).dt.minute.eq(0)].copy()
    return hourly.reset_index(drop=True).loc[:, WEATHER_COLUMNS]


def compact_weather_frame(df: pd.DataFrame) -> pd.DataFrame:
    compact = df.loc[:, WEATHER_COLUMNS].copy()
    compact["location_id"] = compact["location_id"].astype("int16")
    compact[WEATHER_FLOAT_COLUMNS] = compact[WEATHER_FLOAT_COLUMNS].astype("float32")
    return compact


def write_weather_parquet(df: pd.DataFrame, output_path: Path) -> None:
    compact = compact_weather_frame(df)
    compact.to_parquet(
        output_path,
        index=False,
        compression=WEATHER_PARQUET_COMPRESSION,
        compression_level=WEATHER_PARQUET_COMPRESSION_LEVEL,
    )


def prepare_weather_frame(df: pd.DataFrame, interpolate_15min: bool = False) -> pd.DataFrame:
    if interpolate_15min:
        return interpolate_weather_15min(df)
    return keep_hourly_weather(df)


def fetch_weather(
    location: Location,
    year: int = 2010,
    provider: OpenMeteoProvider | None = None,
    interpolate_15min: bool = False,
) -> pd.DataFrame:
    provider = provider or OpenMeteoProvider()
    raw = provider.get_weather_data(
        latitude=location.latitude,
        longitude=location.longitude,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        timezone="Europe/Berlin",
        features=WEATHER_FEATURES,
    )
    return prepare_weather_frame(normalize_weather_frame(raw, location.location_id), interpolate_15min=interpolate_15min)


_TRANSIENT_MARKERS = (
    "too many", "concurrent", "rate limit", "limit exceeded",
    "minutely", "hourly", "per minute", "per hour", "please try again",
    "timeout", "503", "500", "service unavailable",
)


def _retry_delay(e: Exception, attempt: int) -> float:
    msg = str(e).lower()
    if any(m in msg for m in ("hourly", "per hour", "next hour")):
        return 3660.0 + random.uniform(0, 60)
    if any(m in msg for m in ("minutely", "per minute", "limit exceeded", "please try again")):
        return 65.0 + random.uniform(0, 5)
    return 2 ** attempt + random.uniform(0, 1)


def _fetch_with_retry(
    location: Location,
    year: int,
    max_retries: int,
    provider: OpenMeteoProvider | None = None,
    interpolate_15min: bool = False,
) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            return fetch_weather(location, year=year, provider=provider, interpolate_15min=interpolate_15min)
        except Exception as e:
            is_transient = any(m in str(e).lower() for m in _TRANSIENT_MARKERS)
            if is_transient and attempt < max_retries - 1:
                time.sleep(_retry_delay(e, attempt))
            else:
                raise
    raise RuntimeError("unreachable")


def fetch_weather_file(
    location: Location,
    output_dir: Path,
    year: int = 2010,
    overwrite: bool = False,
    max_retries: int = 5,
    interpolate_15min: bool = False,
) -> bool:
    require_parquet_engine()
    output_path = weather_output_path(output_dir, location.location_id)
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather = _fetch_with_retry(
        location,
        year=year,
        max_retries=max_retries,
        interpolate_15min=interpolate_15min,
    )
    write_weather_parquet(weather, output_path)
    return True


def fetch_weather_files(
    grid_path: Path = Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"),
    output_dir: Path = Path("input/weather/2010"),
    year: int = 2010,
    n_jobs: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
    max_retries: int = 5,
    interpolate_15min: bool = False,
    mapping_path: Path = Path("output/location_mapping.csv"),
) -> WeatherBuildResult:
    locations_df = load_grid_locations(grid_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    locations_df[LOCATION_MAPPING_COLUMNS].to_csv(mapping_path, index=False)
    if limit is not None:
        locations_df = locations_df.head(limit)

    locations = [Location(int(row.location_id), float(row.latitude), float(row.longitude)) for row in locations_df.itertuples()]
    n_total = len(locations)

    results = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator_unordered")(
        delayed(fetch_weather_file)(
            loc,
            output_dir,
            year=year,
            overwrite=overwrite,
            max_retries=max_retries,
            interpolate_15min=interpolate_15min,
        )
        for loc in locations
    )

    written = skipped = 0
    with tqdm(total=n_total, unit="loc", desc="Fetching weather") as pbar:
        for flag in results:
            if flag:
                written += 1
            else:
                skipped += 1
            pbar.update(1)
            pbar.set_postfix(written=written, skipped=skipped)

    return WeatherBuildResult(written=written, skipped=n_total - written, output_dir=output_dir)


def _parse_lat_lon_from_name(path: Path) -> tuple[float, float]:
    match = re.match(r"lat-(-?\d+\.\d+)_lon-(-?\d+\.\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Unexpected weather CSV filename: {path.name}")
    return float(match.group(1)), float(match.group(2))


def convert_existing_weather_csvs(
    csv_dir: Path = Path("input/weather/2010"),
    grid_path: Path = Path("input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg"),
    output_dir: Path = Path("input/weather/2010"),
    overwrite: bool = False,
    interpolate_15min: bool = False,
) -> WeatherBuildResult:
    require_parquet_engine()
    locations = load_grid_locations(grid_path)
    written = 0
    skipped = 0

    for csv_path in sorted(csv_dir.glob("lat-*_lon-*.csv")):
        lat, lon = _parse_lat_lon_from_name(csv_path)
        distances = (locations["latitude"] - lat).abs() + (locations["longitude"] - lon).abs()
        location_id = int(locations.loc[distances.idxmin(), "location_id"])
        output_path = weather_output_path(output_dir, location_id)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        weather = prepare_weather_frame(
            normalize_weather_frame(pd.read_csv(csv_path), location_id),
            interpolate_15min=interpolate_15min,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_weather_parquet(weather, output_path)
        written += 1

    return WeatherBuildResult(written=written, skipped=skipped, output_dir=output_dir)


def compact_existing_weather_parquets(
    weather_dir: Path = Path("input/weather/2010"),
) -> WeatherBuildResult:
    require_parquet_engine()
    written = 0
    skipped = 0

    for parquet_path in sorted(weather_dir.glob("loc*.parquet")):
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        try:
            weather = pd.read_parquet(parquet_path)
            write_weather_parquet(weather, tmp_path)
            tmp_path.replace(parquet_path)
            written += 1
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            skipped += 1

    return WeatherBuildResult(written=written, skipped=skipped, output_dir=weather_dir)


def downsample_existing_weather_parquets_to_hourly(
    weather_dir: Path = Path("input/weather/2010"),
) -> WeatherBuildResult:
    require_parquet_engine()
 