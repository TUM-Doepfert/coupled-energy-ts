"""Weather providers.

Two implementations are shipped:

  - OpenMeteoWeatherProvider: wraps `src/weather.py`. Loads a grid of
    location centroids (default: Germany 10 km UTM grid) and fetches
    historical weather from Open-Meteo for each one. Supports parallel
    download, retry/backoff on rate limits, and optional 15-min
    interpolation.

  - FileWeatherProvider: reads pre-existing per-location parquet/CSV
    files matching WeatherSchema. Use this when you already have weather
    data (TMY files, ERA5 extracts, station observations).

`CSVWeatherProvider` is kept as a legacy alias of `FileWeatherProvider`.

Both providers expose `list_locations()`, `get_weather(location_id, year)`,
and `fetch_all(year, output_dir)` so the simulation core can iterate
over locations identically regardless of the underlying source.

Per-location weather is stored as one parquet per location_id (default
naming: ``loc0001.parquet``). This keeps individual files small and
allows lazy loading during simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .base import WeatherSchema, validate_schema


# Standard parquet name convention. Centralising it lets users override.
def weather_filename(location_id: int) -> str:
    return f"loc{int(location_id):04d}.parquet"


@dataclass
class OpenMeteoWeatherProvider:
    """Fetch per-location historical weather from Open-Meteo.

    Parameters
    ----------
    grid_path : Path
        GeoPackage (or any geopandas-readable polygon layer) whose cell
        centroids define the locations. Defaults assume the German
        10 km UTM grid bundled with this repository.
    cache_dir : Path
        Directory for one-parquet-per-location output. Re-runs are
        skipped unless ``overwrite=True``.
    interpolate_15min : bool
        If True, time-interpolate the hourly Open-Meteo response to a
        15-min grid. Default False (keep hourly; resample at sim time).
    n_jobs : int
        Parallel download threads.
    """

    grid_path: Path
    cache_dir: Path
    interpolate_15min: bool = False
    n_jobs: int = 1
    max_retries: int = 5
    locations_csv: Path | None = None  # optional override of grid_path

    def list_locations(self) -> pd.DataFrame:
        from ..weather import load_grid_locations

        if self.locations_csv is not None:
            return pd.read_csv(self.locations_csv).loc[
                :, ["location_id", "latitude", "longitude"]
            ]
        return load_grid_locations(Path(self.grid_path))

    def get_weather(self, location_id: int, year: int) -> pd.DataFrame:
        # Prefer cached parquet if present.
        path = Path(self.cache_dir) / weather_filename(location_id)
        if path.exists():
            df = pd.read_parquet(path)
            validate_schema(df, WeatherSchema.REQUIRED, str(path))
            return df

        # Otherwise fetch fresh and cache.
        from ..weather import Location, fetch_weather

        locations = self.list_locations()
        match = locations.loc[locations["location_id"] == int(location_id)]
        if match.empty:
            raise KeyError(f"location_id={location_id} not found in grid")
        row = match.iloc[0]
        loc = Location(
            location_id=int(row["location_id"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        df = fetch_weather(loc, year=year, interpolate_15min=self.interpolate_15min)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        validate_schema(df, WeatherSchema.REQUIRED, str(path))
        return df

    def fetch_all(self, year: int, output_dir: Path | None = None) -> None:
        from ..weather import fetch_weather_files

        out = Path(output_dir) if output_dir is not None else Path(self.cache_dir)
        fetch_weather_files(
            grid_path=Path(self.grid_path),
            output_dir=out,
            year=year,
            n_jobs=self.n_jobs,
            max_retries=self.max_retries,
            interpolate_15min=self.interpolate_15min,
        )


@dataclass
class FileWeatherProvider:
    """Read per-location weather files already on disk.

    Expects one file per location, named ``loc{id:04d}.parquet`` (or
    .csv) inside ``cache_dir``. A side-car ``location_mapping.csv`` (or
    explicit ``locations_csv``) supplies lat/lon per location_id.

    The schema must match WeatherSchema. Files written by the
    OpenMeteoWeatherProvider are drop-in compatible.
    """

    cache_dir: Path
    locations_csv: Path
    file_extension: str = ".parquet"  # or ".csv"

    def list_locations(self) -> pd.DataFrame:
        df = pd.read_csv(self.locations_csv)
        return df.loc[:, ["location_id", "latitude", "longitude"]]

    def _path(self, location_id: int) -> Path:
        if self.file_extension == ".parquet":
            return Path(self.cache_dir) / weather_filename(location_id)
        # CSV fallback: same stem, different extension.
        return Path(self.cache_dir) / f"loc{int(location_id):04d}.csv"

    def get_weather(self, location_id: int, year: int) -> pd.DataFrame:
        p = self._path(location_id)
        if not p.exists():
            raise FileNotFoundError(f"weather file missing: {p}")
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        validate_schema(df, WeatherSchema.REQUIRED, str(p))
        ts = pd.to_datetime(df["timestamp"])
        return df.loc[ts.dt.year == year].reset_index(drop=True)

    def fetch_all(self, year: int, output_dir: Path | None = None) -> None:
        # No-op: data is already on disk.
        return None


# Legacy alias — older YAML configs reference the CSV-prefixed name.
CSVWeatherProvider = FileWeatherProvider
