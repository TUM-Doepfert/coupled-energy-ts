"""Provider Protocols + canonical schemas.

Each Protocol describes the *interface* — what a provider must return.
The schema dataclasses describe the *shape of the tabular output* —
exact column names, dtypes, and units.

A provider can be implemented in three ways:
  (a) wrapping an existing tool (e.g. TEASER for archetypes, Open-Meteo
      for weather, demandlib for synthetic electricity);
  (b) reading a user-supplied tabular file (parquet or CSV) that already
      adheres to the schema;
  (c) generating data programmatically (unit tests, synthetic studies).

The pipeline calls `provider.get_*()` and works with the resulting
DataFrame — it doesn't care how the data was produced.

All time-indexed providers must agree on the SAME timestamp index for the
target year (15-minute resolution by default; pipeline can resample if
providers disagree).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


# ── Archetypes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArchetypeSchema:
    """Schema for archetype tables.

    REQUIRED COLUMNS (1R1C — minimum to run):
        archetype_id          int    : unique identifier (1..N)
        construction_year     int    : representative year (e.g. 1995)
        area_m2               float  : heated floor area
        n_floors              int    : number of above-ground floors
        height_floor_m        float  : average floor height
        thermal_resistance    float  : envelope resistance R [K/W]
        thermal_capacitance   float  : envelope capacitance C [J/K]
        window_area_total_m2  float  : total transparent envelope area [m^2]
                                        (split equally across N/E/S/W at sim time)

    OPTIONAL COLUMNS (5R1C, ISO 13790):
        H_tr_is               float  [W/K]
        H_tr_ms               float  [W/K]
        H_tr_w                float  [W/K]
        H_tr_em               float  [W/K]
        C_m                   float  [J/K]

    OPTIONAL COLUMNS (7R2C, VDI 6007):
        R_1_AW, C_1_AW, R_1_IW, C_1_IW,
        R_alpha_star_AW, R_alpha_star_IL, R_alpha_star_IW, R_rest_AW

    Pipeline reads only what it needs based on the configured thermal model.
    """
    REQUIRED_1R1C = (
        "archetype_id", "construction_year", "area_m2", "n_floors",
        "height_floor_m", "thermal_resistance", "thermal_capacitance",
        "window_area_total_m2",
    )
    OPTIONAL_5R1C = ("H_tr_is", "H_tr_ms", "H_tr_w", "H_tr_em", "C_m")
    OPTIONAL_7R2C = (
        "R_1_AW", "C_1_AW", "R_1_IW", "C_1_IW",
        "R_alpha_star_AW", "R_alpha_star_IL", "R_alpha_star_IW", "R_rest_AW",
    )


@runtime_checkable
class ArchetypeProvider(Protocol):
    """Returns a tabular DataFrame of building archetypes."""

    def get_archetypes(self) -> pd.DataFrame:
        """Returns a DataFrame matching ArchetypeSchema."""
        ...

    def save(self, path: Path) -> None:
        """Persist archetypes to disk (parquet by default).

        Must round-trip: another provider with this file path should
        produce an identical DataFrame.
        """
        df = self.get_archetypes()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# ── Electricity ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElectricitySchema:
    """Schema for household electricity time series.

    REQUIRED COLUMNS:
        timestamp     datetime    : 15-minute interval (interval-end label)
        profile_id    int         : unique household identifier
        electricity_demand   float [W] : average power over the interval

    Profiles must cover the full target year. Multiple profiles share a
    timestamp index. Recommended storage: a single parquet with one row
    per (timestamp, profile_id).
    """
    REQUIRED = ("timestamp", "profile_id", "electricity_demand")


@runtime_checkable
class ElectricityProvider(Protocol):
    """Returns one or more household electricity time series."""

    def get_profiles(self, year: int) -> pd.DataFrame:
        """Returns a long-form DataFrame matching ElectricitySchema."""
        ...

    def save(self, path: Path, year: int) -> None:
        df = self.get_profiles(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# ── Occupancy ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OccupancySchema:
    """Schema for binary occupancy time series.

    REQUIRED COLUMNS:
        timestamp     datetime    : same index as ElectricitySchema
        profile_id    int         : matches profile_id in electricity
        occupied      int (0/1)   : binary occupancy indicator
    """
    REQUIRED = ("timestamp", "profile_id", "occupied")


@runtime_checkable
class OccupancyProvider(Protocol):
    """Produces binary occupancy time series.

    Two common patterns:
      - DERIVE from electricity: e.g. GeoMA reads an electricity profile
        and returns binary occupancy aligned to it.
      - SYNTHESISE from priors: e.g. probabilistic occupancy generators
        produce a profile from population characteristics.
    """

    def get_occupancy(self, electricity: pd.DataFrame) -> pd.DataFrame:
        """Returns a long-form DataFrame matching OccupancySchema."""
        ...

    def save(self, path: Path, electricity: pd.DataFrame) -> None:
        df = self.get_occupancy(electricity)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# ── Weather ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeatherSchema:
    """Schema for per-location weather time series.

    REQUIRED COLUMNS (one parquet per location_id):
        timestamp                       datetime
        location_id                     int
        air_temperature                 float [°C]
        global_horizontal_irradiance    float [W/m²]
        direct_normal_irradiance        float [W/m²]
        diffuse_horizontal_irradiance   float [W/m²]
        wind_speed                      float [m/s]
        relative_humidity               float [%]

    DNI and DHI are required because the thermal simulation needs the
    beam/diffuse split to compute window solar gains correctly. A
    GHI-only fallback (set DNI=0, DHI=GHI) systematically suppresses
    beam-driven winter gains and inflates heating demand; BYO users
    without DNI/DHI in their source data should decompose GHI via an
    established model (Erbs, DISC, Boland) before constructing the
    canonical parquet.

    Location centroids (lat/lon) live in a separate location-mapping CSV
    referenced by the WeatherProvider; pipeline keeps geometry separate
    from time series for storage efficiency.
    """
    REQUIRED = (
        "timestamp", "location_id", "air_temperature",
        "global_horizontal_irradiance",
        "direct_normal_irradiance", "diffuse_horizontal_irradiance",
        "wind_speed", "relative_humidity",
    )


@runtime_checkable
class WeatherProvider(Protocol):
    """Returns a per-location weather time series for the target year."""

    def list_locations(self) -> pd.DataFrame:
        """Returns DataFrame with columns: location_id, latitude, longitude."""
        ...

    def get_weather(self, location_id: int, year: int) -> pd.DataFrame:
        """Returns time series for one location, matching WeatherSchema."""
        ...

    def fetch_all(self, year: int, output_dir: Path) -> None:
        """Persist all locations' weather as one parquet per location."""
        ...


# ── Schema validation helper ─────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required: tuple[str, ...],
                    name: str = "DataFrame") -> None:
    """Raise ValueError if df is missing required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Have: {list(df.columns)}. "
            f"Required: {list(required)}.")
