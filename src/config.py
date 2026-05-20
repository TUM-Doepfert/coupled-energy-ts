"""YAML configuration loader.

Reads a single YAML file (see ``config/germany_2010.yml`` for the
reference) and dispatches the provider sections to concrete provider
classes via a ``type:`` key.

Public surface:

    load_config(path)   -> dict
    build_providers(cfg) -> Providers (named tuple of 4 instances)

The pipeline orchestrator in ``src/pipeline.py`` is the only intended
caller; advanced users can call ``build_providers`` directly to drive
parts of the pipeline themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .providers import (
    ArchetypeProvider,
    DemandlibElectricityProvider,
    DirectoryElectricityProvider,
    ElectricityProvider,
    FileArchetypeProvider,
    FileOccupancyProvider,
    FileWeatherProvider,
    GeoMAOccupancyProvider,
    OccupancyProvider,
    OpenMeteoWeatherProvider,
    ParquetElectricityProvider,
    PyLPGElectricityProvider,
    TEASERArchetypeProvider,
    WeatherProvider,
)


@dataclass(frozen=True)
class Providers:
    """Bundle of all four provider instances built from a config."""
    archetype: ArchetypeProvider
    electricity: ElectricityProvider
    occupancy: OccupancyProvider
    weather: WeatherProvider


# ── Loader ───────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML config file and return the raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return cfg


# ── Provider dispatch ────────────────────────────────────────────────────

def _pop_type(section: dict[str, Any], section_name: str) -> str:
    if "type" not in section:
        raise ValueError(f"Config section '{section_name}' missing required key 'type'")
    return str(section.pop("type"))


def _path(value: Any) -> Path:
    return Path(value)


def _build_archetype_provider(section: dict[str, Any]) -> ArchetypeProvider:
    s = dict(section)
    typ = _pop_type(s, "archetype_provider")
    if typ == "teaser":
        return TEASERArchetypeProvider(
            archetypes_csv=_path(s["archetypes_csv"]),
            construction_data=s.get("config_data", "tabula_de_standard"),
            geometry_data=s.get("geometry_data", "tabula_de_single_family_house"),
        )
    # "file" (new, format auto-detected by suffix) or "csv" (legacy alias).
    if typ in ("file", "csv"):
        return FileArchetypeProvider(path=_path(s["path"]))
    raise ValueError(
        f"Unknown archetype_provider.type: {typ!r}. "
        f"Supported: 'teaser', 'file' (or legacy 'csv')."
    )


def _build_electricity_provider(section: dict[str, Any]) -> ElectricityProvider:
    s = dict(section)
    typ = _pop_type(s, "electricity_provider")
    # "directory" (new) or "csv" (legacy alias) — one file per profile.
    if typ in ("directory", "csv"):
        return DirectoryElectricityProvider(
            input_dir=_path(s["input_dir"]),
            resolution=str(s.get("resolution", "1h")),
        )
    if typ == "parquet":
        return ParquetElectricityProvider(path=_path(s["path"]))
    if typ == "demandlib":
        return DemandlibElectricityProvider(
            annual_demands_kwh=list(s["annual_demands_kwh"]),
            profile_type=str(s.get("profile_type", "h0")),
            holidays_location=s.get("holidays_location"),
            freq=str(s.get("freq", "1h")),
        )
    if typ == "pylpg":
        return PyLPGElectricityProvider(
            households=list(s["households"]),
            freq=str(s.get("freq", "1h")),
        )
    raise ValueError(
        f"Unknown electricity_provider.type: {typ!r}. "
        f"Supported: 'directory' (or legacy 'csv'), 'parquet', 'demandlib', 'pylpg'."
    )


def _build_occupancy_provider(section: dict[str, Any]) -> OccupancyProvider:
    s = dict(section)
    typ = _pop_type(s, "occupancy_provider")
    if typ == "geoma":
        return GeoMAOccupancyProvider(
            alpha=float(s.get("alpha", 0.05)),
            local_tz=str(s.get("local_tz", "Europe/Berlin")),
        )
    # "file" (new, format auto-detected by suffix) or "parquet" (legacy alias).
    if typ in ("file", "parquet"):
        return FileOccupancyProvider(path=_path(s["path"]))
    raise ValueError(
        f"Unknown occupancy_provider.type: {typ!r}. "
        f"Supported: 'geoma', 'file' (or legacy 'parquet')."
    )


def _build_weather_provider(section: dict[str, Any]) -> WeatherProvider:
    s = dict(section)
    typ = _pop_type(s, "weather_provider")
    if typ == "openmeteo":
        return OpenMeteoWeatherProvider(
            grid_path=_path(s["grid_path"]),
            cache_dir=_path(s["cache_dir"]),
            interpolate_15min=bool(s.get("interpolate_15min", False)),
            n_jobs=int(s.get("n_jobs", 1)),
            max_retries=int(s.get("max_retries", 5)),
        )
    # "file" (new) or "csv" (legacy alias) — pre-existing per-location files.
    if typ in ("file", "csv"):
        # Accept both "weather_dir"/"location_mapping" (YAML doc names)
        # and "cache_dir"/"locations_csv" (provider field names).
        cache_dir = s.get("weather_dir") or s["cache_dir"]
        loc_csv = s.get("location_mapping") or s["locations_csv"]
        return FileWeatherProvider(
            cache_dir=_path(cache_dir),
            locations_csv=_path(loc_csv),
            file_extension=str(s.get("file_extension", ".parquet")),
        )
    raise ValueError(
        f"Unknown weather_provider.type: {typ!r}. "
        f"Supported: 'openmeteo', 'file' (or legacy 'csv')."
    )


def build_providers(cfg: dict[str, Any]) -> Providers:
    """Instantiate the four providers from a parsed config dict."""
    for required in ("archetype_provider", "electricity_provider",
                     "occupancy_provider", "weather_provider"):
        if required not in cfg:
            raise ValueError(f"Config missing required section: {required}")
    return Providers(
        archetype=_build_archetype_provider(cfg["archetype_provider"]),
        electricity=_build_electricity_provider(cfg["electricity_provider"]),
        occupancy=_build_occupancy_provider(cfg["occupancy_provider"]),
        weather=_build_weather_provider(cfg["weather_provider"]),
    )
