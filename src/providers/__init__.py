"""Pluggable input providers for the coupled time-series pipeline.

Four provider families let users swap data sources without touching the
simulation core:

  - ArchetypeProvider     building thermal parameters (R, C, ...)
  - ElectricityProvider   household electricity profiles
  - OccupancyProvider     binary occupancy time series
  - WeatherProvider       per-location weather

Each family has a base.py defining a Protocol + the canonical tabular
schema, plus concrete implementations. Users can either reuse an
implementation or write a thin class that produces the same schema.

The reference Germany 2010 dataset uses:
  - ArchetypeProvider    : TEASER on TABULA-DE
  - ElectricityProvider  : HTW Berlin hourly measured profiles (directory of CSVs)
  - OccupancyProvider    : GeoMA derivation from electricity
  - WeatherProvider      : Open-Meteo fetched per location

Naming
------
File-based readers are prefixed ``File*`` (auto-detect CSV vs parquet by
suffix). The directory-of-CSVs electricity reader is
``DirectoryElectricityProvider`` to signal its different layout from the
single-parquet ``ParquetElectricityProvider``. The older ``CSV*`` /
``Parquet*`` class names are kept as legacy aliases.
"""
from .base import (
    ArchetypeProvider,
    ArchetypeSchema,
    ElectricityProvider,
    ElectricitySchema,
    OccupancyProvider,
    OccupancySchema,
    WeatherProvider,
    WeatherSchema,
    validate_schema,
)
from .archetype import (
    CSVArchetypeProvider,  # legacy alias of FileArchetypeProvider
    FileArchetypeProvider,
    TEASERArchetypeProvider,
)
from .electricity import (
    CSVElectricityProvider,  # legacy alias of DirectoryElectricityProvider
    DemandlibElectricityProvider,
    DirectoryElectricityProvider,
    ParquetElectricityProvider,
    PyLPGElectricityProvider,
)
from .occupancy import (
    FileOccupancyProvider,
    GeoMAOccupancyProvider,
    ParquetOccupancyProvider,  # legacy alias of FileOccupancyProvider
)
from .weather import (
    CSVWeatherProvider,  # legacy alias of FileWeatherProvider
    FileWeatherProvider,
    OpenMeteoWeatherProvider,
)

__all__ = [
    # Protocols + schemas
    "ArchetypeProvider",
    "ArchetypeSchema",
    "ElectricityProvider",
    "ElectricitySchema",
    "OccupancyProvider",
    "OccupancySchema",
    "WeatherProvider",
    "WeatherSchema",
    "validate_schema",
    # Archetype implementations
    "TEASERArchetypeProvider",
    "FileArchetypeProvider",
    "CSVArchetypeProvider",
    # Electricity implementations
    "DirectoryElectricityProvider",
    "CSVElectricityProvider",
    "ParquetElectricityProvider",
    "DemandlibElectricityProvider",
    "PyLPGElectricityProvider",
    # Occupancy implementations
    "GeoMAOccupancyProvider",
    "FileOccupancyProvider",
    "ParquetOccupancyProvider",
    # Weather implementations
    "OpenMeteoWeatherProvider",
    "FileWeatherProvider",
    "CSVWeatherProvider",
]
