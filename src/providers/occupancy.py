"""Occupancy providers.

Two patterns are common for residential occupancy:

  - DERIVE FROM ELECTRICITY: most appropriate when measured electricity
    profiles are available. The shipped `GeoMAOccupancyProvider` wraps
    the GeoMA algorithm in `src/preprocessing.py`. It applies the paper's
    night rule (occupied 00:00–09:00 if active ≥1 h between 21:00–24:00).

  - LOAD FROM FILE: when occupancy is already curated from another
    source (e.g. probabilistic generators, time-use surveys), use
    `FileOccupancyProvider` and point it at a CSV or parquet matching
    OccupancySchema.

`ParquetOccupancyProvider` is kept as a legacy alias of
`FileOccupancyProvider`.

The pipeline calls `get_occupancy(electricity)` so derived providers can
read the electricity DataFrame; file-based providers ignore that
argument but accept it for interface uniformity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .base import ElectricitySchema, OccupancySchema, validate_schema


@dataclass
class GeoMAOccupancyProvider:
    """Derive binary occupancy from a per-profile electricity series.

    Parameters
    ----------
    alpha : float
        GeoMA threshold parameter. Occupancy = 1 when smoothed power
        exceeds ``alpha`` of the long-window mean. The Germany 2010
        dataset uses 0.05.
    local_tz : str
        IANA timezone (e.g. ``"Europe/Berlin"``) in which the paper's
        evening/morning night rule is evaluated. The electricity readers
        coerce timestamps to UTC for DST safety, so the night rule must
        be re-evaluated in local wall-clock time. Default
        ``"Europe/Berlin"`` matches the Germany 2010 reference dataset.
    """

    alpha: float = 0.05
    local_tz: str = "Europe/Berlin"

    def get_occupancy(self, electricity: pd.DataFrame) -> pd.DataFrame:
        # Local import keeps GeoMA / EnTiSe optional at import time.
        from ..preprocessing import calculate_occupancy

        validate_schema(electricity, ElectricitySchema.REQUIRED, "electricity input")

        frames = []
        for profile_id, profile in electricity.groupby("profile_id", sort=True):
            occ = calculate_occupancy(
                profile.loc[:, list(ElectricitySchema.REQUIRED)],
                lambda_occ=self.alpha,
                local_tz=self.local_tz,
            )
            occ.insert(1, "profile_id", int(profile_id))
            frames.append(occ)

        df = pd.concat(frames, ignore_index=True).loc[:, list(OccupancySchema.REQUIRED)]
        validate_schema(df, OccupancySchema.REQUIRED, "GeoMAOccupancyProvider")
        return df

    def save(self, path: Path, electricity: pd.DataFrame) -> None:
        df = self.get_occupancy(electricity)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


@dataclass
class FileOccupancyProvider:
    """Read a long-form occupancy file matching OccupancySchema.

    The file may be CSV or parquet; the extension decides.
    """

    path: Path

    def get_occupancy(self, electricity: pd.DataFrame | None = None) -> pd.DataFrame:
        p = Path(self.path)
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        validate_schema(df, OccupancySchema.REQUIRED, f"FileOccupancyProvider({p})")
        return df

    def save(self, path: Path, electricity: pd.DataFrame | None = None) -> None:
        df = self.get_occupancy(electricity)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# Legacy alias — older YAML configs reference the Parquet-prefixed name.
ParquetOccupancyProvider = FileOccupancyProvider
