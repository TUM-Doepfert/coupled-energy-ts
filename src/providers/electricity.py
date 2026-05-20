"""Electricity providers.

The pipeline accepts any provider that emits a long-form DataFrame
matching ElectricitySchema. The shipped implementations are:

  Measured / file-based
  ─────────────────────
  - DirectoryElectricityProvider: directory of one-column CSVs
                                  (HTW-style — one file per profile)
  - ParquetElectricityProvider:   a single long-form parquet

  EnTiSe-wrapped synthesisers
  ───────────────────────────
  - DemandlibElectricityProvider: BDEW H0 standard load profile (DE)
  - PyLPGElectricityProvider:     LoadProfileGenerator (probabilistic
                                  household behaviour-based profiles)

(``pht`` support is not yet bundled in EnTiSe; once it is, add a thin
``PhtElectricityProvider`` mirroring the two synthesiser classes below.)

`CSVElectricityProvider` is kept as a legacy alias of
`DirectoryElectricityProvider`. The two electricity readers expose
genuinely different layouts (one-file-per-profile vs. single long-form
parquet), so they remain separate classes.

All providers return the same canonical schema, so the pipeline can swap
between measured and synthetic data with no further code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import ElectricitySchema, validate_schema


# ── Measured / file-based ─────────────────────────────────────────────────

@dataclass
class DirectoryElectricityProvider:
    """Directory of one-column CSVs, one per household profile.

    Each CSV must have a parsable timestamp index and a single power
    column (units: W, average over the interval). Files are sorted by
    integer stem so a folder with names like ``1500.csv``, ``2000.csv``
    becomes profile 1, 2, ... in ascending annual demand.
    """

    input_dir: Path
    resolution: str = "1h"

    def get_profiles(self, year: int | None = None) -> pd.DataFrame:
        from ..preprocessing import electricity_files, read_electricity_csv

        frames = []
        for profile_id, path in enumerate(electricity_files(Path(self.input_dir)), start=1):
            profile = read_electricity_csv(path)
            profile.insert(1, "profile_id", profile_id)
            frames.append(profile)
        df = pd.concat(frames, ignore_index=True).loc[:, list(ElectricitySchema.REQUIRED)]

        if year is not None:
            ts = pd.to_datetime(df["timestamp"])
            df = df.loc[ts.dt.year == year].reset_index(drop=True)

        validate_schema(df, ElectricitySchema.REQUIRED, "DirectoryElectricityProvider")
        return df

    def save(self, path: Path, year: int | None = None) -> None:
        df = self.get_profiles(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# Legacy alias — older YAML configs reference the CSV-prefixed name.
CSVElectricityProvider = DirectoryElectricityProvider


@dataclass
class ParquetElectricityProvider:
    """Read a single long-form parquet that already matches ElectricitySchema."""

    path: Path

    def get_profiles(self, year: int | None = None) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        validate_schema(df, ElectricitySchema.REQUIRED, f"ParquetElectricityProvider({self.path})")
        if year is not None:
            ts = pd.to_datetime(df["timestamp"])
            df = df.loc[ts.dt.year == year].reset_index(drop=True)
        return df

    def save(self, path: Path, year: int | None = None) -> None:
        df = self.get_profiles(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# ── EnTiSe-wrapped synthesisers ───────────────────────────────────────────

def _datetimes_df(year: int, freq: str = "1h") -> pd.DataFrame:
    """Build a tz-naive year-long datetime scaffold for EnTiSe synthesisers."""
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq=freq, tz=None)
    return pd.DataFrame({"datetime": idx})


@dataclass
class DemandlibElectricityProvider:
    """Generate N household profiles using demandlib's BDEW H0 SLP.

    Wraps EnTiSe's ``Demandlib`` method. One profile is produced per entry
    in ``annual_demands_kwh``; the profile_id is the 1-based index so the
    output matches the canonical schema.

    Parameters
    ----------
    annual_demands_kwh : list[float]
        Annual electricity demand in kWh per household. The shape (slope,
        morning/evening peaks) of the BDEW H0 profile is fixed; only the
        scaling differs.
    profile_type : str
        BDEW SLP label: "h0" (households), "g0" (commerce), etc. Default
        "h0", which is appropriate for residential.
    holidays_location : str | None
        Federal-state code for German public-holiday weighting (e.g.
        "BW", "BY"). Pass ``None`` (default) to skip.
    freq : str
        Output sampling rate. Default "1h"; demandlib supports up to "15min".
    """

    annual_demands_kwh: list[float]
    profile_type: str = "h0"
    holidays_location: str | None = None
    freq: str = "1h"

    def get_profiles(self, year: int) -> pd.DataFrame:
        from entise.methods.electricity import Demandlib

        gen = Demandlib()
        scaffold = _datetimes_df(year, freq=self.freq)
        frames = []
        for pid, kwh in enumerate(self.annual_demands_kwh, start=1):
            obj = {
                "datetimes": "datetimes",
                "demand[kWh]": float(kwh),
                "profile": self.profile_type,
            }
            if self.holidays_location is not None:
                obj["holidays_location"] = self.holidays_location
            out = gen.generate(obj=obj, data={"datetimes": scaffold})
            ts = out["timeseries"]
            col = [c for c in ts.columns if "load" in c.lower()][0]
            frames.append(pd.DataFrame({
                "timestamp": ts.index,
                "profile_id": pid,
                "electricity_demand": ts[col].astype(float).values,
            }))
        df = pd.concat(frames, ignore_index=True).loc[:, list(ElectricitySchema.REQUIRED)]
        validate_schema(df, ElectricitySchema.REQUIRED, "DemandlibElectricityProvider")
        return df

    def save(self, path: Path, year: int) -> None:
        df = self.get_profiles(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


@dataclass
class PyLPGElectricityProvider:
    """Generate N household profiles using LoadProfileGenerator (PyLPG).

    Wraps EnTiSe's ``PyLPG`` method. Behaviour-based profiles depend on
    occupant composition; supply one ``HouseholdSpec`` per profile.

    Parameters
    ----------
    households : list[dict]
        Each entry: ``{"households": int, "occupants_per_household": int,
        "energy_intensity": str | None}``. PyLPG runs once per entry.
    freq : str
        Output sampling rate. Default "1h". PyLPG natively produces
        minute energies and aggregates upward.
    """

    households: list[dict] = field(default_factory=list)
    freq: str = "1h"

    def get_profiles(self, year: int) -> pd.DataFrame:
        from entise.methods.electricity import PyLPG

        gen = PyLPG()
        scaffold = _datetimes_df(year, freq=self.freq)
        frames = []
        for pid, spec in enumerate(self.households, start=1):
            obj = {
                "datetimes": "datetimes",
                "households": int(spec["households"]),
                "occupants_per_household": int(spec["occupants_per_household"]),
            }
            if spec.get("energy_intensity") is not None:
                obj["energy_intensity"] = str(spec["energy_intensity"])
            out = gen.generate(obj=obj, data={"datetimes": scaffold})
            ts = out["timeseries"]
            col = [c for c in ts.columns if "load" in c.lower()][0]
            frames.append(pd.DataFrame({
                "timestamp": ts.index,
                "profile_id": pid,
                "electricity_demand": ts[col].astype(float).values,
            }))
        df = pd.concat(frames, ignore_index=True).loc[:, list(ElectricitySchema.REQUIRED)]
        validate_schema(df, ElectricitySchema.REQUIRED, "PyLPGElectricityProvider")
        return df

    def save(self, path: Path, year: int) -> None:
        df = self.get_profiles(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
