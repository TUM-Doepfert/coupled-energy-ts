from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from entise.constants import SEP, Types
from entise.constants.columns import Columns
from entise.constants.general import Keys
from entise.constants.objects import Objects

from .ach import build_ach_series
from .preprocessing import require_parquet_engine
from .thermal_models import (
    ThermalModel,
    get_thermal_model,
    validate_archetype_columns,
)
from .weather import interpolate_weather_15min

# ── Output schema ──────────────────────────────────────────────────────────────
HC_COLUMNS = ["timestamp", "profile_id", "q_heat_w", "q_cool_w"]
HC_PARQUET_COMPRESSION = "zstd"
HC_PARQUET_COMPRESSION_LEVEL = 9

# ── Defaults (file-level constants; never mutated at runtime) ──────────────────
#
# These are the paper's published values for the Germany 2010 dataset. They
# are read once in run_simulation() and threaded through the per-pair worker
# (_simulate_one_pair) as explicit arguments — never as module globals,
# because joblib's loky backend spawns fresh interpreters that re-import this
# module and pick up the file-level defaults, not the parent-process state.
HEATING_SETPOINT_C = 20.0
COOLING_SETPOINT_C = 26.0
INHABITANTS = 2
GAINS_PER_PERSON_W = 80

# ── Windows (solar gain geometry) ──────────────────────────────────────────────
# Total transparent envelope area comes from the archetype parquet
# (window_area_total_m2, populated by TEASER from the TABULA construction
# database). Only the orientation distribution and glazing parameters are
# fixed here.
WINDOW_G_VALUE = 0.6            # solar heat gain coefficient (double glazing)
WINDOW_SHADING = 0.75           # overhang shading factor
WINDOW_TILT = 90.0              # vertical glazing
ORIENTATIONS = [0.0, 90.0, 180.0, 270.0]  # N, E, S, W

# ── Ventilation: which ACH model is in use? ───────────────────────────────────
#
# The published Germany 2010 dataset uses the ``rule_based`` ACH model
# from ``src/ach.py``: presence + outdoor-temperature + season + per-profile
# night-window scenario (S1/S2/S3 assigned by ``profile_id % 3``). It adds
# per-profile diversity to ventilation losses, which is necessary because
# constant ACH across the 74-household stock collapses the simultaneity
# diversity that the dataset is meant to expose.
#
# A legacy ``sinusoid`` model (deterministic seasonal sinusoid, identical
# across every triple) is also implemented in ``src/ach.py`` for
# reproducibility of the original Germany 2010 release. It is selected via
# the YAML key ``simulation.ach_model``; the reference config sets
# ``rule_based``.
# ──────────────────────────────────────────────────────────────────────────────


def _tz_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone from a DatetimeIndex (convert to UTC first to avoid ambiguity)."""
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _rename_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename canonical weather columns to EnTiSe-expected names.

    R1C1 reads weather["datetime"] as a plain column (not the index) and the
    solar strategy sets it as the index for pvlib solar-angle calculations.
    Timestamps are converted to UTC so the timezone object is always
    datetime.timezone.utc. DST-aware zones like Europe/Berlin cause
    timezone_info.utcoffset(None) to return None inside EnTiSe's solar
    strategy, which crashes worker processes. UTC.utcoffset(None) returns
    timedelta(0) reliably. pvlib solar positions are unaffected because they
    depend on the absolute UTC instant, not the display timezone.

    DNI and DHI are required columns of the canonical weather schema (see
    ``WeatherSchema``); the OpenMeteoWeatherProvider fetches
    ``direct_normal_irradiance`` and ``diffuse_radiation`` from the API.
    No GHI-only fallback — collapsing all radiation to diffuse would
    systematically suppress beam window gains on clear winter days and
    inflate heating demand.
    """
    out = (
        df.drop(columns=["location_id", "wind_speed", "relative_humidity"], errors="ignore")
        .rename(columns={
            "timestamp": "datetime",
            "air_temperature": "air_temperature[C]",
            "global_horizontal_irradiance": "global_horizontal_irradiance[W m-2]",
            "direct_normal_irradiance": "direct_normal_irradiance[W m-2]",
            "diffuse_horizontal_irradiance": "diffuse_horizontal_irradiance[W m-2]",
        })
    )
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_convert("UTC")
    return out.reset_index(drop=True)


def _build_windows_df(obj_id: str, window_area_total_m2: float) -> pd.DataFrame:
    """4-row windows DataFrame (N/E/S/W) for one simulation object.

    The total transparent envelope area is taken from the archetype
    parquet (TEASER-derived) and split equally across the four cardinal
    orientations.
    """
    area_per_direction = window_area_total_m2 / len(ORIENTATIONS)
    return pd.DataFrame([
        {
            "id": obj_id,
            "area[m2]": area_per_direction,
            "g_value[1]": WINDOW_G_VALUE,
            "orientation[degree]": o,
            "tilt[degree]": WINDOW_TILT,
            "shading[1]": WINDOW_SHADING,
        }
        for o in ORIENTATIONS
    ])


def _build_internal_gains(
    occ: pd.Series,
    elec: pd.Series,
    inhabitants: int = INHABITANTS,
    gains_per_person_W: float = GAINS_PER_PERSON_W,
) -> pd.DataFrame:
    """Combine occupancy heat and electrical appliance heat into total internal gains.

    Both electricity consumption and body heat contribute to the building's
    internal heat load. Gains are pre-computed here to bypass InternalOccupancy
    in EnTiSe, since O.parquet is already available.
    """
    gains = occ.values * inhabitants * gains_per_person_W + elec.values
    idx = _tz_naive(occ.index if isinstance(occ.index, pd.DatetimeIndex) else pd.DatetimeIndex(occ.index))
    df = pd.DataFrame({"gains_internal[W]": gains}, index=idx)
    df.index.name = "datetime"
    return df


def hc_output_path(output_dir: Path, location_id: int, archetype_id: int) -> Path:
    return output_dir / "HC" / f"loc{location_id:04d}" / f"hc_arch{archetype_id:02d}.parquet"


def simulate_archetype(
    archetype_row: pd.Series,
    profiles_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    location_id: int,
    lat: float,
    lon: float,
    thermal_model: ThermalModel | None = None,
    progress_bar=None,
    heating_setpoint_C: float = HEATING_SETPOINT_C,
    cooling_setpoint_C: float = COOLING_SETPOINT_C,
    inhabitants: int = INHABITANTS,
    gains_per_person_W: float = GAINS_PER_PERSON_W,
    ach_model: str = "sinusoid",
) -> pd.DataFrame:
    """Run the configured HVAC model for all profiles at one (location × archetype).

    Returns a long-format DataFrame with columns HC_COLUMNS.
    weather_df must already be renamed via _rename_weather_columns().
    Default ``thermal_model`` is 1R1C; pass an R5C1 / R7C2 descriptor (see
    ``src/thermal_models.py``) to switch.

    Setpoints, inhabitants and gains-per-person are explicit arguments
    (not module globals) so they survive joblib's loky workers — which
    re-import this module in fresh interpreters and would otherwise pick
    up the file-level defaults instead of any YAML-overridden values.
    """
    if thermal_model is None:
        thermal_model = get_thermal_model("r1c1")

    archetype_id = int(archetype_row["archetype_id"])
    weather_key = f"loc{location_id:04d}"
    # All time-indexed series share the same UTC index as weather.
    # R5C1 / R7C2 concat internal & solar gains across the full data
    # bundle, so any tz mismatch raises 'cannot join tz-naive with
    # tz-aware'. R1C1 tolerates tz-naive but UTC works for it too.
    weather_idx = pd.DatetimeIndex(weather_df["datetime"])
    weather_idx_naive = _tz_naive(weather_idx)
    # T_out aligned to the same grid; needed by the rule_based ACH model.
    T_out_array = weather_df["air_temperature[C]"].to_numpy(dtype=float)

    hvac_gen = thermal_model.load_class()()
    model_keys = thermal_model.obj_keys_factory(archetype_row)
    frames: list[pd.DataFrame] = []

    for profile_id, profile_group in profiles_df.groupby("profile_id", sort=True):
        profile_id = int(profile_id)
        occ_group = occupancy_df[occupancy_df["profile_id"] == profile_id]

        elec = profile_group.set_index("timestamp")["electricity_demand"]
        occ = occ_group.set_index("timestamp")["occupied"]

        obj_id = f"loc{location_id:04d}_arch{archetype_id:02d}_prof{profile_id:02d}"
        gains_df = _build_internal_gains(
            occ, elec,
            inhabitants=inhabitants,
            gains_per_person_W=gains_per_person_W,
        )
        # Reindex onto the weather UTC grid. Two distinct gaps to cover:
        #   (1) interior DST spring-forward — the hourly German electricity
        #       profile loses one hour when converted to UTC (8759 vs the
        #       8760-row weather frame); ffill copies the previous hour's
        #       value across the gap, matching the source CSV which also
        #       collapses that hour.
        #   (2) leading-edge offset — the Open-Meteo cache typically starts
        #       ~2 hours before midnight UTC of the requested calendar
        #       year (it includes a small look-back to absorb the CET
        #       offset). Those 2 hours are BEFORE the gains start, so
        #       ffill leaves them as NaN. We bfill afterwards so the
        #       leading-edge boundary uses the first valid year-of-data
        #       value rather than NaN. Without bfill the NaN propagates
        #       through R1C1's recursive T_in update and zeros out the
        #       entire year's heating and cooling output.
        gains_df = gains_df.reindex(_tz_naive(weather_idx)).ffill().bfill()
        gains_df.index = weather_idx
        gains_df.index.name = "datetime"
        windows_df = _build_windows_df(obj_id, float(archetype_row["window_area_total_m2"]))

        # ACH per profile (allows the rule_based model to use profile_id
        # for its night-window scenario assignment).
        ach = build_ach_series(
            model=ach_model,
            dt_index=weather_idx_naive,
            T_out=T_out_array,
            profile_id=profile_id,
            heating_setpoint_C=heating_setpoint_C,
            cooling_setpoint_C=cooling_setpoint_C,
        )
        ach.index = weather_idx
        ach.index.name = "datetime"

        # Use canonical EnTiSe data keys directly. R1C1 supports user-
        # defined aliases (obj["X"] = "my_alias", data["my_alias"] = ...)
        # via an indirection layer; R5C1 / R7C2 normalize that away when
        # they pre-process `data`. Self-referencing keys (obj["X"] = "X",
        # data["X"] = ...) work for all three.
        obj_dict = {
            Objects.ID: obj_id,
            "inhabitants": inhabitants,
            "weather": weather_key,
            "area[m2]": float(archetype_row["area_m2"]),
            "height[m]": float(archetype_row["height_floor_m"]),
            "latitude[degree]": lat,
            "longitude[degree]": lon,
            "min_temperature[C]": heating_setpoint_C,
            "max_temperature[C]": cooling_setpoint_C,
            "stories": int(archetype_row["n_floors"]),
            "gains_internal[W]": "gains_internal[W]",
            "gains_internal_column": "gains_internal[W]",
            "gains_internal[W]_per_person[W]": gains_per_person_W,
            "ventilation[W K-1]": "ventilation[W K-1]",
            "ventilation_column": "typical [1/h]",
            "init_temperature[C]": heating_setpoint_C,
            "windows": "windows",
        }
        obj_dict.update(model_keys)
        obj = pd.Series(obj_dict)

        data = {
            weather_key: weather_df,
            "gains_internal[W]": gains_df,
            "ventilation[W K-1]": ach,
            "windows": windows_df,
        }

        # R7C2 quirk: it strips 'ventilation[W K-1]' from data (not in
        # its optional_data) but always calls VentilationTimeSeries.
        # Workaround: stash ACH under 'H_ve[W K-1]' (which R7C2 keeps),
        # tell R7C2 to keep that slot (obj['H_ve[W K-1]'] = same key),
        # and point the ventilation reference at it. VentilationTimeSeries
        # then converts ACH -> H_ve internally using volume from
        # obj['area[m2]'] * obj['height[m]'].
        if thermal_model.name == "r7c2":
            data["H_ve[W K-1]"] = ach
            obj["H_ve[W K-1]"] = "H_ve[W K-1]"
            obj["ventilation[W K-1]"] = "H_ve[W K-1]"

        hvac = hvac_gen.generate(obj, data)
        ts = hvac[Keys.TIMESERIES]

        frames.append(pd.DataFrame({
            "timestamp": ts.index,
            "profile_id": profile_id,
            "q_heat_w": ts[f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"],
            "q_cool_w": ts[f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"],
        }))

        if progress_bar is not None:
            progress_bar.update(1)

    return pd.concat(frames, ignore_index=True)[HC_COLUMNS]


def simulate_location(
    location_id: int,
    lat: float,
    lon: float,
    archetypes_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
    weather_dir: Path,
    output_dir: Path,
    resolution: str = "60min",
    overwrite: bool = False,
    thermal_model: ThermalModel | None = None,
    progress_bar=None,
    heating_setpoint_C: float = HEATING_SETPOINT_C,
    cooling_setpoint_C: float = COOLING_SETPOINT_C,
    inhabitants: int = INHABITANTS,
    gains_per_person_W: float = GAINS_PER_PERSON_W,
) -> int:
    """Simulate all archetypes for one location. Returns count of Parquet files written.

    Skips archetypes whose output file already exists (checkpointing).
    Skips the entire location if its weather Parquet is missing.
    """
    require_parquet_engine()

    n_profiles = profiles_df["profile_id"].nunique()

    pending_ids = [
        int(row["archetype_id"])
        for _, row in archetypes_df.iterrows()
        if overwrite or not hc_output_path(output_dir, location_id, int(row["archetype_id"])).exists()
    ]
    skipped_ids = [
        int(row["archetype_id"])
        for _, row in archetypes_df.iterrows()
        if int(row["archetype_id"]) not in pending_ids
    ]
    # Tick the progress bar for any cached archetypes so the totals reflect
    # work envelope, not unfinished work.
    if progress_bar is not None and skipped_ids:
        progress_bar.update(len(skipped_ids) * n_profiles)

    if not pending_ids:
        return 0

    weather_path = weather_dir / f"loc{location_id:04d}.parquet"
    if not weather_path.exists():
        # Tick the rest as "skipped due to missing weather".
        if progress_bar is not None:
            progress_bar.update(len(pending_ids) * n_profiles)
        return 0

    weather_df = pd.read_parquet(weather_path)
    if resolution == "15min":
        weather_df = interpolate_weather_15min(weather_df)
    weather_df = _rename_weather_columns(weather_df)

    written = 0
    for _, archetype_row in archetypes_df.iterrows():
        if int(archetype_row["archetype_id"]) not in pending_ids:
            continue

        output_path = hc_output_path(output_dir, location_id, int(archetype_row["archetype_id"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        hc_df = simulate_archetype(
            archetype_row, profiles_df, occupancy_df, weather_df,
            location_id, lat, lon, thermal_model=thermal_model,
            progress_bar=progress_bar,
            heating_setpoint_C=heating_setpoint_C,
            cooling_setpoint_C=cooling_setpoint_C,
            inhabitants=inhabitants,
            gains_per_person_W=gains_per_person_W,
        )
        hc_df.to_parquet(
            output_path,
            index=False,
            compression=HC_PARQUET_COMPRESSION,
            compression_level=HC_PARQUET_COMPRESSION_LEVEL,
        )
        written += 1

    return written


# ── Top-level driver (used by src/pipeline.py) ────────────────────────────


def _simulate_one_pair(
    location_id: int,
    lat: float,
    lon: float,
    archetype_row: dict,
    profiles_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
    weather_dir: Path,
    output_dir: Path,
    resolution: str,
    overwrite: bool,
    thermal_model: ThermalModel,
    heating_setpoint_C: float,
    cooling_setpoint_C: float,
    inhabitants: int,
    gains_per_person_W: float,
    ach_model: str,
) -> int:
    """One (location, archetype) job. Returns 1 if it wrote the parquet,
    0 if it skipped (cached or weather missing). Used as the unit of
    parallel work in ``run_simulation``.

    archetype_row is passed as a plain dict so it pickles cleanly across
    subprocess boundaries. Setpoints / inhabitants / gains are passed
    explicitly so loky workers (which re-import this module in fresh
    interpreters) use the YAML-configured values, not the module
    file-level defaults."""
    from pathlib import Path as _P
    archetype_id = int(archetype_row["archetype_id"])
    output_path = hc_output_path(_P(output_dir), location_id, archetype_id)
    if not overwrite and output_path.exists():
        return 0
    weather_path = _P(weather_dir) / f"loc{location_id:04d}.parquet"
    if not weather_path.exists():
        return 0
    weather_df = pd.read_parquet(weather_path)
    if resolution == "15min":
        weather_df = interpolate_weather_15min(weather_df)
    weather_df = _rename_weather_columns(weather_df)
    arch_series = pd.Series(archetype_row)
    hc_df = simulate_archetype(
        arch_series, profiles_df, occupancy_df, weather_df,
        location_id, lat, lon, thermal_model=thermal_model,
        progress_bar=None,
        heating_setpoint_C=heating_setpoint_C,
        cooling_setpoint_C=cooling_setpoint_C,
        inhabitants=inhabitants,
        gains_per_person_W=gains_per_person_W,
        ach_model=ach_model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hc_df.to_parquet(
        output_path,
        index=False,
        compression=HC_PARQUET_COMPRESSION,
        compression_level=HC_PARQUET_COMPRESSION_LEVEL,
    )
    return 1


def run_simulation(
    output_dir: Path,
    weather_dir: Path | None = None,
    location_mapping: Path | None = None,
    thermal_model_name: str = "r1c1",
    heating_setpoint_C: float = HEATING_SETPOINT_C,
    cooling_setpoint_C: float = COOLING_SETPOINT_C,
    inhabitants: int = INHABITANTS,
    gains_per_person_W: float = GAINS_PER_PERSON_W,
    ach_model: str = "sinusoid",
    resolution: str = "60min",
    overwrite: bool = False,
    n_jobs: int = -1,
    locations_filter=None,
) -> int:
    """Drive the whole HC simulation step from canonical parquet outputs.

    Reads ``B.parquet``, ``E.parquet``, ``O.parquet`` from ``output_dir`` and
    one ``loc####.parquet`` per location from ``weather_dir``. Runs the
    selected thermal model (1R1C / 5R1C / 7R2C) and writes one parquet per
    (location, archetype) pair under ``output_dir / 'HC' / loc#### /``.

    Parallelism: ``n_jobs`` controls subprocess parallelism over
    (location × archetype) pairs. The unit of work is one (location,
    archetype) — each one runs the inner 74-profile loop in-process.
      - n_jobs = -1 (default) : use all physical CPU cores (loky backend)
      - n_jobs = 0 or 1       : sequential
      - n_jobs = N            : N worker processes
    Capped at the number of (location × archetype) pairs.

    Progress bar ticks once per finished pair (n_profiles units at a
    time), so the bar updates every few seconds in parallel mode rather
    than every minute.
    """
    require_parquet_engine()

    # Coerce once; pass through workers as explicit args (never as module
    # globals — loky workers re-import this module).
    heating_setpoint_C = float(heating_setpoint_C)
    cooling_setpoint_C = float(cooling_setpoint_C)
    inhabitants = int(inhabitants)
    gains_per_person_W = float(gains_per_person_W)

    output_dir = Path(output_dir)
    weather_dir = Path(weather_dir) if weather_dir is not None else Path("input/weather/2010")
    location_mapping = (
        Path(location_mapping)
        if location_mapping is not None
        else output_dir / "location_mapping.csv"
    )

    archetypes_df = pd.read_parquet(output_dir / "B.parquet")
    profiles_df = pd.read_parquet(output_dir / "E.parquet")
    occupancy_df = pd.read_parquet(output_dir / "O.parquet")
    locations = pd.read_csv(location_mapping)

    # Apply YAML locations filter.
    if locations_filter is None or (isinstance(locations_filter, str)
                                     and locations_filter.lower() == "all"):
        pass
    elif isinstance(locations_filter, list):
        keep = {int(x) for x in locations_filter}
        locations = locations[locations["location_id"].isin(keep)].reset_index(drop=True)
    elif isinstance(locations_filter, str) and locations_filter.startswith("random:"):
        n = int(locations_filter.split(":", 1)[1])
        locations = locations.sample(n=min(n, len(locations)),
                                      random_state=42).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported locations filter: {locations_filter!r}")

    n_locations = len(locations)
    n_archetypes = len(archetypes_df)
    n_profiles = profiles_df["profile_id"].nunique()
    total_units = n_locations * n_archetypes * n_profiles
    n_pairs = n_locations * n_archetypes

    # Resolve n_jobs.
    if n_jobs is None or n_jobs == -1:
        try:
            from joblib import cpu_count as _jcpu
            physical = _jcpu(only_physical_cores=True)
        except (TypeError, ImportError):
            physical = max(1, (os.cpu_count() or 2) // 2)
        effective_jobs = max(1, physical)
    elif n_jobs <= 0:
        effective_jobs = 1
    else:
        effective_jobs = int(n_jobs)
    effective_jobs = max(1, min(effective_jobs, n_pairs))
    parallel = effective_jobs > 1 and n_pairs > 1

    print(f"[run_simulation] running {n_locations} location(s) "
          f"× {n_archetypes} archetypes × {n_profiles} profiles "
          f"= {total_units:,} time-series, "
          f"{'parallel n_jobs=' + str(effective_jobs) if parallel else 'sequential'}")

    model = get_thermal_model(thermal_model_name)
    validate_archetype_columns(model, archetypes_df)

    # Build the pair list once.
    pairs = [
        (int(loc["location_id"]), float(loc["latitude"]),
         float(loc["longitude"]), arch.to_dict())
        for _, loc in locations.iterrows()
        for _, arch in archetypes_df.iterrows()
    ]

    written_total = 0
    with tqdm(total=total_units, desc="HVAC sim",
              unit="ts", smoothing=0.05) as pbar:
        if not parallel:
            for loc_id, lat, lon, arch_dict in pairs:
                wrote = _simulate_one_pair(
                    loc_id, lat, lon, arch_dict,
                    profiles_df, occupancy_df, weather_dir, output_dir,
                    resolution, overwrite, model,
                    heating_setpoint_C, cooling_setpoint_C,
                    inhabitants, gains_per_person_W,
                    ach_model,
                )
                written_total += wrote
                pbar.update(n_profiles)
        else:
            jobs = (
                delayed(_simulate_one_pair)(
                    loc_id, lat, lon, arch_dict,
                    profiles_df, occupancy_df, weather_dir, output_dir,
                    resolution, overwrite, model,
                    heating_setpoint_C, cooling_setpoint_C,
                    inhabitants, gains_per_person_W,
                    ach_model,
                )
                for (loc_id, lat, lon, arch_dict) in pairs
            )
            results = Parallel(
                n_jobs=effective_jobs,
                backend="loky",
                return_as="generator_unordered",
            )(jobs)
            for wrote in results:
                written_total += wrote
                pbar.update(n_profiles)
    return written_total
