"""US per-building v1: re-simulate NREL EULP buildings with our 1R1C method.

For each qualified SFH in the 6 EULP counties:
  1. Map EULP metadata -> SimInputs (eulp_to_sim.py)
  2. Fit TEASER on EULP geometry to obtain R, C (uses tabula_de_standard
     since that is the project's archetype space; matched by vintage + area)
  3. Run R1C1 with EULP weather + GeoMA-derived occupancy gains
  4. Save simulated q_heat_w / q_cool_w next to NREL's

By default, EULP-derived setpoint schedules (e.g. "Night -2h" 6 °F heating
setbacks, similar daytime setups for cooling) are applied per timestep.
This better matches the EULP simulator and is needed for the active-share /
load-duration validation (constant setpoints collapse the diversity that
EULP exhibits because all households tip into "active" at the same hour).
Pass ``--no-setpoint-schedules`` to fall back to constant base setpoints
(matches the published-dataset method, kept available for ablation).

Usage:
    uv run python validation/us/building_simulate.py \\
        --county G2601590  # one county at a time, or --county all
    uv run python validation/us/building_simulate.py \\
        --county G2601590 --no-setpoint-schedules
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from entise.constants import SEP, Types
from entise.constants.columns import Columns
from entise.constants.general import Keys
from entise.constants.objects import Objects
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.ventilation.selector import Ventilation
from entise.methods.hvac import R1C1
from teaser.project import Project

from eulp_to_sim import (
    F_TO_C,
    build_setpoint_series,
    build_ventilation_series,
    map_metadata_row,
)
from COUNTIES import COUNTIES

# ── Constants ─────────────────────────────────────────────────────────────
GAINS_PER_PERSON_W = 80        # match src.simulation (EN 16798-1 / ISO 17772-1 sedentary residential)
# Note: occupant count per building comes from EULP `in.occupants` via
# eulp_to_sim.SimInputs.n_occupants (with default 2 if missing). No global
# INHABITANTS constant is needed here.
DESIGN_T_F = -10.0             # for HVAC capacity sizing fallback
_MIN_DEADBAND_K = 2.0          # minimum heating/cooling setpoint deadband


def fit_teaser_one(year: int, area_m2: float, n_floors: int,
                   height_floor: float = 2.5,
                   us_vintage_floor: int = 1995):
    """TEASER R, C for a building. Uses tabula_de_standard typology.

    DIAGNOSTIC HACK: For US buildings (US per-building validation against EULP),
    we clip the construction year up to `us_vintage_floor` (default 1995)
    because tabula_de's pre-1945 archetypes assume German mass-brick
    construction (very leaky), while American wood-frame buildings of any
    era are typically much better-insulated. Treating US buildings as
    >=1995 German construction is a coarse but quick correction; a
    principled fix would compute UA from EULP envelope fields directly.
    """
    # Clip up to us_vintage_floor BEFORE the standard tabula_de range clamp
    y = max(us_vintage_floor, min(year, 2015))
    y = max(1860, y)
    prj = Project()
    try:
        prj.add_residential(
            construction_data="tabula_de_standard",
            geometry_data="tabula_de_single_family_house",
            name=f"b{year}",
            year_of_construction=y,
            number_of_floors=n_floors,
            height_of_floors=height_floor,
            net_leased_area=area_m2,
            inner_wall_approximation_approach="teaser_default",
        )
        prj.calc_all_buildings()
        if not prj.buildings:
            return None
        bldg = prj.buildings[0]
        bldg.calc_building_parameter(number_of_elements=1, merge_windows=True,
                                     used_library="IBPSA")
        zone = bldg.thermal_zones[0]
        m = zone.model_attr
        R = float(m.r_total_ow)
        C = float(m.c1_ow) + zone.volume * zone.density_air * zone.heat_capac_air
        return R, C
    except Exception:
        return None


def load_weather(weather_csv: Path) -> pd.DataFrame:
    """Load EULP county weather CSV (hourly), interpolate to 15-min, and
    return DataFrame with tz-aware UTC `datetime` column plus the radiation
    components EnTiSe expects."""
    w = pd.read_csv(weather_csv)
    ts = pd.to_datetime(w["date_time"]) - pd.Timedelta(hours=1)
    hourly = pd.DataFrame({
        "datetime": ts,  # tz-naive for now; localized after interpolation
        "air_temperature[C]": pd.to_numeric(w["Dry Bulb Temperature [°C]"]).astype(float),
        "global_horizontal_irradiance[W m-2]": pd.to_numeric(w["Global Horizontal Radiation [W/m2]"]),
        "direct_normal_irradiance[W m-2]": pd.to_numeric(w["Direct Normal Radiation [W/m2]"]),
        "diffuse_horizontal_irradiance[W m-2]": pd.to_numeric(w["Diffuse Horizontal Radiation [W/m2]"]),
    }).set_index("datetime").sort_index()
    full_15 = pd.date_range(hourly.index.min(),
                            hourly.index.max() + pd.Timedelta(minutes=45),
                            freq="15min")
    out = hourly.reindex(full_15).interpolate(method="time").ffill().bfill()
    out.index.name = "datetime"
    out = out.reset_index()
    # Localize to UTC (EnTiSe needs tz-aware weather)
    out["datetime"] = out["datetime"].dt.tz_localize("UTC")
    return out


def build_internal_gains(electricity_w: np.ndarray, occupancy: np.ndarray,
                         dt_index: pd.DatetimeIndex,
                         n_occupants: int = 2) -> pd.DataFrame:
    gains = occupancy * n_occupants * GAINS_PER_PERSON_W + electricity_w
    df = pd.DataFrame({"gains_internal[W]": gains.astype(np.float32)}, index=dt_index)
    df.index.name = "datetime"
    return df


def detect_occupancy_geoma(electricity_w: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """GeoMA: occupancy when p_t > g_t. Geometric moving average of electricity.
    Plus the night rule from the paper: if occupied >= 1h between 21-24,
    assumed occupied 00-09."""
    n = len(electricity_w)
    g = np.zeros(n, dtype=np.float32)
    g[0] = electricity_w[0]
    for i in range(1, n):
        g[i] = alpha * electricity_w[i] + (1 - alpha) * g[i-1]
    occ = (electricity_w > g).astype(np.float32)
    return occ


def _enforce_deadband_arrays(t_heat: np.ndarray, t_cool: np.ndarray,
                              min_K: float = _MIN_DEADBAND_K) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized deadband fix.

    EULP setback schedules (heating goes down at night, cooling goes up
    during the day) usually preserve a comfortable deadband, but for ~16%
    of EULP buildings the *base* setpoints cross or sit too close, and a
    setback can make it worse. Where t_cool - t_heat < min_K, widen
    symmetrically around the midpoint so the two arrays never meet.
    """
    h = np.asarray(t_heat, dtype=np.float32).copy()
    c = np.asarray(t_cool, dtype=np.float32).copy()
    bad = (c - h) < min_K
    if bad.any():
        mid = (h[bad] + c[bad]) / 2.0
        h[bad] = mid - min_K / 2.0
        c[bad] = mid + min_K / 2.0
    return h, c


def _solve_1r1c_arrays(
    R: float, C: float,
    temp_init: float,
    temp_air: np.ndarray, solar_gains: np.ndarray,
    internal_gains: np.ndarray, ventilation: np.ndarray,
    t_min: np.ndarray, t_max: np.ndarray,
    P_h_max: float, P_c_max: float,
    on_h: bool, on_c: bool,
    timestep_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1R1C solver mirroring entise.methods.hvac.R1C1.calculate_timeseries_1r1c
    but with per-timestep ``t_min`` and ``t_max`` arrays.

    Math is bit-for-bit identical to EnTiSe's solver when ``t_min`` /
    ``t_max`` are constant; the array path is a strict generalization.
    """
    n = temp_air.shape[0]
    temp_in = np.empty(n, dtype=np.float32)
    p_heat = np.zeros(n, dtype=np.float32)
    p_cool = np.zeros(n, dtype=np.float32)

    R = np.float32(R)
    C = np.float32(C)
    inv_R = np.float32(1.0) / R
    inv_dt = np.float32(1.0) / np.float32(timestep_s)
    dt_over_C = np.float32(timestep_s) / C
    P_h_max = np.float32(P_h_max)
    P_c_max = np.float32(P_c_max)

    temp_prev = np.float32(temp_init)
    temp_in[0] = temp_prev

    for t in range(1, n):
        delta = temp_air[t] - temp_prev
        net = delta * inv_R + ventilation[t] * delta + solar_gains[t] + internal_gains[t]
        if on_h:
            req_h = C * (t_min[t] - temp_prev) * inv_dt - net
            if req_h > 0:
                p_heat[t] = min(req_h, P_h_max)
        if on_c:
            req_c = C * (temp_prev - t_max[t]) * inv_dt + net
            if req_c > 0:
                p_cool[t] = min(req_c, P_c_max)
        temp_prev = temp_prev + dt_over_C * (net + p_heat[t] - p_cool[t])
        temp_in[t] = temp_prev

    return temp_in, p_heat, p_cool


def _run_with_setpoint_arrays(
    obj: pd.Series, data: dict, t_heat_arr: np.ndarray, t_cool_arr: np.ndarray,
) -> pd.DataFrame:
    """Drop-in replacement for ``R1C1().generate(obj, data)[Keys.TIMESERIES]``
    that accepts per-timestep heating/cooling setpoint arrays.

    Reuses EnTiSe's internal/solar/ventilation auxiliaries unchanged, then
    runs the local array-aware solver. Returns the timeseries DataFrame in
    the same layout EnTiSe produces (datetime index, columns
    "indoor_temperature[C]", "heating:load[W]", "cooling:load[W]").
    """
    method = R1C1()
    obj_dict = obj.to_dict() if isinstance(obj, pd.Series) else dict(obj)
    obj_dict, data = method._process_kwargs(obj_dict, data)
    obj_dict, data = method._get_input_data(obj_dict, data, Types.HVAC)

    weather = data[Objects.WEATHER]
    idx = weather[Columns.DATETIME].values.astype("datetime64[ns]")
    timestep_s = float((idx[1] - idx[0]) / np.timedelta64(1, "s"))

    data[Objects.GAINS_INTERNAL] = InternalGains().generate(obj_dict, data)
    data[Objects.GAINS_SOLAR] = SolarGains().generate(obj_dict, data)
    data[Objects.VENTILATION] = Ventilation().generate(obj_dict, data)

    temp_air = weather[Columns.TEMP_AIR].to_numpy(dtype=np.float32, copy=False)
    solar = data[Objects.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False).ravel()
    intern = data[Objects.GAINS_INTERNAL].to_numpy(dtype=np.float32, copy=False).ravel()
    vent = data[Objects.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()

    n = temp_air.shape[0]
    if t_heat_arr.shape[0] != n:
        # Setpoint arrays were built on a different (sub-hour or local-tz)
        # grid; reindex by hour-of-year position to match the weather grid.
        t_heat_arr = np.resize(t_heat_arr, n)
        t_cool_arr = np.resize(t_cool_arr, n)

    R = float(obj_dict[Objects.RESISTANCE])
    C = float(obj_dict[Objects.CAPACITANCE])
    temp_init = float(obj_dict.get(Objects.TEMP_INIT, t_heat_arr[0]))
    P_h_max = float(obj_dict.get(Objects.POWER_HEATING, 1e9))
    P_c_max = float(obj_dict.get(Objects.POWER_COOLING, 1e9))
    on_h = bool(obj_dict.get(Objects.ACTIVE_HEATING, True))
    on_c = bool(obj_dict.get(Objects.ACTIVE_COOLING, True))

    temp_in, p_heat, p_cool = _solve_1r1c_arrays(
        R, C, temp_init,
        temp_air, solar, intern, vent,
        t_heat_arr, t_cool_arr,
        P_h_max, P_c_max, on_h, on_c, timestep_s,
    )

    df = pd.DataFrame(
        {
            Columns.TEMP_IN: temp_in.round(3),
            f"{Types.HEATING}{SEP}{Columns.LOAD}[W]": p_heat.round().astype(int),
            f"{Types.COOLING}{SEP}{Columns.LOAD}[W]": p_cool.round().astype(int),
        },
        index=weather.index,
    )
    df.index.name = Columns.DATETIME
    return df


def simulate_building(
    sim_inputs, weather_df: pd.DataFrame,
    electricity_w: np.ndarray, dt_index: pd.DatetimeIndex,
    R: float, C: float,
    use_setpoint_schedules: bool = True,
) -> pd.DataFrame:
    """Run R1C1 for one building.

    Mirrors the timezone-handling pattern of src/simulation.py:
      - weather["datetime"] is tz-aware UTC (set in load_weather)
      - gains_df and ach_df use a tz-NAIVE DatetimeIndex (matching what
        _tz_naive() does in src.simulation)
    EnTiSe expects this asymmetric setup.

    When ``use_setpoint_schedules`` is True (default), EULP-derived hourly
    heating/cooling setpoints (with night setbacks / day setups parsed from
    in.heating_setpoint_offset_period etc.) are applied per timestep. When
    False, the constant base setpoints from EULP metadata are used (with
    deadband enforcement) — this is the published-dataset behavior.
    """
    dt = pd.DatetimeIndex(dt_index)
    if dt.tz is not None:
        dt_naive = dt.tz_convert("UTC").tz_localize(None)
    else:
        dt_naive = dt  # already tz-naive
    occ = detect_occupancy_geoma(electricity_w)
    gains_df = build_internal_gains(electricity_w, occ, dt_naive,
                                     n_occupants=sim_inputs.n_occupants)

    ach_t = build_ventilation_series(dt_naive, sim_inputs.ach50)
    ach_df = pd.DataFrame({"typical [1/h]": ach_t}, index=dt_naive)
    ach_df.index.name = "datetime"

    obj_id = f"eulp_{sim_inputs.bldg_id}"

    # Setpoint arrays (always; constant when schedules are off)
    if use_setpoint_schedules:
        t_heat_arr = build_setpoint_series(
            dt_naive, sim_inputs.base_heat_F, sim_inputs.heat_offset, is_heating=True,
        )
        t_cool_arr = build_setpoint_series(
            dt_naive, sim_inputs.base_cool_F, sim_inputs.cool_offset, is_heating=False,
        )
    else:
        n = len(dt_naive)
        t_heat_arr = np.full(n, F_TO_C(sim_inputs.base_heat_F), dtype=np.float32)
        t_cool_arr = np.full(n, F_TO_C(sim_inputs.base_cool_F), dtype=np.float32)

    # Per-timestep deadband enforcement. ~16% of EULP buildings have
    # heat_setpoint >= cool_setpoint at base, and setbacks can flip more
    # hours. Without this, R1C1 oscillates between heating and cooling on
    # the same step.
    t_heat_arr, t_cool_arr = _enforce_deadband_arrays(t_heat_arr, t_cool_arr)

    # Map cardinal/intercardinal compass codes to azimuth degrees from N
    _AZ = {"N":0.0,"NE":45.0,"E":90.0,"SE":135.0,"S":180.0,
           "SW":225.0,"W":270.0,"NW":315.0}
    win_rows = []
    for orient_code, area in sim_inputs.window_areas_m2.items():
        deg = _AZ.get(orient_code, 180.0)  # default S if unknown
        win_rows.append({
            "id": obj_id, "area[m2]": area,
            "g_value[1]": sim_inputs.window_shgc,
            "orientation[degree]": deg,
            "tilt[degree]": 90.0, "shading[1]": 0.75,
        })
    windows_df = pd.DataFrame(win_rows)

    # HVAC capacity caps from NREL ResStock autosizing. A cap of 0 means
    # the building has no system in that mode (NREL convention) — we
    # honor that by setting active_*=False so cooling/heating is wholly
    # disabled instead of trying to clip against zero.
    P_h_max = float(sim_inputs.p_heat_max_W)
    P_c_max = float(sim_inputs.p_cool_max_W)
    active_heat = P_h_max > 0
    active_cool = P_c_max > 0

    # Object record for EnTiSe. The setpoint scalars passed here are used
    # only when the (constant-setpoint) EnTiSe fast path is taken; the
    # array solver overrides them. We pass the array initial value to keep
    # init_temperature consistent in both paths.
    init_T_C = float(t_heat_arr[0])
    obj = pd.Series({
        Objects.ID: obj_id,
        "inhabitants": sim_inputs.n_occupants,
        "weather": "weather",
        "area[m2]": sim_inputs.floor_area_m2,
        "height[m]": sim_inputs.height_floor_m,
        "latitude[degree]": sim_inputs.latitude,
        "longitude[degree]": sim_inputs.longitude,
        "min_temperature[C]": float(t_heat_arr.mean()),
        "max_temperature[C]": float(t_cool_arr.mean()),
        "stories": sim_inputs.n_floors,
        "gains_internal[W]": "gains_internal",
        "gains_internal_column": "gains_internal[W]",
        "gains_internal[W]_per_person[W]": GAINS_PER_PERSON_W,
        "ventilation[W K-1]": "ach_series",
        "ventilation_column": "typical [1/h]",
        "init_temperature[C]": init_T_C,
        "windows": "windows",
        "resistance[K W-1]": R,
        "capacitance[J K-1]": C,
        # NREL-matched capacity caps + activation flags.
        "power_heating[W]": P_h_max,
        "power_cooling[W]": P_c_max,
        "active_heating": active_heat,
        "active_cooling": active_cool,
    })
    data = {
        "weather": weather_df,
        "gains_internal": gains_df,
        "ach_series": ach_df,
        "windows": windows_df,
    }

    ts = _run_with_setpoint_arrays(obj, data, t_heat_arr, t_cool_arr)

    return pd.DataFrame({
        "timestamp": ts.index,
        "bldg_id": sim_inputs.bldg_id,
        "q_heat_w_sim": ts[f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"].astype(np.float32),
        "q_cool_w_sim": ts[f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"].astype(np.float32),
    })


def process_county(county_id: str, zone: str, data_dir: Path,
                   metadata_pq: Path,
                   use_setpoint_schedules: bool = True) -> pd.DataFrame:
    """Process one county; return simulated data merged across buildings."""
    zone_safe = zone.replace(" ", "_")
    qual_csv = data_dir / f"{county_id}_{zone_safe}_qualified.csv"
    proc_pq = data_dir / "processed" / f"{county_id}_{zone_safe}.parquet"
    weather_csv = data_dir / "weather" / f"{county_id}_2018.csv"

    if not (qual_csv.exists() and proc_pq.exists() and weather_csv.exists()):
        print(f"[skip] {county_id}: missing inputs"); return pd.DataFrame()

    qualified = pd.read_csv(qual_csv)
    # Try to read all needed fields directly from the qualified CSV (the
    # updated buildings_select.py keeps them). Fall back to metadata.parquet
    # only if a needed column is missing.
    needed_cols = [
        "in.heating_setpoint_offset_period",
        "in.heating_setpoint_offset_magnitude",
        "in.cooling_setpoint_offset_period",
        "in.cooling_setpoint_offset_magnitude",
        "in.infiltration", "in.window_areas",
        "in.vintage", "in.geometry_stories",
        "in.heating_setpoint", "in.cooling_setpoint", "in.sqft",
        "in.hvac_heating_type_and_fuel",
        "in.weather_file_latitude", "in.weather_file_longitude",
        "in.county", "in.state",
        # HVAC nameplate capacities (kBtu/h) — used as per-building
        # power caps for apples-to-apples comparison with NREL.
        "out.params.size_heating_system_primary_k_btu_h",
        "out.params.size_heat_pump_backup_primary_k_btu_h",
        "out.params.size_cooling_system_primary_k_btu_h",
    ]
    missing = [c for c in needed_cols if c not in qualified.columns]
    if not missing:
        md_full = qualified.copy()
        print(f"  using fields from qualified CSV ({len(qualified)} buildings, "
              f"{len(qualified.columns)} columns)")
    else:
        if not metadata_pq.exists():
            print(f"  ERROR: qualified CSV missing columns {missing} and no "
                  f"metadata.parquet at {metadata_pq}.")
            print(f"  Re-run: uv run python validation/us/select_buildings.py")
            return pd.DataFrame()
        print(f"  qualified CSV missing {len(missing)} cols; falling back to "
              f"metadata.parquet")
        md_full = pd.read_parquet(metadata_pq, columns=needed_cols).reset_index()
        md_full = md_full[md_full["bldg_id"].isin(qualified["bldg_id"])].copy()
    schedule_label = "ON" if use_setpoint_schedules else "OFF (constant setpoints)"
    print(f"\n[{county_id} {zone}] {len(md_full)} buildings to simulate "
          f"(setpoint schedules: {schedule_label})")

    weather_df = load_weather(weather_csv)
    nrel_proc = pd.read_parquet(proc_pq)

    out_rows = []
    fits = {}  # cache TEASER fits by (year, area_bucket, n_floors)
    t0 = time.time()
    for i, (_, mrow) in enumerate(md_full.iterrows(), 1):
        try:
            si = map_metadata_row(mrow)
        except Exception as e:
            print(f"  bldg {mrow['bldg_id']}: map failed: {e}")
            continue
        # Cache key: round area to 10 m2 to share TEASER fits
        cache_key = (si.vintage_year, round(si.floor_area_m2 / 10) * 10, si.n_floors)
        if cache_key not in fits:
            fits[cache_key] = fit_teaser_one(*cache_key)
        if fits[cache_key] is None:
            print(f"  bldg {si.bldg_id}: TEASER fit failed for "
                  f"(year={cache_key[0]}, area={cache_key[1]}, "
                  f"floors={cache_key[2]}); skipping.")
            continue
        R, C = fits[cache_key]
        # Pull this building's electricity timeseries from the processed parquet
        sub = nrel_proc[nrel_proc["bldg_id"] == si.bldg_id]
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        elec = sub["electricity_w"].to_numpy().astype(np.float32)
        dt_idx = pd.DatetimeIndex(sub["timestamp"])
        try:
            sim = simulate_building(
                si, weather_df, elec, dt_idx, R, C,
                use_setpoint_schedules=use_setpoint_schedules,
            )
        except Exception as e:
            print(f"  bldg {si.bldg_id}: sim failed: {e}")
            continue
        out_rows.append(sim)
        if i % 25 == 0 or i == len(md_full):
            print(f"  {i}/{len(md_full)} ({time.time()-t0:.0f}s, {len(fits)} TEASER fits cached)")

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--county", type=str, default="all")
    parser.add_argument("--data-dir", type=Path,
        default=Path("validation/us/data"))
    parser.add_argument("--metadata", type=Path,
        default=Path("validation/us/data/metadata.parquet"))
    parser.add_argument(
        "--no-setpoint-schedules", action="store_true",
        help="Disable EULP-derived per-timestep setpoint schedules and use "
             "constant base setpoints instead. By default schedules are ON.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="",
        help="Optional suffix appended to output filename, e.g. '_constSP' "
             "to keep ablation runs side-by-side with the default outputs.",
    )
    args = parser.parse_args()
    use_schedules = not args.no_setpoint_schedules

    out_dir = args.data_dir / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If --county all, shell out to ourselves per county to guarantee a
    # fresh Python process per county (workaround for state leak in
    # EnTiSe/TEASER between back-to-back simulations).
    if args.county == "all":
        import subprocess
        for cid, state, zone, _, _ in COUNTIES:
            print(f"\n=== Spawning fresh Python for {cid} {zone} ===")
            cmd = [sys.executable, __file__,
                   "--county", cid,
                   "--data-dir", str(args.data_dir),
                   "--metadata", str(args.metadata)]
            if args.no_setpoint_schedules:
                cmd.append("--no-setpoint-schedules")
            if args.out_suffix:
                cmd += ["--out-suffix", args.out_suffix]
            r = subprocess.call(cmd)
            if r != 0:
                print(f"  WARNING: non-zero exit for {cid}: {r}")
        return

    counties_to_run = [c for c in COUNTIES if c[0] == args.county]
    if not counties_to_run:
        sys.exit(f"county {args.county} not in COUNTIES list")

    for cid, state, zone, _, _ in counties_to_run:
        zone_safe = zone.replace(" ", "_")
        out_pq = out_dir / f"{cid}_{zone_safe}{args.out_suffix}.parquet"
        sim = process_county(cid, zone, args.data_dir, args.metadata,
                             use_setpoint_schedules=use_schedules)
        if sim.empty:
            continue
        sim.to_parquet(out_pq, index=False, compression="zstd", compression_level=19)
        sz_mb = out_pq.stat().st_size / 1024 / 1024
        print(f"  -> {out_pq} ({sz_mb:.1f} MB)")


if __name__ == '__main__':
    main()
