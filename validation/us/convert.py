"""Step 3: convert raw EULP per-building parquets to clean per-county
timeseries with thermal heating, thermal cooling, and total electricity
columns in W (15-min resolution).

Key conversions:
  - Fossil heating (gas, oil, propane): thermal = fuel * AFUE
  - Electric resistance heating: thermal = electric (1:1)
  - Heat pump heating: thermal = electric * COP(T_out)  via Ruhnau et al. 2019
  - Cooling thermal: electric * EER (constant approx; see notes)
  - Electricity total: sum of all electricity end-uses, EXCLUDING heating &
    cooling electric inputs (so we have the "household" electricity you'd
    measure at the meter without HVAC).

Output: one parquet per county at <data>/processed/<county>_<zone>.parquet
        Schema: timestamp, bldg_id, electricity_w, q_heat_w, q_cool_w
        Float32, zstd compression level 19.

Per-county processing keeps the working set small. Parallel over buildings
within county via ThreadPoolExecutor.

Run after select_buildings + download_data:
    uv run python validation/us/convert.py
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from COUNTIES import COUNTIES

# ── Constants ─────────────────────────────────────────────────────────────────

AFUE_FOSSIL = 0.92        # gas/oil/propane combustion efficiency

# ── Cooling efficiency: per-building rated EER + outdoor-T correction ────────
#
# EULP's `in.hvac_cooling_efficiency` is a string like:
#   "AC, SEER 13"           : Central AC with SEER (seasonal) rating
#   "AC, SEER 21"           : Higher-efficiency variable-speed AC
#   "Room AC, EER 10.7"     : Window/wall unit with rated EER (BTU/Wh)
#   "Heat Pump, SEER 14"    : Heat pump in cooling mode (use SEER like AC)
#   "None"                  : No cooling
#
# Conversions:
#   - EER [W/W] = EER [BTU/Wh] / 3.412
#   - SEER [BTU/Wh] -> rated EER [BTU/Wh]: industry rule SEER * 0.875
#     (regression fit from AHRI test data; works for SEER 8-15; slightly
#     under-counts variable-speed SEER 18+, conservative.)
#
# Temperature correction (per AHRI 210/240 / ASHRAE 90.1 sensitivity):
#   EER(T_out) = EER_rated * max(0.5, 1 - 0.0085 * (T_out_C - 35))
#   This gives ~9% EER gain at T_out=25, ~9% loss at T_out=45, clipped at 0.5
#   to prevent unphysical low EER for extreme temperatures.

EER_FALLBACK = 3.0        # if metadata is missing/unparseable, use this
SEER_TO_EER_FACTOR = 0.875    # rule-of-thumb regression
BTU_WH_TO_W_W = 1.0 / 3.412   # SEER/EER are BTU/Wh; this converts to W/W
EER_T_RATED_C = 35.0          # AHRI test outdoor T
EER_T_SENSITIVITY = 0.0085    # fractional EER change per K from rated T
EER_T_FLOOR = 0.5             # min fractional EER under extreme T_out


def _parse_eer_w_per_w(efficiency: str) -> float:
    """Parse EULP cooling efficiency string -> rated EER in W/W."""
    import re
    if not efficiency or str(efficiency).strip() in ("None", "nan", ""):
        return EER_FALLBACK
    s = str(efficiency)
    # Try SEER first (e.g. "AC, SEER 13" or "Heat Pump, SEER 14")
    m = re.search(r"SEER\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        seer_btu = float(m.group(1))
        eer_btu = seer_btu * SEER_TO_EER_FACTOR
        return eer_btu * BTU_WH_TO_W_W
    # Else try direct EER (e.g. "Room AC, EER 10.7")
    m = re.search(r"EER\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        eer_btu = float(m.group(1))
        return eer_btu * BTU_WH_TO_W_W
    return EER_FALLBACK


def _eer_t_corrected(eer_rated_w_w: float, t_out_c) -> "np.ndarray":
    """Temperature-corrected EER series (W/W) at the given outdoor T."""
    correction = 1.0 - EER_T_SENSITIVITY * (t_out_c - EER_T_RATED_C)
    correction = np.maximum(correction, EER_T_FLOOR)
    return eer_rated_w_w * correction

# COP curves (Ruhnau, Hirth & Praktiknjo 2019, Sci. Data) for heat pumps.
# COP = a*ΔT² + b*ΔT + c, where ΔT = T_supply - T_outdoor [°C].
# Supply T depends on heat-distribution: radiator (40 - T_out) or floor (30 - 0.5 T_out).
HP_COP_PARAMS = {
    "air":    (0.0005,  -0.09,  6.80),
    "ground": (0.0012,  -0.21, 10.29),
    "water":  (0.0012,  -0.20,  9.97),
}

# Source (per fuel-type heuristic; ASHP is by far the most common in EULP)
DEFAULT_HP_SOURCE = "air"
DEFAULT_DISTRIBUTION = "radiator"

# Energy units in EULP: kWh per 15-minute interval.
# Power [W] = energy [kWh] * 1000 / 0.25 = energy * 4000
KWH_15MIN_TO_W = 4000.0


# ── Heat-pump conversion ──────────────────────────────────────────────────────

def hp_cop(t_out_c: np.ndarray, source: str = "air",
           distribution: str = "radiator") -> np.ndarray:
    """Per-Ruhnau COP curve. Clipped to [1.0, 8.0] for stability."""
    if distribution == "floor":
        t_supply = 30.0 - 0.5 * t_out_c
    else:  # radiator
        t_supply = 40.0 - t_out_c
    delta = t_supply - t_out_c
    a, b, c = HP_COP_PARAMS[source]
    cop = a * delta**2 + b * delta + c
    return np.clip(cop, 1.0, 8.0)


# ── Per-building conversion ───────────────────────────────────────────────────

HEATING_FUEL_COLS_FOSSIL = (
    "out.natural_gas.heating.energy_consumption",
    "out.fuel_oil.heating.energy_consumption",
    "out.propane.heating.energy_consumption",
)
HEATING_ELEC_COL = "out.electricity.heating.energy_consumption"
HEATING_HP_BACKUP_COL = "out.electricity.heating_hp_bkup.energy_consumption"
HEATING_FANS_COL = "out.electricity.heating_fans_pumps.energy_consumption"

COOLING_ELEC_COL = "out.electricity.cooling.energy_consumption"
COOLING_FANS_COL = "out.electricity.cooling_fans_pumps.energy_consumption"

# Explicit 17-column "power" list (matches thesis pipeline / Columns.xlsx).
# This is the household behavioural electricity used to drive occupancy
# detection; intentionally excludes HVAC electric (heating/cooling),
# domestic hot water, and EV charging.
POWER_COLS_17 = (
    "out.electricity.ceiling_fan.energy_consumption",
    "out.electricity.clothes_dryer.energy_consumption",
    "out.electricity.clothes_washer.energy_consumption",
    "out.electricity.dishwasher.energy_consumption",
    "out.electricity.freezer.energy_consumption",
    "out.electricity.hot_tub_heater.energy_consumption",
    "out.electricity.hot_tub_pump.energy_consumption",
    "out.electricity.lighting_exterior.energy_consumption",
    "out.electricity.lighting_garage.energy_consumption",
    "out.electricity.lighting_interior.energy_consumption",
    "out.electricity.mech_vent.energy_consumption",
    "out.electricity.plug_loads.energy_consumption",
    "out.electricity.pool_heater.energy_consumption",
    "out.electricity.pool_pump.energy_consumption",
    "out.electricity.range_oven.energy_consumption",
    "out.electricity.refrigerator.energy_consumption",
    "out.electricity.well_pump.energy_consumption",
)


def _is_heat_pump(meta_row: pd.Series) -> bool:
    htype = str(meta_row.get("in.hvac_heating_type_and_fuel", ""))
    return ("Heat Pump" in htype) or ("ASHP" in htype) or ("MSHP" in htype) \
        or ("GSHP" in htype)


def convert_building(
    raw_path: Path, weather: pd.DataFrame, meta_row: pd.Series,
) -> pd.DataFrame:
    """Read one EULP raw parquet, return DataFrame with [timestamp, bldg_id,
    electricity_w, q_heat_w, q_cool_w]. All powers in W."""
    bldg_id = int(meta_row["bldg_id"])
    cols = [
        "timestamp",
        *POWER_COLS_17,
        HEATING_ELEC_COL, HEATING_HP_BACKUP_COL, HEATING_FANS_COL,
        COOLING_ELEC_COL, COOLING_FANS_COL,
        *HEATING_FUEL_COLS_FOSSIL,
    ]
    df = pd.read_parquet(raw_path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Align weather to the building timestamps (15-min interpolation).
    w = weather.set_index("timestamp").reindex(
        df["timestamp"], method="nearest", tolerance=pd.Timedelta("60min"),
    )
    t_out_c = w["air_temperature_c"].astype(np.float32).to_numpy()

    # Heating thermal [kWh per 15min]
    fossil_kwh = sum(df[c].fillna(0.0) for c in HEATING_FUEL_COLS_FOSSIL)
    fossil_thermal_kwh = AFUE_FOSSIL * fossil_kwh

    elec_heat_kwh = df[HEATING_ELEC_COL].fillna(0.0)
    hp_backup_kwh = df[HEATING_HP_BACKUP_COL].fillna(0.0)

    if _is_heat_pump(meta_row):
        cop = hp_cop(t_out_c, DEFAULT_HP_SOURCE, DEFAULT_DISTRIBUTION)
        # HP compressor electric * COP, plus resistance backup (1:1)
        elec_thermal_kwh = elec_heat_kwh.to_numpy() * cop
        elec_thermal_kwh = pd.Series(elec_thermal_kwh, index=df.index)
        elec_thermal_kwh = elec_thermal_kwh + hp_backup_kwh
    else:
        # Resistance: thermal = electric (HP backup is 0 for non-HP buildings)
        elec_thermal_kwh = elec_heat_kwh + hp_backup_kwh

    q_heat_kwh = fossil_thermal_kwh + elec_thermal_kwh

    # Cooling thermal [kWh per 15min]: electric input * EER(T_out)
    # Per-building rated EER from metadata, T-corrected per timestep.
    elec_cool_kwh = df[COOLING_ELEC_COL].fillna(0.0)
    eer_rated = _parse_eer_w_per_w(meta_row.get("in.hvac_cooling_efficiency", ""))
    eer_series = _eer_t_corrected(eer_rated, t_out_c)
    q_cool_kwh = eer_series * elec_cool_kwh.to_numpy()

    # Household electricity [kWh per 15min]: explicit sum of the 17 "power"
    # end-uses (lighting, plug loads, large appliances, ventilation, well/pool
    # pumps, hot tub). Matches the thesis pipeline and Columns.xlsx category
    # "power". Excludes HVAC electric, DHW, and EV charging.
    elec_household_kwh = sum(df[c].fillna(0.0) for c in POWER_COLS_17)

    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "bldg_id": np.int32(bldg_id),
        "electricity_w": (elec_household_kwh * KWH_15MIN_TO_W).astype(np.float32),
        "q_heat_w":      (q_heat_kwh         * KWH_15MIN_TO_W).astype(np.float32),
        "q_cool_w":      (q_cool_kwh         * KWH_15MIN_TO_W).astype(np.float32),
    })
    return out


# ── Per-county driver ─────────────────────────────────────────────────────────

def _load_weather(weather_csv: Path) -> pd.DataFrame:
    w = pd.read_csv(weather_csv)
    # EULP weather: 'date_time' column, hourly. Standardise.
    ts_col = "date_time" if "date_time" in w.columns else (
        "Date/Time" if "Date/Time" in w.columns else w.columns[0]
    )
    t_col = next((c for c in w.columns
                  if "Drybulb" in c or "drybulb" in c
                  or "air_temperature" in c.lower()), w.columns[1])
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(w[ts_col]),
        "air_temperature_c": pd.to_numeric(w[t_col], errors="coerce"),
    }).dropna()
    return out


def convert_county(
    county_id: str, zone: str, raw_dir: Path, weather_dir: Path,
    qualified_csv: Path, processed_dir: Path,
) -> dict:
    zone_safe = zone.replace(" ", "_")
    in_dir = raw_dir / f"{county_id}_{zone_safe}"
    weather_csv = weather_dir / f"{county_id}_2018.csv"
    out_path = processed_dir / f"{county_id}_{zone_safe}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not weather_csv.exists():
        return {"county_id": county_id, "status": "no_weather"}
    if not in_dir.exists():
        return {"county_id": county_id, "status": "no_raw"}

    meta = pd.read_csv(qualified_csv)
    meta["bldg_id"] = meta["bldg_id"].astype(int)
    weather = _load_weather(weather_csv)

    print(f"\n[{county_id} {zone}] converting {len(meta)} buildings...")
    t0 = time.time()
    parts: list[pd.DataFrame] = []
    n_fail = 0
    for _, row in meta.iterrows():
        raw_path = in_dir / f"{int(row['bldg_id'])}-0.parquet"
        if not raw_path.exists():
            n_fail += 1
            continue
        try:
            parts.append(convert_building(raw_path, weather, row))
        except Exception as e:
            n_fail += 1
            print(f"  bldg {row['bldg_id']}: {e}")

    if not parts:
        return {"county_id": county_id, "status": "no_data", "n_fail": n_fail}

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_path, index=False, compression="zstd",
                  compression_level=19)
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {out_path} ({sz_mb:.1f} MB, {len(parts)} buildings, "
          f"{n_fail} failed) in {time.time()-t0:.1f}s")

    return {
        "county_id": county_id, "zone": zone,
        "n_buildings_ok": len(parts), "n_buildings_failed": n_fail,
        "out_path": str(out_path), "size_mb": sz_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
        default=Path("validation/us/data"))
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    weather_dir = args.data_dir / "weather"
    processed_dir = args.data_dir / "processed"

    summary = []
    for county_id, state, zone, _expected, _label in COUNTIES:
        zone_safe = zone.replace(" ", "_")
        qcsv = args.data_dir / f"{county_id}_{zone_safe}_qualified.csv"
        if not qcsv.exists():
            print(f"[skip] {qcsv} not found")
            continue
        summary.append(convert_county(
            county_id, zone, raw_dir, weather_dir, qcsv, processed_dir
        ))

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(args.data_dir / "convert_summary.csv", index=False)
        print("\n=== Convert summary ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
