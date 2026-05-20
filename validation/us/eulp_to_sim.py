"""Map EULP/ResStock metadata fields to simulation inputs for our 1R1C
re-simulation of NREL buildings (US per-building).

Per qualified building this produces:
  - Constant scalars: R [K/W], C [J/K], floor_area [m2], stories, lat, lon
  - Setpoint *time series*: min_temp[°C] (heating) and max_temp[°C] (cooling),
    with EULP-derived offset schedule applied (e.g., "Night -2h" 6 °F setback)
  - Ventilation *time series*: ACH derived from in.infiltration class with
    a small synthetic diurnal pattern
  - HVAC capacity: scalar nameplate, sized as design heat load × 1.4

R, C are derived from TEASER fitted to a German archetype matched by
construction year and floor area (TEASER's tabula_de typology). TABULA is
European-only and does not include a US residential typology, so each EULP
building is mapped to the closest TABULA-DE archetype by vintage and area
to stay consistent with the published German dataset's archetype space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SQFT_TO_M2 = 0.092903
F_TO_C = lambda f: (f - 32.0) * 5.0 / 9.0  # noqa: E731
DEFAULT_FLOOR_HEIGHT_M = 2.5
KBTUH_TO_W = 293.071  # 1 kBtu/h = 293.071 W (exact: 1 Btu = 1055.06 J)


def parse_capacity_kbtuh(value) -> float:
    """Convert ResStock sized-capacity field (kBtu/h) to W.

    Returns 0.0 if missing/NaN/<=0 — the caller is expected to interpret
    a 0 cap as "this building has no system in this mode" (matches the
    NREL autosizer's convention for buildings without an AC or heater).
    Buildings with a real system have nameplate >> 0.
    """
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(f) or f <= 0:
        return 0.0
    return f * KBTUH_TO_W

# Heating/cooling setpoint defaults from EULP if missing
DEFAULT_HEAT_F = 68.0
DEFAULT_COOL_F = 75.0

# ACH50 → natural ACH using N-factor heuristic (Sherman-Grimsrud, ~20 for
# typical US single-storey, ~14 for multi-storey or windy/cold climates).
ACH50_TO_NATURAL = 1.0 / 17.5

# HVAC sizing oversize factor (typical residential field practice)
HVAC_OVERSIZE = 1.4

# ── Glazing SHGC lookup by EULP window class ──────────────────────────────
# Values per ASHRAE 90.1 / NFRC typical glazing SHGCs.
# EULP "in.windows" examples:
#   "Single, Clear, Non-metal"
#   "Double, Clear, Non-metal, Air"
#   "Double, Low-E, Non-metal, Air, M-Gain"
#   "Triple, Low-E, Non-metal, Air, L-Gain"
def parse_window_shgc(window_class: str) -> float:
    if not window_class or str(window_class).strip() in ("", "None", "nan"):
        return 0.55  # neutral default
    s = str(window_class).lower()
    # Detect glazing layers
    if "triple" in s:
        if "low-e" in s:
            if "l-gain" in s: return 0.25
            if "h-gain" in s: return 0.50
            return 0.35  # m-gain or unspecified
        return 0.65
    if "double" in s:
        if "low-e" in s:
            if "l-gain" in s: return 0.30
            if "h-gain" in s: return 0.60
            return 0.40  # m-gain or unspecified
        if "tinted" in s: return 0.50
        return 0.76  # clear double
    if "single" in s:
        if "tinted" in s: return 0.50
        return 0.86  # clear single
    return 0.55  # fallback


# ── Building orientation -> facade-to-compass mapping ─────────────────────
# EULP "in.orientation" specifies which compass direction the FRONT of the
# house faces. Front=F, Back=B, Left=L, Right=R as seen from outside facing
# the front. So if orientation="South", F=S, B=N, L=E, R=W.
_ORIENT_FROM_FRONT = {
    "north":     {"F":"N","B":"S","L":"E","R":"W"},
    "northeast": {"F":"NE","B":"SW","L":"NW","R":"SE"},
    "east":      {"F":"E","B":"W","L":"N","R":"S"},
    "southeast": {"F":"SE","B":"NW","L":"NE","R":"SW"},
    "south":     {"F":"S","B":"N","L":"W","R":"E"},
    "southwest": {"F":"SW","B":"NE","L":"SE","R":"NW"},
    "west":      {"F":"W","B":"E","L":"S","R":"N"},
    "northwest": {"F":"NW","B":"SE","L":"SW","R":"NE"},
}
# Cardinal-to-azimuth (degrees from North, clockwise)
_AZIMUTH = {"N":0.0,"NE":45.0,"E":90.0,"SE":135.0,"S":180.0,
            "SW":225.0,"W":270.0,"NW":315.0}


def parse_orientation(orientation: str) -> dict:
    """Return dict {F,B,L,R -> compass code} given building orientation.
    Falls back to F=S (south-facing front) if missing/unparseable."""
    if not orientation: return _ORIENT_FROM_FRONT["south"]
    key = str(orientation).strip().lower()
    if key in _ORIENT_FROM_FRONT:
        return _ORIENT_FROM_FRONT[key]
    return _ORIENT_FROM_FRONT["south"]


# ── Setpoint parsers ────────────────────────────────────────────────────────

_OFFSET_RE = re.compile(
    r"(?P<period>Night Setback|Night|Day|Day and Night)\s*"
    r"(?P<sign>[+-])?(?P<hours>\d+)?h?", re.IGNORECASE,
)


def parse_setpoint_offset(period: str, magnitude: str) -> dict:
    """Parse EULP 'in.heating_setpoint_offset_period' and '_magnitude'.

    Returns dict with keys:
        magnitude_K : float (positive = setback DOWN for heating, UP for cooling)
        start_h     : int hour-of-day when setback BEGINS (24h clock)
        end_h       : int hour-of-day when setback ENDS
    Empty dict if no offset.
    """
    if not period or str(period).lower() in ("", "none", "nan"):
        return {}
    if not magnitude:
        return {}
    mag_str = str(magnitude).strip().rstrip("F")
    try:
        mag_F = float(mag_str)
    except ValueError:
        return {}
    if mag_F <= 0:
        return {}
    mag_K = mag_F * 5.0 / 9.0

    # Default offset windows:
    #   Night        : 22:00 -> 06:00
    #   Day          : 09:00 -> 17:00
    #   Day and Night: combined day + night windows (we model as continuous
    #                  off-peak setback — close enough for our purposes)
    p_low = str(period).lower()
    base_start, base_end = 22, 6
    if "day and night" in p_low:
        base_start, base_end = 18, 9      # extended off-peak
    elif "day" in p_low and "night" not in p_low:
        base_start, base_end = 9, 17

    # Time shift (e.g. "Night -2h" means start 2 hours earlier)
    m = re.search(r"([+-])\s*(\d+)\s*h", str(period))
    shift_h = 0
    if m:
        sgn = -1 if m.group(1) == "-" else 1
        shift_h = sgn * int(m.group(2))
    start_h = (base_start + shift_h) % 24
    end_h = (base_end + shift_h) % 24
    return {"magnitude_K": mag_K, "start_h": start_h, "end_h": end_h}


def build_setpoint_series(
    timestamps: pd.DatetimeIndex,
    base_setpoint_F: float,
    offset_dict: dict,
    is_heating: bool,
) -> np.ndarray:
    """Build a per-timestep setpoint series in °C with the EULP offset applied.

    For heating: setback subtracts magnitude during the offset window.
    For cooling: setup adds magnitude during the offset window.
    """
    base_C = F_TO_C(base_setpoint_F)
    out = np.full(len(timestamps), base_C, dtype=np.float32)
    if not offset_dict:
        return out
    h = timestamps.hour.values
    s, e = offset_dict["start_h"], offset_dict["end_h"]
    if s <= e:
        in_window = (h >= s) & (h < e)
    else:
        in_window = (h >= s) | (h < e)
    delta = offset_dict["magnitude_K"]
    if is_heating:
        out[in_window] -= delta
    else:
        out[in_window] += delta
    return out


# ── Infiltration / ventilation ──────────────────────────────────────────────

def parse_infiltration_ach50(in_infil: str) -> float:
    """'15 ACH50' -> 15.0; default 15."""
    if not in_infil:
        return 15.0
    m = re.search(r"(\d+(\.\d+)?)\s*ACH50", str(in_infil))
    return float(m.group(1)) if m else 15.0


def build_ventilation_series(
    timestamps: pd.DatetimeIndex,
    ach50: float,
    volume_m3: float | None = None,  # kept for signature compatibility
) -> np.ndarray:
    """ACH50 → natural ACH series (1/h) with a mild diurnal pattern.

    EnTiSe expects ventilation in 1/h (it computes the W/K conductance
    internally from building volume), so we return ACH directly.
    """
    natural_ach = ach50 * ACH50_TO_NATURAL
    h = timestamps.hour.values
    diurnal = 1.0 + 0.15 * np.cos(2 * np.pi * (h - 14) / 24.0)
    return (natural_ach * diurnal).astype(np.float32)


# ── Geometry ────────────────────────────────────────────────────────────────

def vintage_to_year(vintage: str) -> int:
    """'1970s' -> 1975, '<1940' -> 1930, '2000s' -> 2005, etc."""
    if not vintage:
        return 1970
    s = str(vintage).strip()
    if s.startswith("<"):
        try:
            return int(s.lstrip("<")) - 10
        except ValueError:
            return 1930
    m = re.match(r"(\d{4})s", s)
    if m:
        return int(m.group(1)) + 5
    try:
        return int(s)
    except ValueError:
        return 1970


def parse_window_areas(spec: str, total_floor_area_m2: float,
                       orientation: str = "South") -> dict:
    """Parse EULP window_areas (e.g. 'F18 B18 L18 R18') and map F/B/L/R to
    actual compass orientation based on building orientation. Returns dict
    of compass-code -> area in m^2.
    """
    side_map = parse_orientation(orientation)
    if not spec:
        per = 0.15 * total_floor_area_m2 / 4.0
        return {"N": per, "E": per, "S": per, "W": per}
    out: dict[str, float] = {}
    for token in str(spec).split():
        m = re.match(r"([FBLR])(\d+(?:\.\d+)?)", token)
        if not m:
            continue
        side, val = m.group(1), float(m.group(2))
        compass = side_map[side]
        out[compass] = out.get(compass, 0.0) + val * SQFT_TO_M2
    if not out:
        per = 0.15 * total_floor_area_m2 / 4.0
        return {"N": per, "E": per, "S": per, "W": per}
    # Sanity check: if EULP encoding is actually WWR-percent (ambiguous in
    # dict), our ft^2 interpretation could be wrong. If total > 50% of
    # floor area, fall back to a 15% default split per cardinal direction.
    if sum(out.values()) > 0.5 * total_floor_area_m2:
        per = 0.15 * total_floor_area_m2 / 4.0
        return {"N": per, "E": per, "S": per, "W": per}
    return out


# ── HVAC capacity ───────────────────────────────────────────────────────────

def estimate_hvac_capacity_w(
    R: float, t_set_K: float, t_design_K: float,
    floor_area_m2: float, oversize: float = HVAC_OVERSIZE,
) -> float:
    """Estimate nameplate heating capacity from (T_set - T_design) / R + small
    margin for ventilation, multiplied by oversize factor (typical 1.4)."""
    delta = t_set_K - t_design_K
    p_design = delta / R if R > 0 else 5000.0
    return p_design * oversize


# ── Main mapping function ───────────────────────────────────────────────────

@dataclass
class SimInputs:
    bldg_id: int
    state: str
    county_id: str
    latitude: float
    longitude: float
    floor_area_m2: float
    n_floors: int
    height_floor_m: float
    volume_m3: float
    vintage_year: int
    is_heat_pump: bool
    R_K_per_W: float
    C_J_per_K: float
    ach50: float
    base_heat_F: float
    base_cool_F: float
    heat_offset: dict
    cool_offset: dict
    window_areas_m2: dict
    window_shgc: float          # solar heat gain coefficient (0..1)
    n_occupants: int            # from EULP in.occupants
    p_heat_max_W: float
    p_cool_max_W: float


def map_metadata_row(row: pd.Series) -> SimInputs:
    """Map one row of EULP qualified metadata to SimInputs (sans R, C)."""
    sqft = float(row.get("in.sqft", 1500.0) or 1500.0)
    floor_area_m2 = sqft * SQFT_TO_M2
    try:
        n_floors = int(row.get("in.geometry_stories", 1) or 1)
    except (ValueError, TypeError):
        n_floors = 1
    volume_m3 = floor_area_m2 * DEFAULT_FLOOR_HEIGHT_M * n_floors

    base_heat_F = float(str(row.get("in.heating_setpoint", "68F")).rstrip("F") or DEFAULT_HEAT_F)
    base_cool_F = float(str(row.get("in.cooling_setpoint", "75F")).rstrip("F") or DEFAULT_COOL_F)

    heat_off = parse_setpoint_offset(
        row.get("in.heating_setpoint_offset_period", ""),
        row.get("in.heating_setpoint_offset_magnitude", ""),
    )
    cool_off = parse_setpoint_offset(
        row.get("in.cooling_setpoint_offset_period", ""),
        row.get("in.cooling_setpoint_offset_magnitude", ""),
    )

    ach50 = parse_infiltration_ach50(row.get("in.infiltration", ""))
    orient = str(row.get("in.orientation", "South") or "South")
    win_a = parse_window_areas(row.get("in.window_areas", ""), floor_area_m2, orient)
    win_shgc = parse_window_shgc(row.get("in.windows", ""))

    htype = str(row.get("in.hvac_heating_type_and_fuel", ""))
    is_hp = bool(re.search(r"Heat Pump|ASHP|MSHP|GSHP", htype))

    # HVAC capacity from NREL ResStock autosizing (kBtu/h -> W). For
    # heat-pump buildings the total heating capacity is primary + backup
    # electric resistance; for non-HP buildings backup is 0.
    p_heat_W = (
        parse_capacity_kbtuh(row.get("out.params.size_heating_system_primary_k_btu_h"))
        + parse_capacity_kbtuh(row.get("out.params.size_heat_pump_backup_primary_k_btu_h"))
    )
    p_cool_W = parse_capacity_kbtuh(row.get("out.params.size_cooling_system_primary_k_btu_h"))

    # Per-building occupants from EULP in.occupants (default 2 if missing).
    try:
        n_occ = int(round(float(row.get("in.occupants", 2.0) or 2.0)))
    except (ValueError, TypeError):
        n_occ = 2
    n_occ = max(1, n_occ)

    try:
        lat = float(row.get("in.weather_file_latitude", 42.0) or 42.0)
        lon = float(row.get("in.weather_file_longitude", -86.0) or -86.0)
    except (ValueError, TypeError):
        lat, lon = 42.0, -86.0
    return SimInputs(
        bldg_id=int(row["bldg_id"]),
        state=str(row.get("in.state", "")),
        county_id=str(row.get("in.county", "")),
        latitude=lat,
        longitude=lon,
        floor_area_m2=floor_area_m2,
        n_floors=n_floors,
        height_floor_m=DEFAULT_FLOOR_HEIGHT_M,
        volume_m3=volume_m3,
        vintage_year=vintage_to_year(row.get("in.vintage", "")),
        is_heat_pump=is_hp,
        R_K_per_W=0.0,    # to be filled via TEASER
        C_J_per_K=0.0,    # to be filled via TEASER
        ach50=ach50,
        base_heat_F=base_heat_F,
        base_cool_F=base_cool_F,
        heat_offset=heat_off,
        cool_offset=cool_off,
        window_areas_m2=win_a,
        window_shgc=win_shgc,
        n_occupants=n_occ,
        p_heat_max_W=p_heat_W,
        p_cool_max_W=p_cool_W,
    )
