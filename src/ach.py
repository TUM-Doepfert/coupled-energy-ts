"""Air-change-rate (ACH) models for the thermal simulation.

Two built-in models, selected via the YAML key ``simulation.ach_model``:

- ``"sinusoid"`` (legacy) — deterministic seasonal sinusoid, identical for
  every (location, archetype, profile) triple. Cheap. Used in the
  initial Germany 2010 release; published for reproducibility of that
  release.
- ``"rule_based"`` (default) — presence + outdoor-temperature + season +
  per-profile night-window scenario (S1/S2/S3). More physically motivated.
  Adds per-profile diversity to ventilation losses, eliminating the
  methodological gap between the dataset and its validation. Per-profile
  scenario assignment uses ``profile_id % 3``.

Comfort-related temperature thresholds (spring/summer presence triggers,
heat-guard, free-cooling evening cap) are driven by the indoor-comfort
setpoints ``heating_setpoint_C`` and ``cooling_setpoint_C`` so that a
single change in the YAML config propagates consistently across both the
HVAC setpoint logic in the thermal simulator and the ventilation rule
here. Hysteresis offsets, dwell times, ACH level constants, and the
night free-cooling weather window [18, 22] °C are engineering or physics
parameters that stay as module-level constants.

To plug in a custom model, add a branch in ``build_ach_series`` (or refactor
to a Protocol if more than three models exist).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Sinusoid (legacy) ────────────────────────────────────────────────────────

_ACH_BASE = 0.5
_ACH_SEASON_AMP = 0.20


def _build_sinusoid(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sinusoidal synthetic ACH timeseries aligned to dt_index.

    Temperate-climate parameters (base 0.5, seasonal amp 0.2). A fixed
    RNG seed keeps it reproducible across workers.
    """
    day = dt_index.dayofyear.values
    hour = dt_index.hour.values + dt_index.minute.values / 60.0
    rng = np.random.default_rng(seed=0)
    ach = (
        _ACH_BASE
        + _ACH_SEASON_AMP * np.cos(2 * np.pi * (day - 200) / 365)
        + 0.3 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 0.05, len(dt_index))
    )
    df = pd.DataFrame({"typical [1/h]": np.maximum(ach, 0.0)}, index=dt_index)
    df.index.name = "datetime"
    return df


# ── Rule-based (presence + outdoor-T + season + per-profile scenario) ────────

# ACH levels (1/h)
_LOW = 0.4
_MIN = 0.5
_BASE = 1.1
_MAX = 2.8

# Dwell on hysteresis state transitions (hours)
_DWELL_HOURS = 2

# Hysteresis deadbands relative to the heating setpoint (°C). Off threshold
# = heating_setpoint_C - K. Spring uses a tighter deadband than summer.
_HYSTERESIS_K_SPRING = 1.0
_HYSTERESIS_K_SUMMER = 2.0

# Night free-cooling "useful weather" window: night-minimum outdoor
# temperature must lie in this band for the night to commit to MAX.
# Below the lower bound the indoor mass cools fast enough without forced
# ventilation; above the upper bound the night air can no longer offer
# useful cooling. These are physics/behavior thresholds and intentionally
# not tied to the comfort setpoints.
_NIGHT_COOL_T_MIN = 18.0
_NIGHT_COOL_T_MAX = 22.0

# Summer night windows: (weekday_start, weekday_end, weekend_start, weekend_end)
# Hours expressed as start-inclusive, end-exclusive on the clock; the night
# wraps past midnight when start > end.
_SUMMER_NIGHT_WINDOWS = {
    "S1": (21,  6, 22,  7),
    "S2": (22,  7, 23,  8),
    "S3": (23,  8, 23,  8),
}
_SCENARIOS = ("S1", "S2", "S3")


def _is_in_window(hours: np.ndarray, start_h: int, end_h: int) -> np.ndarray:
    if start_h <= end_h:
        return (hours >= start_h) & (hours < end_h)
    return (hours >= start_h) | (hours < end_h)


def _hysteresis(cond_on: np.ndarray, cond_off: np.ndarray, dwell_h: int) -> np.ndarray:
    n = cond_on.size
    on = np.zeros(n, dtype=bool)
    state = False
    dwell = 0
    for i in range(n):
        if state:
            if cond_off[i]:
                dwell += 1
                if dwell >= dwell_h:
                    state = False
                    dwell = 0
            else:
                dwell = 0
        else:
            if cond_on[i]:
                dwell += 1
                if dwell >= dwell_h:
                    state = True
                    dwell = 0
            else:
                dwell = 0
        on[i] = state
    return on


def _commit_summer_nights(
    ach: np.ndarray,
    T_out: np.ndarray,
    is_summer: np.ndarray,
    hours: np.ndarray,
    is_weekend: np.ndarray,
    scenario: str,
    cooling_setpoint_C: float,
) -> np.ndarray:
    wd_s, wd_e, we_s, we_e = _SUMMER_NIGHT_WINDOWS[scenario]
    in_wd_night = is_summer & ~is_weekend & _is_in_window(hours, wd_s, wd_e)
    in_we_night = is_summer &  is_weekend & _is_in_window(hours, we_s, we_e)
    in_night = in_wd_night | in_we_night

    # Identify contiguous night runs (state machine); commit ACH per run.
    n = ach.size
    out = ach.copy()
    i = 0
    while i < n:
        if in_night[i]:
            j = i
            while j < n and in_night[j]:
                j += 1
            T_night = T_out[i:j]
            if (
                len(T_night) > 0
                and T_night.min() >= _NIGHT_COOL_T_MIN
                and T_night.min() <= _NIGHT_COOL_T_MAX
                and T_out[i] < cooling_setpoint_C
            ):
                out[i:j] = _MAX
            else:
                out[i:j] = _MIN
            i = j
        else:
            i += 1
    return out


def _build_rule_based(
    dt_index: pd.DatetimeIndex,
    T_out: np.ndarray,
    profile_id: int,
    heating_setpoint_C: float,
    cooling_setpoint_C: float,
) -> pd.DataFrame:
    """Rule-based ACH with per-profile scenario S1/S2/S3.

    Scenario assignment is deterministic: ``profile_id % 3 -> S1/S2/S3``.
    Comfort-range bounds derive from the YAML-configured setpoints, so the
    ventilation rule and the HVAC setpoint logic always agree.
    """
    if T_out is None or len(T_out) != len(dt_index):
        raise ValueError(
            "rule_based ACH needs an outdoor-temperature array aligned to dt_index"
        )
    scenario = _SCENARIOS[int(profile_id) % 3]

    T_on = float(heating_setpoint_C)
    T_off_spring = T_on - _HYSTERESIS_K_SPRING
    T_off_summer = T_on - _HYSTERESIS_K_SUMMER
    T_max = float(cooling_setpoint_C)

    hours = dt_index.hour.values
    months = dt_index.month.values
    weekday = dt_index.weekday.values
    is_weekend = weekday >= 5

    is_winter = (months == 12) | (months <= 2)
    is_autumn = (months >= 9) & (months <= 11)
    is_spring = (months >= 3) & (months <= 5)
    is_summer = (months >= 6) & (months <= 8)

    ach = np.full(len(dt_index), _LOW, dtype=np.float32)

    # Spring rule: presence + outdoor-T hysteresis (on T_on, off T_off_spring).
    hot_on_spring = _hysteresis(T_out >= T_on, T_out <= T_off_spring, _DWELL_HOURS)
    spring_pres_wd = is_spring & ~is_weekend & (
        ((hours >= 6) & (hours < 9)) | ((hours >= 17) & (hours < 22))
    )
    spring_pres_we = is_spring & is_weekend & (hours >= 8) & (hours < 22)
    spring_present = spring_pres_wd | spring_pres_we
    ach[spring_present] = np.where(hot_on_spring[spring_present], _BASE, _MIN)
    spring_day_off = is_spring & ~is_weekend & (hours >= 9) & (hours < 17)
    ach[spring_day_off] = _MIN

    # Summer rule: night commit + presence + comfort range + hysteresis
    # (on T_on, off T_off_summer).
    ach = _commit_summer_nights(
        ach, T_out, is_summer, hours, is_weekend, scenario, T_max,
    )
    hot_on_summer = _hysteresis(T_out >= T_on, T_out <= T_off_summer, _DWELL_HOURS)
    in_comfort = (T_out >= T_on) & (T_out <= T_max)
    hot_guard = T_out >= T_max
    # Presence windows for summer days: weekday 06-22, weekend 08-22.
    summer_day_wd = is_summer & ~is_weekend & ((hours >= 6) & (hours < 22))
    summer_day_we = is_summer &  is_weekend & ((hours >= 8) & (hours < 22))
    summer_day = summer_day_wd | summer_day_we
    # Daytime: BASE if in comfort range AND hysteresis on, else MIN.
    day_active = summer_day & in_comfort & hot_on_summer
    ach[day_active] = _BASE
    ach[summer_day & hot_guard] = _MIN

    # Smoothing: centered median-3 on non-MAX values.
    ach_smoothed = ach.copy()
    not_max = ach != _MAX
    if ach.size >= 3:
        triple = np.stack([
            np.roll(ach, 1),
            ach,
            np.roll(ach, -1),
        ], axis=0)
        med = np.median(triple, axis=0)
        ach_smoothed[not_max] = med[not_max]
        ach_smoothed[0] = ach[0]
        ach_smoothed[-1] = ach[-1]

    df = pd.DataFrame({"typical [1/h]": ach_smoothed.astype(float)}, index=dt_index)
    df.index.name = "datetime"
    return df


# ── Dispatcher ───────────────────────────────────────────────────────────────

def build_ach_series(
    model: str,
    dt_index: pd.DatetimeIndex,
    T_out: np.ndarray | None = None,
    profile_id: int | None = None,
    heating_setpoint_C: float = 20.0,
    cooling_setpoint_C: float = 26.0,
) -> pd.DataFrame:
    """Build an ACH series for one (location, archetype, profile) triple.

    Parameters
    ----------
    model : "sinusoid" or "rule_based"
    dt_index : tz-naive DatetimeIndex aligned to the simulation grid
    T_out : outdoor air temperature in °C, length-matched to dt_index
            (required for "rule_based"; ignored for "sinusoid")
    profile_id : integer profile_id used to assign the S1/S2/S3 night
                 scenario in "rule_based" (ignored for "sinusoid")
    heating_setpoint_C : indoor heating setpoint (°C). Drives the spring
            and summer BASE triggers and the lower edge of the comfort
            range. Defaults to the EN 16798-1 Category II residential
            value (20 °C). (ignored for "sinusoid")
    cooling_setpoint_C : indoor cooling setpoint (°C). Drives the upper
            edge of the comfort range, the heat-guard, and the free-cooling
            evening cap. Defaults to the EN 16798-1 Category II residential
            value (26 °C). (ignored for "sinusoid")
    """
    if model == "sinusoid":
        return _build_sinusoid(dt_index)
    if model == "rule_based":
        if T_out is None or profile_id is None:
            raise ValueError(
                "rule_based ACH requires T_out and profile_id"
            )
        return _build_rule_based(
            dt_index,
            np.asarray(T_out, dtype=float),
            int(profile_id),
            float(heating_setpoint_C),
            float(cooling_setpoint_C),
        )
    raise ValueError(
        f"Unknown ACH model {model!r}; valid: 'sinusoid', 'rule_based'"
    )
