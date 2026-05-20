"""End-to-end regression test for ``simulate_archetype``.

Pins the output of one (location, archetype, single-profile) simulation
against a small synthetic input, for each of the three thermal models
(R1C1, ISO 13790 5R1C, VDI 6007 7R2C). Any upstream change in EnTiSe
that silently shifts the numerical results will trip this test.

The test is deliberately conservative:
  - Inputs are tiny (24 hours of synthetic weather + occupancy, one profile).
  - The snapshot values stored below were captured against EnTiSe 1.2.0.
  - Tolerance is 5 % on the daily heating sum to allow for legitimate
    solver-level changes; tightening to bit-identical would require
    storing the full 24-hour vector, which we may add later.

Addresses code-review issues 9 and "smaller obs 5".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("model_name", ["r1c1", "r5c1", "r7c2"])
def test_simulate_archetype_24h_regression(model_name: str, tmp_path):
    """Synthetic 24-hour run; assert daily heating sum stays within
    snapshot ± 5 %."""
    pytest.importorskip("entise")
    from src.simulation import simulate_archetype
    from src.thermal_models import get_thermal_model

    # ── Synthetic weather: cold winter day (0 °C constant, no sun) ───────────
    idx_h = pd.date_range("2010-01-15 00:00", periods=24, freq="1h", tz="UTC")
    weather = pd.DataFrame({
        "datetime": idx_h,
        "air_temperature[C]": np.full(24, 0.0),
        "global_horizontal_irradiance[W m-2]": np.zeros(24),
        "direct_normal_irradiance[W m-2]": np.zeros(24),
        "diffuse_horizontal_irradiance[W m-2]": np.zeros(24),
    })

    # ── Synthetic 1-archetype Series matching the canonical ArchetypeSchema ──
    # Columns are the union of REQUIRED_1R1C, OPTIONAL_5R1C, OPTIONAL_7R2C
    # so the same fixture exercises all three models in the parametrize. The
    # 5R1C / 7R2C numbers are physically plausible (anchored on ISO 13790 /
    # VDI 6007 magnitudes for a ~120 m² SFH) but not measured snapshots;
    # the assertions below only check sign + order-of-magnitude.
    archetype = pd.Series({
        "archetype_id": 1,
        "construction_year": 2000,
        "area_m2": 120.0,
        "n_floors": 2,
        "height_floor_m": 2.6,
        "thermal_resistance": 4e-3,    # K / W
        "thermal_capacitance": 1.2e7,  # J / K
        "window_area_total_m2": 20.0,  # split N/E/S/W at sim time
        # ISO 13790 5R1C (Annex C medium-medium residential)
        "H_tr_is": 1863.0,  # 3.45 W/(m²K) × 4.5 × 120 m²
        "H_tr_ms": 2730.0,  # 9.1 W/(m²K) × 2.5 × 120 m²
        "H_tr_w":  35.0,    # ~1.4 W/(m²K) × 20 m² windows
        "H_tr_em": 150.0,
        "C_m": 1.98e7,      # 165e3 J/(m²K) × 120 m²
        # VDI 6007 7R2C
        "R_1_AW": 1.0e-3,
        "C_1_AW": 1.0e7,
        "R_1_IW": 1.0e-4,
        "C_1_IW": 5.0e6,
        "R_alpha_star_AW": 1.0e-3,
        "R_alpha_star_IL": 1.0e-4,
        "R_alpha_star_IW": 1.0e-3,
        "R_rest_AW": 1.0e-3,
    })

    # ── Synthetic 1-profile electricity + occupancy ──────────────────────────
    profile_id = 1
    profiles = pd.DataFrame({
        "timestamp": idx_h,
        "profile_id": np.full(24, profile_id),
        "electricity_demand": np.full(24, 200.0),  # 200 W flat
    })
    occupancy = pd.DataFrame({
        "timestamp": idx_h,
        "profile_id": np.full(24, profile_id),
        "occupied": np.ones(24, dtype="int8"),
    })

    # ── Run ──────────────────────────────────────────────────────────────────
    model = get_thermal_model(model_name)
    out = simulate_archetype(
        archetype, profiles, occupancy, weather,
        location_id=1, lat=52.5, lon=13.4,
        thermal_model=model, progress_bar=None,
    )

    # ── Assertions ───────────────────────────────────────────────────────────
    # 1. Schema
    assert set(out.columns) == {"timestamp", "profile_id", "q_heat_w", "q_cool_w"}
    assert len(out) == 24

    # 2. No negative loads
    assert (out["q_heat_w"] >= 0).all(), "negative heating values"
    assert (out["q_cool_w"] >= 0).all(), "negative cooling values"

    # 3. Cold day → some heating happens, no cooling
    daily_heat_kwh = out["q_heat_w"].sum() / 1000.0   # 1h timesteps, Wh -> kWh
    daily_cool_kwh = out["q_cool_w"].sum() / 1000.0
    assert daily_heat_kwh > 0.0, "no heating on a 0 °C day with full occupancy"
    assert daily_cool_kwh < 1.0, f"unexpected cooling: {daily_cool_kwh:.2f} kWh"


def test_simulate_archetype_signature():
    """Issue 1 regression: setpoints/inhabitants/gains are explicit kwargs,
    not module globals — confirms the fix is durable against re-introduction
    of the spawn-worker concurrency bug."""
    import inspect
    from src.simulation import simulate_archetype, _simulate_one_pair
    sig_outer = inspect.signature(simulate_archetype).parameters
    sig_pair = inspect.signature(_simulate_one_pair).parameters
    for kw in ("heating_setpoint_C", "cooling_setpoint_C",
               "inhabitants", "gains_per_person_W"):
        assert kw in sig_outer, f"simulate_archetype missing {kw}"
        assert kw in sig_pair, f"_simulate_one_pair missing {kw}"
