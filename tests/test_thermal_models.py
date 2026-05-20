"""Tests for the thermal-model registry.

Verifies that ``r1c1`` / ``r5c1`` / ``r7c2`` resolve to a ThermalModel
descriptor and that ``obj_keys_factory`` produces the EnTiSe-style
bracketed-unit keys each model expects.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.thermal_models import (
    REGISTRY,
    ThermalModel,
    get_thermal_model,
    validate_archetype_columns,
)


def test_registry_has_three_models():
    assert set(REGISTRY) == {"r1c1", "r5c1", "r7c2"}


@pytest.mark.parametrize("name", ["r1c1", "r5c1", "r7c2"])
def test_get_thermal_model_returns_descriptor(name: str):
    m = get_thermal_model(name)
    assert isinstance(m, ThermalModel)
    assert m.name == name
    assert callable(m.obj_keys_factory)
    assert m.entise_class_name in {"R1C1", "R5C1", "R7C2"}


def test_get_thermal_model_is_case_insensitive():
    assert get_thermal_model("R1C1") is REGISTRY["r1c1"]


def test_get_thermal_model_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown thermal_model"):
        get_thermal_model("r99c99")


def test_r1c1_obj_keys():
    m = get_thermal_model("r1c1")
    row = pd.Series({
        "archetype_id": 1,
        "thermal_resistance": 0.005,
        "thermal_capacitance": 5.0e7,
    })
    keys = m.obj_keys_factory(row)
    assert keys == {
        "resistance[K W-1]": 0.005,
        "capacitance[J K-1]": 5.0e7,
    }


def test_r5c1_obj_keys():
    m = get_thermal_model("r5c1")
    row = pd.Series({
        "H_tr_is": 1.0, "H_tr_ms": 2.0, "H_tr_w": 3.0, "H_tr_em": 4.0,
        "C_m": 5.0e7,
    })
    keys = m.obj_keys_factory(row)
    assert set(keys) == {
        "H_tr_is[W K-1]", "H_tr_ms[W K-1]", "H_tr_w[W K-1]",
        "H_tr_em[W K-1]", "C_m[J K-1]",
    }
    assert keys["C_m[J K-1]"] == 5.0e7


def test_r7c2_obj_keys():
    m = get_thermal_model("r7c2")
    row = pd.Series({
        "R_1_AW": 1.0, "C_1_AW": 1.0, "R_1_IW": 1.0, "C_1_IW": 1.0,
        "R_alpha_star_AW": 1.0, "R_alpha_star_IL": 1.0,
        "R_alpha_star_IW": 1.0, "R_rest_AW": 1.0,
    })
    keys = m.obj_keys_factory(row)
    assert len(keys) == 8
    assert all("[" in k and "]" in k for k in keys), "EnTiSe keys must carry units"


def test_validate_archetype_columns_passes_for_r1c1():
    m = get_thermal_model("r1c1")
    df = pd.DataFrame([{
        "thermal_resistance": 0.005, "thermal_capacitance": 5e7,
        "archetype_id": 1, "area_m2": 120.0,
    }])
    validate_archetype_columns(m, df)  # no raise


def test_validate_archetype_columns_raises_for_r5c1_on_r1c1_df():
    m = get_thermal_model("r5c1")
    df = pd.DataFrame([{"thermal_resistance": 0.005, "thermal_capacitance": 5e7}])
    with pytest.raises(ValueError, match="missing columns"):
        validate_archetype_columns(m, df)
