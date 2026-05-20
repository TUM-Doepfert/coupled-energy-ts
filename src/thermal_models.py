"""Thermal model registry.

Maps the user-facing model name (``r1c1`` / ``r5c1`` / ``r7c2``) to:

  - The EnTiSe HVAC class that runs the simulation.
  - The set of schema columns the archetype table must provide.
  - The translation from canonical schema names (e.g. ``thermal_resistance``)
    to EnTiSe's object-key convention with embedded units
    (e.g. ``"resistance[K W-1]"``).

The pipeline reads ``thermal_model`` from the YAML config and looks the
descriptor up here. New models can be added by appending to ``REGISTRY``.

Design rationale
----------------
EnTiSe's R1C1, R5C1, R7C2 each declare a ``required_keys`` class
attribute that lists the obj-dict keys they expect. We keep our schema
column names short (``H_tr_is`` not ``H_tr_is[W K-1]``) because the
units are documented in ``ArchetypeSchema`` once and adding bracketed
suffixes to parquet columns is awkward. The translation lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .providers import ArchetypeSchema


@dataclass(frozen=True)
class ThermalModel:
    """Descriptor for one thermal model variant."""

    name: str
    """User-facing name: 'r1c1', 'r5c1', 'r7c2'."""

    required_columns: tuple[str, ...]
    """Schema columns that must be present in the archetype table."""

    obj_keys_factory: Callable[[pd.Series], dict[str, float]]
    """Build the model-specific subset of the EnTiSe object dict from
    one archetype row. Returns keys with EnTiSe's unit-bracketed naming."""

    entise_class_name: str
    """Class name as exported by ``entise.methods.hvac`` —
    e.g. ``"R1C1"``, ``"R5C1"``, ``"R7C2"``."""

    def load_class(self):
        """Import and return the EnTiSe class. Lazy so EnTiSe stays optional.

        EnTiSe's ``entise.methods.hvac/__init__.py`` re-exports R1C1 and
        R5C1 as classes but not R7C2 (as of entise 0.x). For the missing
        case we fall back to the per-model submodule (``entise.methods.
        hvac.R7C2.R7C2``). When EnTiSe re-exports R7C2 too this fallback
        becomes a no-op.
        """
        import entise.methods.hvac as _hvac
        attr = getattr(_hvac, self.entise_class_name)
        if isinstance(attr, type):
            return attr
        # `attr` is the per-model submodule; the class is named the same.
        return getattr(attr, self.entise_class_name)


# ── Per-model object-key factories ───────────────────────────────────────

def _r1c1_obj_keys(row: pd.Series) -> dict[str, float]:
    return {
        "resistance[K W-1]": float(row["thermal_resistance"]),
        "capacitance[J K-1]": float(row["thermal_capacitance"]),
    }


def _r5c1_obj_keys(row: pd.Series) -> dict[str, float]:
    return {
        "H_tr_is[W K-1]": float(row["H_tr_is"]),
        "H_tr_ms[W K-1]": float(row["H_tr_ms"]),
        "H_tr_w[W K-1]":  float(row["H_tr_w"]),
        "H_tr_em[W K-1]": float(row["H_tr_em"]),
        "C_m[J K-1]":     float(row["C_m"]),
    }


def _r7c2_obj_keys(row: pd.Series) -> dict[str, float]:
    return {
        "R_1_AW[K W-1]":         float(row["R_1_AW"]),
        "C_1_AW[J K-1]":         float(row["C_1_AW"]),
        "R_1_IW[K W-1]":         float(row["R_1_IW"]),
        "C_1_IW[J K-1]":         float(row["C_1_IW"]),
        "R_alpha_star_AW[K W-1]": float(row["R_alpha_star_AW"]),
        "R_alpha_star_IL[K W-1]": float(row["R_alpha_star_IL"]),
        "R_alpha_star_IW[K W-1]": float(row["R_alpha_star_IW"]),
        "R_rest_AW[K W-1]":       float(row["R_rest_AW"]),
    }


# ── Registry ─────────────────────────────────────────────────────────────

REGISTRY: dict[str, ThermalModel] = {
    "r1c1": ThermalModel(
        name="r1c1",
        required_columns=("thermal_resistance", "thermal_capacitance"),
        obj_keys_factory=_r1c1_obj_keys,
        entise_class_name="R1C1",
    ),
    "r5c1": ThermalModel(
        name="r5c1",
        required_columns=ArchetypeSchema.OPTIONAL_5R1C,  # ISO 13790
        obj_keys_factory=_r5c1_obj_keys,
        entise_class_name="R5C1",
    ),
    "r7c2": ThermalModel(
        name="r7c2",
        required_columns=ArchetypeSchema.OPTIONAL_7R2C,  # VDI 6007
        obj_keys_factory=_r7c2_obj_keys,
        entise_class_name="R7C2",
    ),
}


def get_thermal_model(name: str) -> ThermalModel:
    """Look up a thermal model by name. Raises ValueError on unknown name."""
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown thermal_model={name!r}. "
            f"Supported: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[key]


def validate_archetype_columns(model: ThermalModel, df: pd.DataFrame) -> None:
    """Raise ValueError if the archetype DataFrame lacks columns the model needs."""
    missing = [c for c in model.required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Archetype table is missing columns required by "
            f"thermal_model={model.name!r}: {missing}.\n"
            f"Have: {list(df.columns)}\n"
            f"Required: {list(model.required_columns)}\n"
            f"Provide these via the archetype provider (extend the parquet "
            f"or supply a CSV/parquet that already includes them). For "
            f"R5C1 the source is typically ISO 13790 calc; for R7C2, "
            f"VDI 6007."
        )


__all__ = [
    "ThermalModel",
    "REGISTRY",
    "get_thermal_model",
    "validate_archetype_columns",
]
