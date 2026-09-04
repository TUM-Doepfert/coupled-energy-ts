"""Audit the 1R1C envelope conductance against TEASER's own element-wise UA.

Task A of the magnitude follow-up. ``src/providers/archetype.py::_build_one``
sets the published ``thermal_resistance`` from TEASER's one-element model::

    r_1r1c = float(m1.r_total_ow)

The Methods section describes this as "the aggregated outer-wall resistance
of the TEASER one-element model", which reads as if roof, floor and windows
might be missing. This script rebuilds each archetype through the SAME
TEASER call path and dumps every envelope element's area, U-value and UA, so
the question is settled by measurement rather than by reading the name.

Cross-check: ``B.parquet`` also carries the ISO 13790 5R1C columns exported
from the same TEASER model, so the 5R1C series combination

    H_tr_op = 1 / (1/H_tr_em + 1/H_tr_ms)      # opaque, in series
    UA_5r1c = H_tr_op + H_tr_w                 # + windows in parallel

must reproduce the 1R1C UA if the two exports are consistent.

Usage
-----
    uv run python validation/germany/teaser_envelope_audit.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.providers.archetype import _load_archetypes_csv  # noqa: E402


def audit_one(row, construction_data: str, geometry_data: str) -> tuple[dict, list[dict]]:
    """Rebuild one archetype exactly as ``_build_one`` does and dump elements."""
    from teaser.project import Project

    prj = Project()
    prj.add_residential(
        construction_data=construction_data,
        geometry_data=geometry_data,
        name=f"arch{int(row.archetype_id):02d}",
        year_of_construction=int(row.year_of_construction),
        number_of_floors=int(row.number_of_floors),
        height_of_floors=float(row.height_of_floors),
        net_leased_area=float(row.net_leased_area),
        inner_wall_approximation_approach="teaser_default",
    )
    prj.calc_all_buildings()
    bldg = prj.buildings[0]

    # Identical to the production call: one element, windows MERGED, IBPSA.
    bldg.calc_building_parameter(
        number_of_elements=1, merge_windows=True, used_library="IBPSA"
    )
    zone = bldg.thermal_zones[0]
    m1 = zone.model_attr

    aid = int(row.archetype_id)
    elements: list[dict] = []
    # The four opaque groups that one_element.py:481-486 aggregates into
    # "outer_walls", plus windows which merge_windows=True folds in on top.
    groups = {
        "OuterWall": zone.outer_walls,
        "Rooftop": zone.rooftops,
        "GroundFloor": zone.ground_floors,
        "InterzonalOuter": zone.find_izes_outer(),
        "Window": zone.windows,
    }
    for gname, elems in groups.items():
        for e in elems:
            elements.append({
                "archetype_id": aid,
                "group": gname,
                "name": getattr(e, "name", ""),
                "orientation": getattr(e, "orientation", None),
                "area_m2": float(e.area),
                "u_value_w_m2k": float(getattr(e, "u_value", float("nan"))),
                "ua_value_w_k": float(e.ua_value),
            })

    opaque = [e for e in elements if e["group"] != "Window"]
    windows = [e for e in elements if e["group"] == "Window"]
    ua_opaque_sum = sum(e["ua_value_w_k"] for e in opaque)
    ua_win_sum = sum(e["ua_value_w_k"] for e in windows)

    summary = {
        "archetype_id": aid,
        "year": int(row.year_of_construction),
        "net_leased_area_m2": float(row.net_leased_area),
        # element-wise, summed by this script
        "ua_opaque_elementwise": ua_opaque_sum,
        "ua_window_elementwise": ua_win_sum,
        "ua_total_elementwise": ua_opaque_sum + ua_win_sum,
        "area_opaque_m2": sum(e["area_m2"] for e in opaque),
        "area_window_m2": sum(e["area_m2"] for e in windows),
        # TEASER's own aggregates
        "teaser_ua_value_ow": float(m1.ua_value_ow),
        "teaser_ua_value_win": float(m1.ua_value_win),
        "teaser_area_ow": float(m1.area_ow),
        "teaser_r_total_ow": float(m1.r_total_ow),
        "ua_from_r_total_ow": 1.0 / float(m1.r_total_ow),
    }
    return summary, elements


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archetypes-csv", type=Path, default=Path("input/archetypes.csv"))
    p.add_argument("--b-parquet", type=Path, default=Path("output/B.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("validation/germany/data"))
    p.add_argument("--construction-data", default="tabula_de_standard")
    p.add_argument("--geometry-data", default="tabula_de_single_family_house")
    args = p.parse_args()

    arch = _load_archetypes_csv(args.archetypes_csv)
    summaries, all_elements = [], []
    for row in arch.itertuples():
        s, e = audit_one(row, args.construction_data, args.geometry_data)
        summaries.append(s)
        all_elements.extend(e)
        print(f"  arch{s['archetype_id']:02d} ({s['year']}) "
              f"UA_opaque {s['ua_opaque_elementwise']:8.1f}  "
              f"UA_win {s['ua_window_elementwise']:7.1f}  "
              f"UA_total {s['ua_total_elementwise']:8.1f}  "
              f"1/r_total_ow {s['ua_from_r_total_ow']:8.1f}")

    S = pd.DataFrame(summaries).set_index("archetype_id")
    E = pd.DataFrame(all_elements)

    b = pd.read_parquet(args.b_parquet).set_index("archetype_id")
    S["published_R"] = b["thermal_resistance"]
    S["published_UA"] = 1.0 / b["thermal_resistance"]
    S["area_m2_published"] = b["area_m2"]

    # --- the headline ratio -------------------------------------------------
    S["ratio_published_over_elementwise"] = S["published_UA"] / S["ua_total_elementwise"]
    S["ratio_published_over_opaque_only"] = S["published_UA"] / S["ua_opaque_elementwise"]

    # --- 5R1C cross-check ---------------------------------------------------
    H_op = 1.0 / (1.0 / b["H_tr_em"] + 1.0 / b["H_tr_ms"])
    S["ua_5r1c_series"] = H_op + b["H_tr_w"]
    S["ratio_1r1c_over_5r1c"] = S["published_UA"] / S["ua_5r1c_series"]

    S["published_UA_per_m2"] = S["published_UA"] / S["area_m2_published"]
    S["elementwise_UA_per_m2"] = S["ua_total_elementwise"] / S["area_m2_published"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    S.to_csv(args.out_dir / "teaser_envelope_audit.csv")
    E.to_csv(args.out_dir / "teaser_envelope_elements.csv", index=False)

    cols = ["ua_opaque_elementwise", "ua_window_elementwise", "ua_total_elementwise",
            "published_UA", "ratio_published_over_elementwise",
            "ratio_published_over_opaque_only", "ua_5r1c_series", "ratio_1r1c_over_5r1c"]
    print("\n=== envelope audit ===")
    print(S[cols].round(4).to_string())
    print("\n=== per-element areas / U-values (grouped) ===")
    g = (E.groupby(["archetype_id", "group"])
           .agg(n=("area_m2", "size"), area_m2=("area_m2", "sum"),
                ua_w_k=("ua_value_w_k", "sum"),
                u_mean=("u_value_w_m2k", "mean")).round(3))
    print(g.to_string())
    print(f"\nwrote teaser_envelope_audit.csv / teaser_envelope_elements.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()
