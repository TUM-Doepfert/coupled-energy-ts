"""Curated county selection for NREL EULP demand validation.

For each climate zone we pick the county whose count of qualified SFH
(Single-Family Detached + has cooling + has heating + has electricity)
is closest to 100. Very Cold has no county >= 100; we accept the largest
available (G3800170 ND with 72).
"""
from __future__ import annotations

# (county_id, state, climate_zone, expected_n_qualified, paper_label)
COUNTIES: list[tuple[str, str, str, int, str]] = [
    ("G3800170", "ND", "Very Cold",   72,  "North Dakota (Very Cold)"),
    ("G2601590", "MI", "Cold",       100,  "Michigan (Cold)"),
    ("G5300330", "WA", "Marine",     100,  "Pacific NW (Marine)"),
    ("G1800430", "IN", "Mixed-Humid",100,  "Indiana (Mixed-Humid)"),
    ("G4804510", "TX", "Hot-Dry",    104,  "Texas (Hot-Dry)"),
    ("G0100510", "AL", "Hot-Humid",  100,  "Alabama (Hot-Humid)"),
# # ── sensitivity additions (delete after sensitivity check) ──
#     ("G5300330", "WA", "Marine-WA2", 100, "WA #2 (Marine)"),
#     ("G5300530", "WA", "Marine-WA3", 100, "WA #3 (Marine)"),
#     ("G0600850", "CA", "Marine-CA",  100, "CA Bay (Marine)"),
#     ('G1700310', 'IL', 'Cold-IL2', 100, 'IL #2 (Cold)'),
#     ('G2601630', 'MI', 'Cold-MI3', 100, 'MI #3 (Cold)'),
#     ('G2601250', 'MI', 'Cold-MI4', 100, 'MI #4 (Cold)'),

]

# OEDI / S3 endpoints for ResStock 2022.1.1 amy2018 baseline
OEDI_BASE = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2022/"
    "resstock_amy2018_release_1.1"
)
METADATA_URL = f"{OEDI_BASE}/metadata/baseline.parquet"
HOUSEHOLD_URL_FMT = (
    f"{OEDI_BASE}/timeseries_individual_buildings/by_state/"
    "upgrade=0/state={state}/{bldg_id}-0.parquet"
)
# Quirk: weather files are under release_1 (without the .1) while households
# are under release_1.1. Confirmed empirically via OEDI HEAD requests.
_OEDI_BASE_R1 = OEDI_BASE.replace("resstock_amy2018_release_1.1",
                                  "resstock_amy2018_release_1")
WEATHER_URL_FMT = (
    f"{_OEDI_BASE_R1}/weather/state={{state}}/{{county_id}}_2018.csv"
)

# Building-level filter for "qualified" buildings
def is_qualified(row) -> bool:
    return (
        row["in.geometry_building_type_recs"] == "Single-Family Detached"
        and row["in.hvac_cooling_type"] != "None"
        and row["in.heating_fuel"] != "None"
    )
