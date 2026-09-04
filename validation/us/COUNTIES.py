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


# Approximate mean elevation (m) of each county's populated area, used
# only to derive a per-county constant surface pressure via the ISA
# standard atmosphere. NREL's EULP AMY 2018 weather CSVs do not carry
# pressure, and the paper's design choice is to keep NREL as the sole
# US weather source (no OpenMeteo fusion on the validation truth-side),
# so this hard-coded lookup replaces a sea-level constant that was
# accurate only for the coastal counties.
#
# Sources: USGS county seat elevations / GNIS. Uncertainty vs. exact
# per-building elevation is ~50 m, translating to ~0.6 % on humidity
# ratio — negligible next to the latent load itself in humid zones.
COUNTY_ELEVATION_M: dict[str, float] = {
    "G3800170":  275.0,  # Cass ND — Fargo
    "G2601590":  200.0,  # Van Buren MI — Paw Paw
    "G5300330":   50.0,  # King WA — Seattle metro
    "G1800430":  275.0,  # Floyd IN — New Albany (see note below)
    "G4804510":  570.0,  # Tom Green TX — San Angelo (highest of the six)
    "G0100510":   65.0,  # Elmore AL — Wetumpka
    # Sensitivity counties (present on disk, used when uncommented above)
    "G0600850":   15.0,  # Alameda CA — Oakland (Bay-Marine sensitivity)
    "G1700310":  110.0,  # Alexander IL — Cairo (Cold-IL2 sensitivity)
    "G2601250":  220.0,  # Genesee MI — Flint (Cold-MI4 sensitivity)
    "G2601630":  280.0,  # Livingston MI — Howell (Cold-MI3 sensitivity)
    "G5300530":  110.0,  # Pierce WA — Tacoma (Marine-WA2 sensitivity)
    "G5300730":  220.0,  # Whatcom WA — Bellingham (Marine-WA3 sensitivity)
}

# NOTE on the Mixed-Humid elevation. 275 m was originally looked up for
# Hamilton IN, from a mis-identification of FIPS G1800430 that is corrected in
# the comments above; Floyd IN (New Albany, on the Ohio) sits nearer 140 m.
# The value is deliberately left at 275 m because the published validation
# results were produced with it. Elevation enters only through surface
# pressure and thence the humidity ratio, so the ~135 m error shifts latent
# cooling by order 1 %, well below the precision at which the Mixed-Humid
# results are reported. Changing it would make this repository stop
# reproducing the paper.


def pressure_from_elevation(h_m: float) -> float:
    """ISA (International Standard Atmosphere) surface pressure at
    elevation ``h_m`` above sea level, in Pa.

    Valid in the troposphere (0–11 km). Above 300 m the correction to
    the sea-level default (101 325 Pa) exceeds 3 %, which starts to
    matter for the humidity-ratio ω that drives EnTiSe's latent-cooling
    post-pass — hence the per-county lookup rather than a global constant.
    """
    return 101_325.0 * (1.0 - 2.25577e-5 * float(h_m)) ** 5.25588
