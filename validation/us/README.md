# US validation against NREL EULP

This folder validates the pipeline's coupled demand profiles against the
NREL **End-Use Load Profiles** dataset (ResStock 2022.1.1, AMY 2018) on
six US climate zones. Two complementary tiers are produced here:

| Tier | What it tests | Headline script |
| --- | --- | --- |
| **per-building** | hourly Pearson r / RMSE / energy ratio per building, six US zones | `building_comparison.py` |
| **load-diversity** | per-zone load-duration curves of our pipeline vs. NREL EULP on the same buildings | `diversity_comparison.py` |

Both tiers run the pipeline on the *same* EULP buildings under matched
inputs (geometry, vintage, setpoints, occupants, infiltration, glazing,
weather) — so any discrepancy is attributable to the simulation method,
not to stock representativeness or input mismatch.

> The Winter et al. peak simultaneity factor `g(n)` is **not** computed.
> The load-diversity figure uses load-duration curves with sum-of-peaks
> normalisation per zone, which preserves a full year of dynamics. The
> rationale is documented at the top of `diversity_comparison.py`.

---

## Quick start

```bash
# 1. Pick qualified single-family buildings per county (downloads the
#    global ResStock metadata.parquet, ~176 MB).
uv run python validation/us/buildings_select.py

# 2. Parallel HTTPS download of per-building parquets + weather CSVs
#    (~12 GB raw under data/).
uv run python validation/us/download_data.py --workers 12

# 3. Convert raw NREL outputs to clean per-county thermal-demand parquets.
uv run python validation/us/convert.py

# 4. Re-simulate each building with our 1R1C pipeline (one subprocess per
#    county to avoid EnTiSe module-level state leak between counties).
uv run python validation/us/building_simulate.py --county all

# 5a. Per-building diagnostics + paper figures 6 (scatter) and 7 (overlay).
uv run python validation/us/building_comparison.py
uv run python validation/us/timeseries_comparison.py

# 5b. Validation-set overview map (paper figure 5).
uv run python validation/us/validation_map.py

# 6. Load-diversity figure (paper figure 8). Needs the EULP per-building
#    parquets plus the German pipeline outputs in ../output/.
uv run python validation/us/diversity_comparison.py
```

`runme.py` chains steps 1–5a end-to-end:

```bash
uv run python validation/us/runme.py                  # full run
uv run python validation/us/runme.py --skip-download  # reuse existing data/ cache
```

Total compute time: download ~5–10 min, simulate ~20–40 min, compare <1 min.
Disk footprint: ~12 GB raw download, ~80 MB processed, ~80 MB simulated.

---

## Data source

- **ResStock 2022.1.1 amy2018** — NREL's stock-aggregated EnergyPlus
  simulations of the US residential building stock with actual-meteorological-year
  2018 weather. Per-building 15-minute time series for ~110 end uses.
  Hosted on the OEDI S3 bucket.
- **OEDI URL patterns** (verified against live S3, see `COUNTIES.py`):
  - Metadata: `…/2022/resstock_amy2018_release_1.1/metadata/baseline.parquet`
  - Per-building parquet:
    `…/release_1.1/timeseries_individual_buildings/by_state/upgrade=0/state={state}/{bldg_id}-0.parquet`
  - Weather CSV:
    `…/release_1/weather/state={state}/{county_id}_2018.csv`
    *(quirk: weather is under `release_1`, not `release_1.1`)*

---

## Curated 6-county selection

For each US Building America climate zone we pick the county whose count
of qualified single-family-detached buildings (with both heating and
cooling HVAC equipment recorded) is closest to 100. Very Cold has no
county ≥100 qualified, so we accept the largest (G3800170 ND, 72
qualified).

| County    | State | Climate Zone | Qualified SFH | Paper label             |
|-----------|-------|--------------|---------------|-------------------------|
| G5300730  | WA    | Marine       | 97            | Pacific NW (Marine)     |
| G2601590  | MI    | Cold         | 100           | Michigan (Cold)         |
| G1800430  | IN    | Mixed-Humid  | 100           | Indiana (Mixed-Humid)   |
| G3800170  | ND    | Very Cold    | 72            | North Dakota (Very Cold)|
| G0100510  | AL    | Hot-Humid    | 100           | Alabama (Hot-Humid)     |
| G4804510  | TX    | Hot-Dry      | 104           | Texas (Hot-Dry)         |

Total: 573 qualified buildings. Hot-Dry / Hot-Humid have no German
climate analog and serve as stress-tests outside the dataset's intended
scope.

`find_zone_counties.py` is the sensitivity helper: pass a different
zone (`--zone Marine`, `--zone Cold`, …) to print the top counties by
qualified-SFH count and pick a different county.

---

## Pipeline files

| File                       | Role |
|----------------------------|------|
| `COUNTIES.py`              | Constants: 6 canonical counties, OEDI URL templates, qualified-SFH filter |
| `buildings_select.py`      | Step 1 — download `metadata.parquet`, filter to qualified buildings per county, write per-county CSVs |
| `download_data.py`         | Step 2 — parallel HTTPS download of per-building parquets + weather CSVs |
| `convert.py`               | Step 3 — clean per-county thermal-demand timeseries (sensible-only conversion of EULP electrical end-uses into q_heat_w / q_cool_w via per-building HVAC efficiency) |
| `eulp_to_sim.py`           | Parse EULP metadata (setpoints, infiltration, glazing class, orientation, occupants) into the `SimInputs` records consumed by `building_simulate.py` |
| `building_simulate.py`     | Step 4 — re-simulate each EULP building with our 1R1C pipeline using EULP-derived inputs. Spawns one subprocess per county for state isolation. |
| `building_comparison.py`   | Step 5a — per-building hourly RMSE, Pearson r, energy ratio. Produces `data/building_comparison_metrics.csv` and **paper Figure 6** (`img/us_building_comparison.{png,pdf}`). |
| `timeseries_comparison.py` | Step 5b — 6-panel best-fit-per-zone overlay. Produces **paper Figure 7** (`img/us_timeseries_comparison.{png,pdf}`). |
| `validation_map.py`        | Validation-set overview map. Produces **paper Figure 5** (`img/us_validation_map.{png,pdf}`). |
| `diversity_comparison.py`  | Per-zone load-duration curves vs. NREL EULP. Produces **paper Figure 8** (`img/us_diversity_comparison.{png,pdf}`). |
| `find_zone_counties.py`    | Sensitivity helper — pick alternative counties in a given climate zone. |
| `find_capacity_columns.py` | One-off diagnostic — inspect EULP HVAC capacity columns. |
| `runme.py`                 | Chains steps 1–5a end-to-end. |

> Tier 1B (the German-archetype substitution sensitivity test) was
> investigated and **removed**. The country-replacement guide in the
> paper's *Usage Notes* covers archetype substitution.

---

## Output layout

```
validation/us/
├── data/                                # gitignored
│   ├── metadata.parquet                     # ResStock global metadata (~176 MB)
│   ├── selection_summary.csv                # 6 rows, one per county
│   ├── {COUNTY}_{ZONE}_qualified.csv        # 6 files, per-county building list with rich metadata
│   ├── raw/{COUNTY}_{ZONE}/{bldg_id}-0.parquet     # raw EULP per-building (input to step 3)
│   ├── weather/{county_id}_2018.csv         # per-county hourly weather
│   ├── processed/{COUNTY}_{ZONE}.parquet    # cleaned per-county thermal demand (output of convert.py)
│   ├── sim/{COUNTY}_{ZONE}.parquet          # our 1R1C simulation output (output of building_simulate.py)
│   ├── building_comparison_metrics.csv      # per-building r / RMSE / energy ratio / outlier flag
│   └── building_comparison_outliers.csv     # buildings with sim/real ratio > 3
└── (figures land in repo-root img/, not here)
```

### Processed parquet schema (NREL "truth" side)

One row per (timestamp, bldg_id). All powers in W.

| Column         | dtype   | Unit | Description                                                  |
|----------------|---------|------|--------------------------------------------------------------|
| timestamp      | dt[ns]  | —    | 15-min interval (UTC-naive, 35 040 rows per building per yr) |
| bldg_id        | int32   | —    | EULP building identifier                                     |
| electricity_w  | float32 | W    | 17-column household power sum (lighting, plug loads, large appliances, ventilation, well/pool/hot-tub pumps). Excludes HVAC electric, DHW, EV charging. |
| q_heat_w       | float32 | W    | Thermal heating: fossil_fuel × AFUE + elec_resistance, or elec_HP × COP(T_out) per Ruhnau et al. 2019. HP backup resistance added 1:1. |
| q_cool_w       | float32 | W    | Thermal cooling: cooling_electric × EER(T_out). Rated EER parsed from `in.hvac_cooling_efficiency`. T-correction: EER falls ~0.85 %/K above 35 °C, rises symmetrically below, clipped at 50 % of rated. |

### Simulated parquet schema (our "sim" side)

| Column         | dtype   | Unit | Description                |
|----------------|---------|------|----------------------------|
| timestamp      | dt[ns]  | —    | matches NREL timestamps    |
| bldg_id        | int32   | —    | matches NREL bldg_id       |
| q_heat_w_sim   | float32 | W    | EnTiSe R1C1 heating output |
| q_cool_w_sim   | float32 | W    | EnTiSe R1C1 cooling output |

---

## Methodology choices

### Building parameters (R, C)
TEASER `tabula_de_standard` typology, fitted per (vintage_year,
area_bucket=10 m², n_floors). US construction is generally less leaky
than German pre-1945; we clip the fed `year_of_construction` to **≥1995**
to avoid TABULA's pre-war mass-brick archetypes producing wildly low R
for US wood-frame stock. Buildings where TEASER fitting fails are
**skipped** (no fallback values).

### Setpoints
Per-building from EULP `in.heating_setpoint` / `in.cooling_setpoint`
(constant, not setback). Buildings with `cool − heat < 2 K` have their
setpoints reshaped symmetrically around the midpoint to enforce a 2 K
minimum deadband (~16 % of EULP buildings have inverted or zero-deadband
configurations that EnergyPlus handles internally with mode-priority
logic; our 1R1C controller would oscillate without this fix).

### Ventilation
ACH derived from EULP `in.infiltration` class (e.g. "15 ACH50") via
Sherman–Grimsrud: `natural_ACH = ACH50 / 17.5`. A mild diurnal modulation
(±15 %) is applied; EnTiSe converts to W/K internally given the building
volume.

### Internal gains
`occupancy_gain[t] = occupancy[t] × in.occupants × 80 W/person`
+ `electricity_w[t]` (the 17-column NREL household electricity).
Occupancy is detected per-building from electricity using GeoMA (α=0.05).

### Glazing
Per-building SHGC from EULP `in.windows` class:

| Class                                    | SHGC |
|------------------------------------------|------|
| Single, Clear                            | 0.86 |
| Double, Clear                            | 0.76 |
| Double, Low-E (M-Gain default)           | 0.40 |
| Double, Low-E, L-Gain                    | 0.30 |
| Double, Low-E, H-Gain                    | 0.60 |
| Triple, Clear                            | 0.65 |
| Triple, Low-E (M-Gain default)           | 0.35 |
| Triple, Low-E, L-Gain                    | 0.25 |
| Triple, Low-E, H-Gain                    | 0.50 |

Window areas from `in.window_areas` (e.g. "F18 B18 L18 R18", ft²)
mapped to compass orientation via `in.orientation` (which face the
front F points to). Tilt 90°, shading 0.75. Sanity fallback to 15 %
glazing if parsed total > 50 % of floor area.

### Heat-pump COP and EER (NREL truth-side computation)
- Air-source heat pump: `COP(ΔT) = 0.0005·ΔT² − 0.09·ΔT + 6.80`, with
  `ΔT = T_supply − T_outdoor` and `T_supply = 40 − T_outdoor` (radiator)
  or `T_supply = 30 − 0.5·T_outdoor` (floor). Source: Ruhnau et al.,
  *Time series of heat demand and heat pump efficiency for energy system
  modeling*, Sci. Data 6:189 (2019).
- AC EER parsed per building from `in.hvac_cooling_efficiency`
  ("AC, SEER 13" → 13 × 0.875 = 11.4 BTU/Wh = 3.34 W/W; "Room AC,
  EER 10.7" → 10.7 / 3.412 = 3.14 W/W; default 3.0 W/W). Temperature
  correction: `EER(T) = EER_rated × max(0.5, 1 − 0.0085 × (T − 35 °C))`.

### Cooling — known structural limitation
The 1R1C method is **sensible-only**: it does not model latent cooling
(humidity removal). EULP cooling-electric includes both sensible and
latent, the latter dominating in humid climates. Our sim under-predicts
cooling intensity by ~50 % in Hot-Humid AL and Hot-Dry TX. This is a
structural limit of lumped-capacitance methods, not a bug.

---

## Filters applied during validation

The full sample is 573 qualified SFH; reports use these transparent filters:

1. **TEASER fit failed** (very rare, <1 %) — skipped at simulation time.
2. **Inverted setpoint** (heat ≥ cool, ~16 % of buildings) — handled by
   the 2 K minimum-deadband reshape; buildings stay in the sample and
   count toward `n_used`.
3. **Low real signal** (real annual heating or cooling < 1 MWh, varies
   per zone) — excluded from the corresponding per-variable metric only;
   shown as light-grey points in the figure with an `n_dropped_lowsig`
   count printed in the per-zone diagnostic table.

---

## Headline validation findings

> The metrics below are from an earlier run. They will be refreshed after
> the regeneration with the new internal-gain (80 W/person), comfort-bound
> alignment (EN 16798-1 Cat II), window-area schema, and consolidated
> `src/ach.py` rule. Use `data/building_comparison_metrics.csv` as the
> authoritative source after the rerun.

After all filters, on the six climate zones:

| Zone        | n_used (heat / cool) | Heat r | Heat sim/real | Cool r | Cool sim/real |
|-------------|----------------------|--------|---------------|--------|---------------|
| Cold        |   91 / 93            | 0.91   | 1.03          | 0.71   | 0.50          |
| Hot-Dry     |   94 / 104           | 0.77   | 1.10          | 0.78   | 0.56          |
| Hot-Humid   |   92 / 100           | 0.81   | 1.23          | 0.76   | 0.50          |
| Marine      |   91 / 58            | 0.70   | 1.13          | 0.33   | 0.18          |
| Mixed-Humid |   93 / 100           | 0.93   | 1.20          | 0.86   | 0.61          |
| Very Cold   |   72 / 62            | 0.94   | 1.15          | 0.71   | 1.04          |

- **Heating** captured well: r 0.69–0.94 across zones; intensity within
  ~30 % of EULP everywhere.
- **Cooling** temporally captured (r 0.71–0.86 outside Marine) but
  systematically under-predicted in magnitude by ~50 %, traced to the
  absence of latent cooling in 1R1C.
- **Marine cooling** has very low real signal (median 16 kWh/m²·yr); 39
  of 97 buildings are excluded as low-signal, leaving 58. Remaining
  correlation is weak but the absolute errors are tiny.

---

## Reproducibility notes

- All inputs are derived deterministically from the OEDI-hosted ResStock
  release. Random seeds are fixed where applicable.
- The simulation step uses a **fresh Python subprocess per county**
  (`building_simulate.py --county all`) to avoid module-level state
  leak from EnTiSe / TEASER between back-to-back simulations — a real
  bug observed during development that produced inconsistent per-county
  results in single-process mode.
- All intermediate and final outputs are zstd-compressed Parquet files;
  building-level CSVs are reviewable in any spreadsheet application.

## Known limitations

1. **Sensible-only cooling.** 1R1C cannot model latent cooling; cooling
   intensity in humid climates is systematically under-predicted by ~50 %.
2. **German envelope archetypes for US buildings.** TABULA-DE fits give
   reasonable order-of-magnitude R, C but introduce ~10–25 % systematic
   over-prediction of heating intensity.
3. **Constant setpoints.** No setback / schedule. EULP buildings with
   active setbacks are simulated with their *base* setpoint, missing
   some intra-day temporal structure.
4. **Single-zone.** The entire dwelling is one thermal mass. Multi-zone
   thermal coupling (between rooms, between floors) is not represented.
5. **Hot-Dry / Hot-Humid have no German analog.** These zones are
   stress-tests for the method rather than representative of the
   dataset's intended scope.

These are honest scope statements rather than failures: the 1R1C method
is the project's deliberate scalability / transparency choice for a
3.3 M-building-year dataset. Users requiring higher-fidelity per-building
estimates should layer additional models on top of this dataset
(`thermal_model: r5c1` or `r7c2` in the YAML is the in-repo path).

## Provenance

The predecessor master's thesis (M. F. Escobar Viegas, *Deriving coupled
electricity, heating and cooling time series from electricity demands*,
TUM, 2025) ran a 26-building Marine validation against G4100070 OR — all
SFH-detached in that single county. Cooling-type breakdown for those 26:
17 None, 5 Room AC, 2 Heat Pump, 2 Central AC — so the cooling
validation in the thesis was effectively built on 9 buildings. This
pipeline scales the qualified-SFH count to ~100 per climate zone by
selecting larger counties (G5300730 instead of G4100070 for Marine,
etc.) while keeping the SFH + cooling + heating filter explicit.
