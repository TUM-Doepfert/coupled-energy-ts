# Validation

This folder produces the validation evidence reported in the paper. The
paper section *Technical Validation* is the authoritative reference;
this README is a map of the scripts and their outputs so a reviewer can
rerun any tier.

## What's in the paper

The paper reports three validation tiers, in two subfolders:

| Tier | What it tests | Folder | Canonical script |
| --- | --- | --- | --- |
| **per-building** | Per-building reproduction against NREL EULP across six US climate zones (Pearson r, RMSE, energy ratio per building) | `us/` | `building_simulate.py` → `building_comparison.py` (+ `timeseries_comparison.py`, `validation_map.py`) |
| **load-diversity** | Per-zone load-duration curves (sum-of-peaks normalisation) against NREL EULP on the same buildings | `us/` | `diversity_comparison.py` |
| **When2Heat** | Aggregate cross-validation against When2Heat (Ruhnau et al., 2019) on the published Germany 2010 dataset | `germany/` | `demand_comparison.py` |

> **Winter et al. simultaneity factor** is not used. The load-diversity
> figure compares load-duration curves against NREL EULP, not against
> the Winter `g(n)` curve. Rationale: `g(n)` collapses a year of
> dynamics to one number dominated by the coldest hour, where any
> 74-household stock saturates near 1. See the docstring at the top
> of `us/diversity_comparison.py`.
>
> **Tier 1B** (the German-archetype substitution test) was investigated
> and is no longer in the repo. The country-replacement guide in the
> paper's *Usage Notes* covers archetype substitution.

## Order to run

```bash
# US tier (per-building + load-diversity). First run does the ~12 GB
# EULP download + 5-step chain end-to-end; subsequent runs reuse the
# data/ cache. Renders paper Figures 5 (map), 6 (scatter), 7 (overlay).
uv run python validation/us/runme.py
#   produces: img/us_validation_map.{png,pdf}
#             img/us_building_comparison.{png,pdf}
#             img/us_timeseries_comparison.{png,pdf}

# Load-diversity figure (needs the EULP cache from the step above AND
# the German pipeline outputs in output/).
uv run python validation/us/diversity_comparison.py
#   produces: img/us_diversity_comparison.{png,pdf}

# Germany tier — When2Heat aggregate cross-validation (needs the German
# pipeline outputs and the When2Heat CSV). Download the CSV once from
# https://data.open-power-system-data.org/when2heat/ and save to
# validation/germany/data/when2heat.csv.
uv run python validation/germany/demand_comparison.py
#   produces: img/de_demand_comparison.{png,pdf}
```

The German pipeline outputs (`output/B.parquet`, `E.parquet`,
`O.parquet`, `HC/locXXXX/hc_archYY.parquet`) come from the top-level
CLI: `uv run coupled-energy-ts run config/germany_2010.yml`.

## Layout

```
validation/
├── README.md              # this file
├── runme.py               # convenience wrapper: us/ then germany/
├── us/                    # NREL EULP comparison (per-building + load-diversity)
│   ├── README.md              full methodology
│   ├── runme.py               in-folder orchestrator (5-step chain)
│   ├── COUNTIES.py            6 canonical counties + URL templates + qualified-SFH filter
│   ├── buildings_select.py    pick qualified SFH from EULP metadata
│   ├── download_data.py       parallel HTTPS download of EULP parquets + weather
│   ├── convert.py             EULP raw → cleaned per-county thermal-demand parquets
│   ├── eulp_to_sim.py         EULP metadata → SimInputs records
│   ├── building_simulate.py   re-simulate each EULP building (one subprocess per county)
│   ├── building_comparison.py per-building r / RMSE / energy ratio + paper Figure 6
│   ├── timeseries_comparison.py 6-panel best-fit-per-zone overlay (paper Figure 7)
│   ├── validation_map.py      validation-set overview map (paper Figure 5)
│   ├── diversity_comparison.py per-zone LDC paper figure (paper Figure 8)
│   ├── find_zone_counties.py  sensitivity helper (e.g. `--zone Marine`)
│   ├── find_capacity_columns.py diagnostic
│   └── data/                  cached EULP downloads + per-step parquets (gitignored)
│
└── germany/               # When2Heat aggregate cross-validation
    ├── demand_comparison.py   When2Heat aggregate cross-validation (paper Figure 9)
    └── data/                  cached When2Heat CSV + per-run aggregates (gitignored)
```

## Outputs

| Tier | Numerical output | Figure (paper) |
| --- | --- | --- |
| per-building | `validation/us/data/building_comparison_metrics.csv`, `…_outliers.csv` | `img/us_building_comparison.{png,pdf}`, `img/us_timeseries_comparison.{png,pdf}`, `img/us_validation_map.{png,pdf}` |
| load-diversity | per-zone metrics printed by the figure script | `img/us_diversity_comparison.{png,pdf}` (sum-of-peaks normalisation) |
| When2Heat | `validation/germany/data/when2heat_metrics.csv` | `img/de_demand_comparison.{png,pdf}` |

## Prerequisites

- **US tier** needs the NREL EULP cache. First run does the full
  download + convert prep, writing ~12 GB raw + ~80 MB processed under
  `us/data/`.
- **Load-diversity** additionally needs the German pipeline outputs
  under `output/`.
- **When2Heat** needs the German pipeline outputs plus the single
  ~10 MB When2Heat CSV from Open Power System Data.
