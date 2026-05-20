# Input directory

Source data the pipeline reads. Different inputs have different
provenance — some are bundled with the repo, some you must download
yourself, some are fetched automatically by the pipeline. The table
below is the contract.

## What ships in git, what doesn't

| Path | In git? | Provenance | What to do |
| --- | --- | --- | --- |
| `README.md`                                    | ✅ tracked     | repo doc | nothing |
| `archetypes.csv`                               | ✅ tracked     | repo (canonical archetype list — 11 rows, drives TEASER) | nothing |
| `grid/DE_Grid_ETRS89-UTM32_10km.gpkg`          | ✅ tracked     | BKG (Datenlizenz Deutschland 2.0, redistributable) | nothing |
| `electricity/60min/*.csv`                      | ❌ ignored     | HTW Berlin residential load profile dataset, Tjaden et al. | **download manually** (see below) |
| `electricity/15min/*.csv`                      | ❌ ignored     | same source, finer resolution | optional, only if you want to drive the pipeline at 15-min |
| `weather/2010/loc####.parquet`                 | ❌ ignored     | Open-Meteo historical | **fetched automatically** by the pipeline |

The `.gitignore` configuration enforces this contract. A fresh clone
has the three tracked items and nothing else; everything else is
absent until you provide it or the pipeline downloads it.

## Manual downloads (one-off)

### Electricity profiles — HTW Berlin

74 measured residential load profiles for 2010, curated by Tjaden et al.
(HTW Berlin) from underlying 15-minute IZES smart-meter measurements.
Publicly available via the HTW Berlin research dataset page. Place the
hourly-aggregated CSVs at:

```
input/electricity/60min/<annual_kwh>.csv     # default the pipeline reads
input/electricity/15min/<annual_kwh>.csv     # optional finer resolution
```

Format per file: a CSV with a parsable timestamp index (column 0) and
a single power column in W. The filename stem is the annual demand in
kWh (e.g. `4500.csv`).

### Building grid — BKG 10 km UTM

This file IS bundled with the repo, but if you ever need to refresh it:
Federal Agency for Cartography and Geodesy (BKG), Geographical Grid for
Germany in UTM projection
(https://gdz.bkg.bund.de/index.php/default/geographische-gitter-fur-deutschland-in-utm-projektion-geogitter-utm.html).
Download the 10 km layer as GeoPackage and overwrite
`input/grid/DE_Grid_ETRS89-UTM32_10km.gpkg`.

## Auto-fetched (pipeline-managed)

### Weather — Open-Meteo

`OpenMeteoWeatherProvider` calls the Open-Meteo Historical archive for
each of the 4,045 grid-cell centroids. Free, no API key, rate-limited.
First-run download writes ~538 MB to `input/weather/2010/`; subsequent
runs reuse the cached parquets.

```
coupled-energy-ts run config/germany_2010.yml
```

To skip the download (e.g. on a re-run):

```
coupled-energy-ts run config/germany_2010.yml --skip-weather-fetch
```

## Switching to a different country

Replace the contents of `input/` per the schema, repoint the YAML at
the new files, and the rest of the pipeline runs unchanged. See
`examples/02_byo_country.py` and `examples/02_byo_country.ipynb` for a
runnable template.
