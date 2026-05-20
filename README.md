# coupled-energy-ts

[![tests](https://github.com/TUM-Doepfert/coupled-energy-ts/actions/workflows/test.yml/badge.svg)](https://github.com/TUM-Doepfert/coupled-energy-ts/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code for the journal article **"Coupled time series of electricity demand, occupancy, weather, and thermal demand for residential buildings"** (Doepfert et al., TU Munich).

A pluggable Python pipeline that derives coupled hourly residential electricity, occupancy, weather, and heating/cooling time series from any combination of measured load profiles, building archetypes, and historical weather. The pipeline is organised around four small provider Protocols, three thermal-model classes (1R1C, 5R1C ISO 13790, 7R2C VDI 6007), and a single YAML configuration.

The repository ships one example instantiation: Germany 2010, 4,045 locations × 11 archetypes × 74 measured profiles ≈ 3.3 million building-year simulations at hourly resolution. The dataset itself is published on Zenodo at [10.5281/zenodo.20288026](https://doi.org/10.5281/zenodo.20288026); this repository regenerates it byte-for-byte.

## Quick start

```bash
pip install uv
uv sync

# Smoke test (no network, no simulation): ~10 s
uv run coupled-energy-ts run config/germany_2010.yml \
    --skip-weather-fetch --skip-simulation

# Full reproduction of the Germany 2010 dataset:
uv run coupled-energy-ts run config/germany_2010.yml
```

The canonical run takes approximately 8 h on 8 physical cores and produces ~24 GB of output (mostly the per-location, per-archetype heating + cooling parquet files). Steps are idempotent: existing outputs short-circuit unless `--overwrite` is passed.

## Pipeline overview

The four input families (`ArchetypeProvider`, `ElectricityProvider`, `OccupancyProvider`, `WeatherProvider`) each have a `@runtime_checkable` Protocol and a canonical schema defined in `src/providers/base.py`. A new provider is a thin class with one `get_*()` method that returns a DataFrame matching the schema; the YAML `type:` key dispatches in `src/config.py`. Thermal models (`r1c1`, `r5c1`, `r7c2`) live in `src/thermal_models.py`.

See `src/providers/` for provider implementations and schema definitions, and `config/germany_2010.yml` for a complete worked configuration.

## Reproducing the paper figures

Figures 2–4 (Methods illustrations) are produced by scripts in [`figures/`](figures/). Figures 5–9 (Technical Validation) are produced by scripts in [`validation/`](validation/) — see [`validation/README.md`](validation/README.md) for tier order and prerequisites. The US validation tier downloads ~12 GB of NREL EULP data on first run; the German validation needs the When2Heat CSV from [data.open-power-system-data.org/when2heat](https://data.open-power-system-data.org/when2heat/).

## Adapting to another country

The pipeline core does not change; only the four provider sections of the YAML are repointed. A minimal end-to-end example with synthetic data is in [`examples/02_byo_country.py`](examples/02_byo_country.py) and its companion notebook. Each provider documents its expected input shape inline.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/                          # offline suite, ~10 s
uv run pytest tests/ --run-integration        # adds network-touching tests
```

CI runs the offline suite on every push; the integration suite runs weekly.

## Citation

```bibtex
@article{Doepfert2026CoupledTimeSeries,
  title   = {Coupled time series of electricity demand, occupancy, weather, and thermal demand for residential buildings},
  author  = {Doepfert, Markus and {Escobar Viegas}, Maximiliano Fernando and Zinsmeister, Daniel and Tzscheutschler, Peter and Goebel, Christoph and Hamacher, Thomas},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.0000/PLACEHOLDER-PAPER-DOI}
}
```

The Germany 2010 reference dataset has its own DOI on Zenodo: [10.5281/zenodo.20288026](https://doi.org/10.5281/zenodo.20288026). A `CITATION.cff` is included for tooling (Zotero, GitHub "Cite this repository", etc.).

## License

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Germany 2010 dataset (Zenodo)**: CC BY 4.0.
- **BKG 10 km UTM grid bundled at `input/grid/`**: Datenlizenz Deutschland 2.0 (redistributable).
- Third-party inputs (HTW Berlin profiles, Open-Meteo, NREL EULP, When2Heat) retain their original licenses; the pipeline does not redistribute them.

## Repository layout

```
config/germany_2010.yml   reference run specification
src/                      pipeline core (providers, config, thermal models, simulation, CLI)
examples/                 end-to-end usage examples (reproduce paper, BYO country)
figures/                  paper Figs 2, 3, 4 renderers
scripts/                  data preprocessing (HTW Berlin PL1/PL2/PL3 → Record E)
validation/               paper Figs 5–9 (US per-building + load diversity, DE aggregate)
tests/                    offline + opt-in integration pytest suites
input/                    tracked README, archetypes.csv, BKG grid; manual/auto-fetched data
```

## Environment

Python 3.12 (pinned in `.python-version`), managed via [uv](https://docs.astral.sh/uv/). Dependencies pinned in `uv.lock`. The thermal-simulation core uses [EnTiSe](https://github.com/TUM-Doepfert/entise) (by the same author).
