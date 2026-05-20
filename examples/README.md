# Examples

Two pairs of identical-content files: a Jupyter notebook and a runnable
Python script for each example.

## Reproducing the paper dataset

| File | When to use it |
| --- | --- |
| `01_germany_2010.ipynb` | click-through walkthrough with prerequisites + spot-checks |
| `01_germany_2010.py`    | one-shot script (same logic, no narration) |

The notebook documents which inputs you must place on disk yourself
(HTW Berlin profiles, German 10 km grid GeoPackage) and which are
downloaded automatically (Open-Meteo weather). A pre-flight cell aborts
with a clear message if a required file is missing.

```bash
uv run jupyter notebook examples/01_germany_2010.ipynb
# or:
uv run python examples/01_germany_2010.py
```

## Bring your own data

| File | When to use it |
| --- | --- |
| `02_byo_country.ipynb` | step-by-step walkthrough you can edit cell-by-cell |
| `02_byo_country.py`    | one-shot script that materialises the demo |

Both generate a tiny synthetic 2-archetype × 2-profile dataset under
`examples/_byo_demo/`, write a YAML config that points the four
providers at parquet files, and run the B / E / O steps. **No network,
no TEASER, no TABULA** — the smallest end-to-end demo of the pipeline.

Use these as templates when adapting the pipeline to a different
country or data source: replace the synthetic builders with your real
files (matching the canonical schema columns), point the YAML providers
at them, and the rest of the pipeline runs unchanged.

```bash
uv run jupyter notebook examples/02_byo_country.ipynb
# or:
uv run python examples/02_byo_country.py
```

## Paper-figure renderers

Paper figures live in their own folder. See `figures/` for the
Methods-illustration scripts (Figs 2, 3, 4) and `validation/` for the
validation-figure scripts (Figs 5--9).
