"""Reference example: reproduce the paper's Germany 2010 dataset.

This script runs the same pipeline the paper relies on, driven entirely
by ``config/germany_2010.yml`` (TEASER archetypes + HTW Berlin profiles
+ GeoMA occupancy + Open-Meteo weather + 1R1C simulation).

Notes
-----
* First run: weather download is slow (rate-limited; ~4045 grid cells).
  Subsequent runs reuse cached parquet files in ``input/weather/2010/``.
* Simulation over the full grid produces ~3.6 M building-year sims and
  takes hours on a laptop. To smoke-test the wiring quickly, pass
  ``--skip-simulation`` or set ``locations: [1, 2, 3]`` in the YAML.

Usage
-----
    uv run python examples/01_germany_2010.py
    # or:
    uv run coupled-energy-ts run config/germany_2010.yml \\
        --skip-weather-fetch  --skip-simulation
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when running from the examples/ folder.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.resolve()))

from src.pipeline import run  # noqa: E402

CONFIG = (HERE.parent / "config" / "germany_2010.yml").resolve()


def main() -> None:
    if not CONFIG.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG}")

    result = run(
        CONFIG,
        overwrite=False,            # idempotent re-runs skip done steps
        skip_weather_fetch=True,    # toggle off only on a fresh checkout
        skip_simulation=True,       # full sim is hours; flip when ready
    )
    print(result)


if __name__ == "__main__":
    main()
