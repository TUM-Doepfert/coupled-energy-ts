"""Pipeline orchestrator.

Reads a YAML config, builds the four providers, and runs them in order:

    1. Archetype provider  -> output/B.parquet
    2. Electricity provider -> output/E.parquet
    3. Occupancy provider   -> output/O.parquet (depends on E)
    4. Weather provider     -> input/weather/<year>/loc####.parquet
                               + output/location_mapping.csv
    5. Simulation           -> output/HC/locXXXX/hc_archYY.parquet

Each step is idempotent on a per-output basis. Re-running with existing
outputs short-circuits unless ``--overwrite`` is passed (or the YAML's
``simulation.overwrite: true`` is set for the simulation step).

The simulation step reuses the existing ``main_simulation`` driver in
``src/simulation.py`` for now. Phase 3 will introduce thermal-model
swap (1R1C / 5R1C / 7R2C) here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Providers, build_providers, load_config


@dataclass(frozen=True)
class StepResult:
    name: str
    output: Path | None
    rows: int | None = None
    skipped: bool = False
    detail: str = ""


@dataclass
class RunResult:
    output_dir: Path
    steps: list[StepResult]

    def __str__(self) -> str:  # pragma: no cover - human-readable summary
        lines = [f"Pipeline complete. Outputs in {self.output_dir}/"]
        for s in self.steps:
            tag = "skip" if s.skipped else "ok  "
            extras = []
            if s.rows is not None:
                extras.append(f"{s.rows:,} rows")
            if s.detail:
                extras.append(s.detail)
            extra = f"  ({', '.join(extras)})" if extras else ""
            lines.append(f"  [{tag}] {s.name:<14} -> {s.output}{extra}")
        return "\n".join(lines)


def _b_path(output_dir: Path) -> Path:
    return output_dir / "B.parquet"


def _e_path(output_dir: Path) -> Path:
    return output_dir / "E.parquet"


def _o_path(output_dir: Path) -> Path:
    return output_dir / "O.parquet"


def _location_mapping_path(output_dir: Path) -> Path:
    return output_dir / "location_mapping.csv"


def run(
    config_path: Path,
    overwrite: bool = False,
    skip_weather_fetch: bool = False,
    skip_simulation: bool = False,
) -> RunResult:
    """Execute the full pipeline as configured by ``config_path``."""
    cfg = load_config(config_path)
    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    providers = build_providers(cfg)
    year = int(cfg.get("year", 2010))
    steps: list[StepResult] = []

    # 1. Archetypes
    b = _b_path(output_dir)
    if b.exists() and not overwrite:
        steps.append(StepResult("archetypes", b, skipped=True))
    else:
        providers.archetype.save(b)
        df = providers.archetype.get_archetypes()
        steps.append(StepResult("archetypes", b, rows=len(df)))

    # 2. Electricity
    e = _e_path(output_dir)
    if e.exists() and not overwrite:
        steps.append(StepResult("electricity", e, skipped=True))
    else:
        providers.electricity.save(e, year=year)
        # cheap re-read just for the count
        import pandas as pd
        rows = len(pd.read_parquet(e))
        steps.append(StepResult("electricity", e, rows=rows))

    # 3. Occupancy (depends on E)
    o = _o_path(output_dir)
    if o.exists() and not overwrite:
        steps.append(StepResult("occupancy", o, skipped=True))
    else:
        import pandas as pd
        elec = pd.read_parquet(e)
        providers.occupancy.save(o, electricity=elec)
        steps.append(StepResult("occupancy", o, rows=len(pd.read_parquet(o))))

    # 4. Weather mapping + per-location fetch
    locs = providers.weather.list_locations()
    locs.to_csv(_location_mapping_path(output_dir), index=False)
    weather_step_detail = f"{len(locs)} locations"
    if skip_weather_fetch:
        steps.append(StepResult(
            "weather", _location_mapping_path(output_dir),
            skipped=True, detail=weather_step_detail + " (fetch skipped)",
        ))
    else:
        providers.weather.fetch_all(year=year, output_dir=None)
        steps.append(StepResult(
            "weather", _location_mapping_path(output_dir),
            detail=weather_step_detail,
        ))

    # 5. Simulation
    if skip_simulation:
        steps.append(StepResult("simulation", output_dir / "HC", skipped=True))
    else:
        sim_cfg = cfg.get("simulation", {}) or {}
        thermal_model_name = str(cfg.get("thermal_model", "r1c1"))
        try:
            from . import simulation as _sim
        except ImportError as exc:
            raise RuntimeError(
                "src/simulation.py could not be imported — usually because "
                "EnTiSe is not installed in the active environment."
            ) from exc

        # Require the load-bearing simulation keys explicitly rather than
        # silently falling back: a user who drops one of these from the YAML
        # is asking for a different dataset than the paper published, and a
        # silent default produces wrong numbers without a stack trace.
        for required_key in ("gains_per_person_W", "ach_model"):
            if required_key not in sim_cfg:
                raise KeyError(
                    f"Config section 'simulation' missing required key "
                    f"{required_key!r}. See config/germany_2010.yml for the "
                    f"reference values."
                )

        weather_cache = cfg.get("weather_provider", {}).get("cache_dir")
        # Resolve YAML 'locations' to a list of ints, "all", or "random:N".
        locations_cfg = cfg.get("locations", "all")
        n_written = _sim.run_simulation(
            output_dir=output_dir,
            weather_dir=Path(weather_cache) if weather_cache else None,
            location_mapping=output_dir / "location_mapping.csv",
            thermal_model_name=thermal_model_name,
            heating_setpoint_C=float(sim_cfg.get("heating_setpoint_C", 20.0)),
            cooling_setpoint_C=float(sim_cfg.get("cooling_setpoint_C", 26.0)),
            inhabitants=int(sim_cfg.get("inhabitants", 2)),
            gains_per_person_W=float(sim_cfg["gains_per_person_W"]),
            ach_model=str(sim_cfg["ach_model"]),
            resolution=str(sim_cfg.get("resolution", "60min")),
            overwrite=bool(sim_cfg.get("overwrite", overwrite)),
            n_jobs=int(cfg.get("n_jobs", -1)),
            locations_filter=locations_cfg,
        )
        steps.append(StepResult(
            "simulation", output_dir / "HC", rows=n_written,
            detail=f"thermal_model={thermal_model_name!r}",
        ))

    return RunResult(output_dir=output_dir, steps=steps)


__all__ = ["run", "RunResult", "StepResult"]
