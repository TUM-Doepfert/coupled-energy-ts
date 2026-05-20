"""Integration test: OpenMeteoWeatherProvider fetches one location.

Hits the Open-Meteo historical archive over the public internet. Auto-
skips when outbound HTTPS to ``archive-api.open-meteo.com`` is not
reachable so the test can run on disconnected machines without failing.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest

from src.providers import OpenMeteoWeatherProvider, WeatherSchema

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _internet_reachable(host: str = "archive-api.open-meteo.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_openmeteo_fetches_one_location(tmp_path: Path):
    if not _internet_reachable():
        pytest.skip("archive-api.open-meteo.com not reachable")

    # Build a 1-row locations CSV (Berlin) and feed it through the provider.
    cache_dir = tmp_path / "weather"
    cache_dir.mkdir()
    locations_csv = tmp_path / "locations.csv"
    pd.DataFrame([
        {"location_id": 1, "latitude": 52.52, "longitude": 13.40},
    ]).to_csv(locations_csv, index=False)

    provider = OpenMeteoWeatherProvider(
        grid_path=tmp_path / "_unused.gpkg",  # only locations_csv path is read
        cache_dir=cache_dir,
        interpolate_15min=False,
        locations_csv=locations_csv,
    )

    df = provider.get_weather(location_id=1, year=2010)
    for col in WeatherSchema.REQUIRED:
        assert col in df.columns, f"Open-Meteo output missing {col}"
    assert 8700 <= len(df) <= 8800, len(df)
    # Berlin Jan 2010 was unusually cold (~-2 °C monthly mean) but
    # bounds well within any plausible reading.
    air = df["air_temperature"]
    assert -40.0 < air.min() < air.max() < 50.0
