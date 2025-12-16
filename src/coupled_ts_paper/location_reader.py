"""Location reader module for fetching weather data from Open-Meteo API."""

import logging
from typing import Dict, List, Optional
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class LocationReader:
    """Reader for fetching weather and location data from Open-Meteo API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        """Initialize the LocationReader."""
        self.session = requests.Session()

    def fetch_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        hourly_params: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for a location.

        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            hourly_params: List of hourly parameters to fetch. Default includes
                          temperature, relative humidity, and precipitation.

        Returns:
            DataFrame with weather data indexed by datetime
        """
        if hourly_params is None:
            hourly_params = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "shortwave_radiation",
            ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_params),
            "timezone": "auto",
        }

        logger.info(f"Fetching weather data for lat={latitude}, lon={longitude}")
        response = self.session.get(self.ARCHIVE_URL, params=params)
        response.raise_for_status()

        data = response.json()
        
        # Convert to DataFrame
        hourly_data = data.get("hourly", {})
        df = pd.DataFrame(hourly_data)
        
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        
        logger.info(f"Fetched {len(df)} hours of weather data")
        return df

    def fetch_locations(
        self,
        location_configs: List[Dict[str, any]]
    ) -> Dict[str, Dict[str, any]]:
        """
        Fetch weather data for multiple locations.

        Args:
            location_configs: List of location configuration dictionaries.
                Each dict should contain: name, latitude, longitude, start_date, end_date

        Returns:
            Dictionary mapping location names to their data
        """
        locations_data = {}
        
        for config in location_configs:
            name = config["name"]
            logger.info(f"Processing location: {name}")
            
            weather_df = self.fetch_weather_data(
                latitude=config["latitude"],
                longitude=config["longitude"],
                start_date=config["start_date"],
                end_date=config["end_date"],
                hourly_params=config.get("hourly_params"),
            )
            
            locations_data[name] = {
                "latitude": config["latitude"],
                "longitude": config["longitude"],
                "weather_data": weather_df,
            }
        
        return locations_data
