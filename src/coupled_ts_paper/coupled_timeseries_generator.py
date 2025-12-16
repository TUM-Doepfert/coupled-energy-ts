"""Generator for coupled electricity, occupancy, and HVAC time series."""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class CoupledTimeSeriesGenerator:
    """Generator for creating coupled energy time series data."""

    # Typical occupancy schedules (hourly patterns 0-23)
    OCCUPANCY_SCHEDULES = {
        "office": {
            "weekday": [0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.7, 0.9, 0.9, 0.8,
                       0.5, 0.8, 0.9, 0.9, 0.7, 0.3, 0.1, 0, 0, 0, 0, 0],
            "weekend": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1,
                       0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "residential": {
            "weekday": [0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2,
                       0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 0.8],
            "weekend": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.7, 0.7,
                       0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.8],
        },
        "institute": {
            "weekday": [0, 0, 0, 0, 0, 0, 0.1, 0.4, 0.8, 0.9, 0.9, 0.9,
                       0.6, 0.9, 0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0, 0, 0, 0],
            "weekend": [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2,
                       0.2, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
        },
    }

    # Base electricity consumption profiles (W/m²)
    BASE_ELECTRICITY = {
        "office": {"base": 5.0, "occupied": 15.0},
        "residential": {"base": 3.0, "occupied": 8.0},
        "institute": {"base": 4.0, "occupied": 12.0},
    }

    def __init__(self):
        """Initialize the CoupledTimeSeriesGenerator."""
        pass

    def generate_occupancy_series(
        self,
        start_date: str,
        end_date: str,
        building_type: str,
        max_occupants: int = 100,
    ) -> pd.DataFrame:
        """
        Generate occupancy time series.

        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            building_type: Type of building
            max_occupants: Maximum number of occupants

        Returns:
            DataFrame with occupancy time series
        """
        # Create hourly datetime index
        date_range = pd.date_range(start=start_date, end=end_date, freq="h")
        
        # Get occupancy schedule
        schedule = self.OCCUPANCY_SCHEDULES.get(
            building_type, self.OCCUPANCY_SCHEDULES["office"]
        )
        
        occupancy = []
        for timestamp in date_range:
            hour = timestamp.hour
            is_weekend = timestamp.dayofweek >= 5
            
            if is_weekend:
                base_occupancy = schedule["weekend"][hour]
            else:
                base_occupancy = schedule["weekday"][hour]
            
            # Add some random variation
            occupancy_factor = base_occupancy + np.random.normal(0, 0.05)
            occupancy_factor = np.clip(occupancy_factor, 0, 1)
            
            occupancy.append(int(occupancy_factor * max_occupants))
        
        df = pd.DataFrame({
            "timestamp": date_range,
            "occupancy": occupancy,
        })
        df.set_index("timestamp", inplace=True)
        
        return df

    def generate_electricity_series(
        self,
        occupancy_df: pd.DataFrame,
        building_type: str,
        area: float,
    ) -> pd.DataFrame:
        """
        Generate electricity consumption time series.

        Args:
            occupancy_df: DataFrame with occupancy data
            building_type: Type of building
            area: Building area in m²

        Returns:
            DataFrame with electricity consumption time series
        """
        electricity_params = self.BASE_ELECTRICITY.get(
            building_type, self.BASE_ELECTRICITY["office"]
        )
        
        max_occupancy = occupancy_df["occupancy"].max()
        if max_occupancy == 0:
            max_occupancy = 1
        
        electricity = []
        for idx, row in occupancy_df.iterrows():
            occupancy_ratio = row["occupancy"] / max_occupancy
            
            # Base load + occupancy-dependent load
            power_density = (
                electricity_params["base"] +
                occupancy_ratio * (electricity_params["occupied"] - electricity_params["base"])
            )
            
            # Add some random variation
            power_density *= (1 + np.random.normal(0, 0.1))
            power_density = max(0, power_density)
            
            # Total power in kW
            total_power = power_density * area / 1000
            electricity.append(total_power)
        
        df = occupancy_df.copy()
        df["electricity_kw"] = electricity
        
        return df

    def generate_hvac_series(
        self,
        weather_df: pd.DataFrame,
        occupancy_df: pd.DataFrame,
        building_type: str,
        area: float,
        insulation_quality: str,
    ) -> pd.DataFrame:
        """
        Generate HVAC energy consumption time series.

        Args:
            weather_df: DataFrame with weather data
            occupancy_df: DataFrame with occupancy data
            building_type: Type of building
            area: Building area in m²
            insulation_quality: Quality of insulation

        Returns:
            DataFrame with HVAC consumption time series
        """
        # Insulation factors (better insulation = lower energy consumption)
        insulation_factors = {
            "poor": 1.5,
            "moderate": 1.0,
            "good": 0.7,
            "excellent": 0.5,
        }
        insulation_factor = insulation_factors.get(insulation_quality, 1.0)
        
        # Align dataframes
        combined_df = occupancy_df.copy()
        
        # Get temperature from weather data if available
        if "temperature_2m" in weather_df.columns:
            combined_df = combined_df.join(weather_df[["temperature_2m"]], how="left")
            combined_df["temperature_2m"] = combined_df["temperature_2m"].fillna(20)
        else:
            # Default temperature
            combined_df["temperature_2m"] = 20
        
        hvac_heating = []
        hvac_cooling = []
        
        for idx, row in combined_df.iterrows():
            temp = row["temperature_2m"]
            occupancy_ratio = row["occupancy"] / occupancy_df["occupancy"].max() if occupancy_df["occupancy"].max() > 0 else 0
            
            # Heating when temp < 18°C, cooling when temp > 24°C
            heating_power = 0
            cooling_power = 0
            
            if temp < 18:
                # Heating demand (W/m²)
                heating_demand = (18 - temp) * 15 * insulation_factor
                heating_power = heating_demand * area / 1000  # kW
            elif temp > 24:
                # Cooling demand (W/m²)
                cooling_demand = (temp - 24) * 20 * insulation_factor
                cooling_power = cooling_demand * area / 1000  # kW
            
            # Increase HVAC with occupancy (ventilation)
            ventilation_load = occupancy_ratio * area * 0.01  # kW
            
            # Add random variation
            heating_power *= (1 + np.random.normal(0, 0.1))
            cooling_power *= (1 + np.random.normal(0, 0.1))
            
            hvac_heating.append(max(0, heating_power))
            hvac_cooling.append(max(0, cooling_power + ventilation_load))
        
        combined_df["hvac_heating_kw"] = hvac_heating
        combined_df["hvac_cooling_kw"] = hvac_cooling
        combined_df["hvac_total_kw"] = combined_df["hvac_heating_kw"] + combined_df["hvac_cooling_kw"]
        
        return combined_df

    def generate_coupled_series(
        self,
        location_data: Dict[str, any],
        building_config: Dict[str, any],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Generate coupled time series for a building at a location.

        Args:
            location_data: Dictionary with location and weather data
            building_config: Dictionary with building configuration
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'

        Returns:
            DataFrame with all coupled time series
        """
        logger.info(
            f"Generating coupled series for {building_config['building_type']} "
            f"at {building_config.get('location', 'unknown')}"
        )
        
        # Generate occupancy
        occupancy_df = self.generate_occupancy_series(
            start_date=start_date,
            end_date=end_date,
            building_type=building_config["building_type"],
            max_occupants=building_config.get("max_occupants", 100),
        )
        
        # Generate electricity
        electricity_df = self.generate_electricity_series(
            occupancy_df=occupancy_df,
            building_type=building_config["building_type"],
            area=building_config.get("area", 1000),
        )
        
        # Generate HVAC
        weather_df = location_data.get("weather_data", pd.DataFrame())
        hvac_df = self.generate_hvac_series(
            weather_df=weather_df,
            occupancy_df=occupancy_df,
            building_type=building_config["building_type"],
            area=building_config.get("area", 1000),
            insulation_quality=building_config.get("insulation_quality", "moderate"),
        )
        
        return hvac_df

    def save_time_series(
        self,
        df: pd.DataFrame,
        output_path: str,
        object_id: str,
    ):
        """
        Save time series to CSV file.

        Args:
            df: DataFrame with time series data
            output_path: Directory path where to save the file
            object_id: Object identifier for filename
        """
        output_file = Path(output_path) / f"{object_id}_timeseries.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file)
        logger.info(f"Saved time series to {output_file}")
