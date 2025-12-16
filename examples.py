"""
Example usage of the coupled time series generation components.

This script demonstrates how to use individual components of the package.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coupled_ts_paper.location_reader import LocationReader
from coupled_ts_paper.building_rc_calculator import BuildingRCCalculator
from coupled_ts_paper.objects_csv_generator import ObjectsCSVGenerator
from coupled_ts_paper.coupled_timeseries_generator import CoupledTimeSeriesGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_location_reader():
    """Example: Fetch weather data for a single location."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Fetching weather data from OpenMeteo")
    logger.info("=" * 80)
    
    reader = LocationReader()
    
    # Fetch weather data for Munich
    weather_df = reader.fetch_weather_data(
        latitude=48.1351,
        longitude=11.5820,
        start_date="2023-01-01",
        end_date="2023-01-07",
    )
    
    logger.info(f"\nWeather data for Munich (first 5 rows):")
    logger.info(f"\n{weather_df.head()}")
    

def example_rc_calculator():
    """Example: Calculate RC values for a building."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Calculating building RC values with TEASER")
    logger.info("=" * 80)
    
    calculator = BuildingRCCalculator()
    
    # Calculate RC values for an office building
    rc_values = calculator.calculate_rc_values(
        building_name="example_office",
        building_type="office",
        insulation_quality="good",
        net_leased_area=1500.0,
        number_of_floors=4,
        height_of_floors=3.5,
    )
    
    logger.info(f"\nRC values for office building:")
    for key, value in rc_values.items():
        logger.info(f"  {key}: {value}")


def example_objects_csv():
    """Example: Generate objects.csv."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Generating objects.csv")
    logger.info("=" * 80)
    
    # First calculate some RC values
    calculator = BuildingRCCalculator()
    rc_configs = [
        {
            "building_name": "office_good",
            "building_type": "office",
            "insulation_quality": "good",
        },
        {
            "building_name": "residential_moderate",
            "building_type": "residential",
            "insulation_quality": "moderate",
        },
    ]
    rc_df = calculator.calculate_multiple_buildings(rc_configs)
    
    # Generate objects CSV
    generator = ObjectsCSVGenerator()
    generator.add_multiple_buildings(
        locations=["Munich", "Berlin"],
        building_types=["office", "residential"],
        insulation_qualities=["good", "moderate"],
        rc_values_df=rc_df,
    )
    
    objects_df = generator.get_dataframe()
    logger.info(f"\nObjects DataFrame (first 5 rows):")
    logger.info(f"\n{objects_df.head()}")


def example_timeseries():
    """Example: Generate coupled time series."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Generating coupled time series")
    logger.info("=" * 80)
    
    # Fetch weather data
    reader = LocationReader()
    weather_df = reader.fetch_weather_data(
        latitude=48.1351,
        longitude=11.5820,
        start_date="2023-01-01",
        end_date="2023-01-07",
    )
    
    location_data = {
        "latitude": 48.1351,
        "longitude": 11.5820,
        "weather_data": weather_df,
    }
    
    building_config = {
        "location": "Munich",
        "building_type": "office",
        "insulation_quality": "good",
        "area": 1500.0,
        "max_occupants": 150,
    }
    
    # Generate coupled time series
    generator = CoupledTimeSeriesGenerator()
    coupled_df = generator.generate_coupled_series(
        location_data=location_data,
        building_config=building_config,
        start_date="2023-01-01",
        end_date="2023-01-07",
    )
    
    logger.info(f"\nCoupled time series (first 24 hours):")
    logger.info(f"\n{coupled_df.head(24)}")


if __name__ == "__main__":
    # Run all examples
    example_location_reader()
    example_rc_calculator()
    example_objects_csv()
    example_timeseries()
    
    logger.info("\n" + "=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)
