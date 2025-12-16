"""
Main script for generating coupled energy time series data.

This script demonstrates the complete workflow:
1. Reading locations from OpenMeteo
2. Calculating building RC values with TEASER
3. Generating objects.csv for ENTISE
4. Creating coupled electricity, occupancy, and HVAC time series
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
from config.germany_config import (
    GERMAN_LOCATIONS,
    BUILDING_TYPES,
    INSULATION_QUALITIES,
    BUILDING_PARAMETERS,
    TIME_PERIOD,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to generate all coupled time series data."""
    
    logger.info("=" * 80)
    logger.info("Starting Coupled Energy Time Series Data Generation")
    logger.info("=" * 80)
    
    # Create output directories
    data_dir = Path(__file__).parent / "data"
    timeseries_dir = data_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch location data from OpenMeteo
    logger.info("\nStep 1: Fetching weather data from OpenMeteo")
    logger.info("-" * 80)
    
    location_reader = LocationReader()
    location_configs = [
        {
            **loc,
            "start_date": TIME_PERIOD["start_date"],
            "end_date": TIME_PERIOD["end_date"],
        }
        for loc in GERMAN_LOCATIONS
    ]
    
    locations_data = location_reader.fetch_locations(location_configs)
    logger.info(f"Fetched weather data for {len(locations_data)} locations")
    
    # Step 2: Calculate RC values for different building types and insulation qualities
    logger.info("\nStep 2: Calculating building RC values using TEASER")
    logger.info("-" * 80)
    
    rc_calculator = BuildingRCCalculator()
    building_configs = []
    
    for building_type in BUILDING_TYPES:
        for insulation_quality in INSULATION_QUALITIES:
            building_name = f"{building_type}_{insulation_quality}"
            params = BUILDING_PARAMETERS[building_type]
            
            config = {
                "building_name": building_name,
                "building_type": building_type,
                "insulation_quality": insulation_quality,
                **params,
            }
            building_configs.append(config)
    
    rc_values_df = rc_calculator.calculate_multiple_buildings(building_configs)
    logger.info(f"Calculated RC values for {len(rc_values_df)} building configurations")
    
    # Save RC values
    rc_values_path = data_dir / "rc_values.csv"
    rc_values_df.to_csv(rc_values_path, index=False)
    logger.info(f"Saved RC values to {rc_values_path}")
    
    # Step 3: Generate objects.csv for ENTISE
    logger.info("\nStep 3: Generating objects.csv for ENTISE")
    logger.info("-" * 80)
    
    objects_generator = ObjectsCSVGenerator()
    location_names = [loc["name"] for loc in GERMAN_LOCATIONS]
    
    objects_generator.add_multiple_buildings(
        locations=location_names,
        building_types=BUILDING_TYPES,
        insulation_qualities=INSULATION_QUALITIES,
        rc_values_df=rc_values_df,
    )
    
    objects_csv_path = data_dir / "objects.csv"
    objects_df = objects_generator.generate_csv(objects_csv_path)
    logger.info(f"Generated objects.csv with {len(objects_df)} objects")
    
    # Step 4: Generate coupled time series for each object
    logger.info("\nStep 4: Generating coupled time series")
    logger.info("-" * 80)
    
    ts_generator = CoupledTimeSeriesGenerator()
    
    for location_name in location_names:
        location_data = locations_data[location_name]
        
        for building_type in BUILDING_TYPES:
            for insulation_quality in INSULATION_QUALITIES:
                object_id = f"{location_name}_{building_type}_{insulation_quality}"
                
                # Get building parameters
                params = BUILDING_PARAMETERS[building_type]
                
                # Get RC values
                rc_row = rc_values_df[
                    (rc_values_df["building_type"] == building_type) &
                    (rc_values_df["insulation_quality"] == insulation_quality)
                ]
                
                if rc_row.empty:
                    logger.warning(f"No RC values for {object_id}, skipping")
                    continue
                
                area = rc_row.iloc[0]["area"] if "area" in rc_row.columns else params["net_leased_area"]
                
                building_config = {
                    "location": location_name,
                    "building_type": building_type,
                    "insulation_quality": insulation_quality,
                    "area": area,
                    "max_occupants": params["max_occupants"],
                }
                
                # Generate coupled time series
                try:
                    coupled_df = ts_generator.generate_coupled_series(
                        location_data=location_data,
                        building_config=building_config,
                        start_date=TIME_PERIOD["start_date"],
                        end_date=TIME_PERIOD["end_date"],
                    )
                    
                    # Save time series
                    ts_generator.save_time_series(
                        df=coupled_df,
                        output_path=timeseries_dir,
                        object_id=object_id,
                    )
                except Exception as e:
                    logger.error(f"Error generating time series for {object_id}: {e}")
                    continue
    
    logger.info("\n" + "=" * 80)
    logger.info("Data generation completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nGenerated files:")
    logger.info(f"  - RC values: {rc_values_path}")
    logger.info(f"  - Objects CSV: {objects_csv_path}")
    logger.info(f"  - Time series: {timeseries_dir}/ ({len(list(timeseries_dir.glob('*.csv')))} files)")
    logger.info("\n")


if __name__ == "__main__":
    main()
