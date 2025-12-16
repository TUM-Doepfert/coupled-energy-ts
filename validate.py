"""
Simple validation script to test the implementation without making API calls.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coupled_ts_paper.building_rc_calculator import BuildingRCCalculator
from coupled_ts_paper.objects_csv_generator import ObjectsCSVGenerator
from coupled_ts_paper.coupled_timeseries_generator import CoupledTimeSeriesGenerator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rc_calculator():
    """Test building RC calculator."""
    logger.info("\nTesting Building RC Calculator...")
    
    calculator = BuildingRCCalculator()
    
    try:
        rc_values = calculator.calculate_rc_values(
            building_name="test_office",
            building_type="office",
            insulation_quality="good",
            net_leased_area=1000.0,
            number_of_floors=3,
            height_of_floors=3.5,
        )
        
        logger.info("✓ RC calculator works")
        logger.info(f"  Sample RC values: building_name={rc_values['building_name']}, "
                   f"year={rc_values['year_of_construction']}")
        return True
    except Exception as e:
        logger.error(f"✗ RC calculator failed: {e}")
        return False


def test_objects_generator():
    """Test objects CSV generator."""
    logger.info("\nTesting Objects CSV Generator...")
    
    try:
        # Create dummy RC values
        rc_df = pd.DataFrame([
            {
                "building_type": "office",
                "insulation_quality": "good",
                "year_of_construction": 2000,
                "net_leased_area": 1000,
                "number_of_floors": 3,
                "height_of_floors": 3.5,
                "volume": 10500,
                "area": 1000,
                "c1_value": 1000000,
                "c2_value": 500000,
                "r1_value": 0.001,
                "r2_value": 0.002,
                "r3_value": 0.0001,
            }
        ])
        
        generator = ObjectsCSVGenerator()
        generator.add_building(
            object_id="test_munich_office_good",
            location="Munich",
            building_type="office",
            insulation_quality="good",
            rc_values=rc_df.iloc[0].to_dict(),
        )
        
        df = generator.get_dataframe()
        logger.info("✓ Objects CSV generator works")
        logger.info(f"  Generated {len(df)} object(s)")
        return True
    except Exception as e:
        logger.error(f"✗ Objects CSV generator failed: {e}")
        return False


def test_timeseries_generator():
    """Test time series generator."""
    logger.info("\nTesting Time Series Generator...")
    
    try:
        # Create dummy weather data
        date_range = pd.date_range(start="2023-01-01", end="2023-01-03", freq="h")
        weather_df = pd.DataFrame({
            "temperature_2m": np.random.uniform(0, 20, len(date_range)),
            "relative_humidity_2m": np.random.uniform(40, 80, len(date_range)),
        }, index=date_range)
        
        location_data = {
            "latitude": 48.1351,
            "longitude": 11.5820,
            "weather_data": weather_df,
        }
        
        building_config = {
            "location": "Munich",
            "building_type": "office",
            "insulation_quality": "good",
            "area": 1000.0,
            "max_occupants": 100,
        }
        
        generator = CoupledTimeSeriesGenerator()
        
        # Generate occupancy
        occupancy_df = generator.generate_occupancy_series(
            start_date="2023-01-01",
            end_date="2023-01-03",
            building_type="office",
            max_occupants=100,
        )
        
        # Generate electricity
        electricity_df = generator.generate_electricity_series(
            occupancy_df=occupancy_df,
            building_type="office",
            area=1000.0,
        )
        
        # Generate coupled series
        coupled_df = generator.generate_coupled_series(
            location_data=location_data,
            building_config=building_config,
            start_date="2023-01-01",
            end_date="2023-01-03",
        )
        
        logger.info("✓ Time series generator works")
        logger.info(f"  Generated {len(coupled_df)} hours of data")
        logger.info(f"  Columns: {list(coupled_df.columns)}")
        return True
    except Exception as e:
        logger.error(f"✗ Time series generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("=" * 80)
    logger.info("Running Validation Tests")
    logger.info("=" * 80)
    
    results = []
    results.append(("RC Calculator", test_rc_calculator()))
    results.append(("Objects Generator", test_objects_generator()))
    results.append(("Time Series Generator", test_timeseries_generator()))
    
    logger.info("\n" + "=" * 80)
    logger.info("Validation Results")
    logger.info("=" * 80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n✓ All validation tests passed!")
        return 0
    else:
        logger.info("\n✗ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
