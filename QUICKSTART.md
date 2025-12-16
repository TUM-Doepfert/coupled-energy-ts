# Quick Start Guide

This guide will help you quickly get started with the coupled energy time series data generation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TUM-Doepfert/coupled_ts_paper.git
cd coupled_ts_paper
```

2. Install the package:
```bash
pip install -e .
```

## Quick Start

### Option 1: Generate All Data (Recommended for First-Time Users)

Run the main script to generate all coupled time series data:

```bash
python generate_data.py
```

This will:
- Fetch weather data from OpenMeteo for all German locations
- Calculate RC values for all building configurations
- Generate `data/objects.csv` for ENTISE
- Create coupled time series files in `data/timeseries/`

**Note**: This may take several minutes depending on your internet connection and the number of locations/building combinations.

### Option 2: Run Examples

See individual component usage:

```bash
python examples.py
```

This demonstrates how to use each module independently.

### Option 3: Use as a Library

Import and use individual components in your own scripts:

```python
from coupled_ts_paper.location_reader import LocationReader
from coupled_ts_paper.building_rc_calculator import BuildingRCCalculator
from coupled_ts_paper.coupled_timeseries_generator import CoupledTimeSeriesGenerator

# Fetch weather data
reader = LocationReader()
weather_df = reader.fetch_weather_data(
    latitude=48.1351,
    longitude=11.5820,
    start_date="2023-01-01",
    end_date="2023-12-31",
)

# Calculate RC values
calculator = BuildingRCCalculator()
rc_values = calculator.calculate_rc_values(
    building_name="my_office",
    building_type="office",
    insulation_quality="good",
)

# Generate time series
generator = CoupledTimeSeriesGenerator()
# ... see examples.py for more details
```

## Customization

### Modify Locations

Edit `config/germany_config.py` to add/remove locations:

```python
GERMAN_LOCATIONS = [
    {
        "name": "YourCity",
        "latitude": 50.0,
        "longitude": 10.0,
    },
    # ... more locations
]
```

### Modify Building Types

Edit `config/germany_config.py` to change building types and parameters:

```python
BUILDING_TYPES = [
    "office",
    "residential",
    "institute",
]

BUILDING_PARAMETERS = {
    "office": {
        "net_leased_area": 1500.0,
        "number_of_floors": 4,
        "height_of_floors": 3.5,
        "max_occupants": 150,
    },
    # ... more types
}
```

### Modify Time Period

Edit `config/germany_config.py`:

```python
TIME_PERIOD = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}
```

## Output Files

After running `generate_data.py`, you'll find:

- **`data/rc_values.csv`**: RC values for all building configurations
- **`data/objects.csv`**: Objects file for ENTISE framework
- **`data/timeseries/`**: Directory containing time series CSV files

Each time series file is named: `{location}_{building_type}_{insulation_quality}_timeseries.csv`

Example: `Munich_office_good_timeseries.csv`

## Time Series Data Format

Each time series CSV contains:

| Column | Description | Unit |
|--------|-------------|------|
| timestamp | Hourly timestamp | - |
| occupancy | Number of occupants | persons |
| electricity_kw | Electricity consumption | kW |
| hvac_heating_kw | Heating consumption | kW |
| hvac_cooling_kw | Cooling consumption | kW |
| hvac_total_kw | Total HVAC consumption | kW |
| temperature_2m | Outdoor temperature | °C |

## Validation

Test your installation:

```bash
python validate.py
```

All tests should pass with ✓ PASS status.

## Common Issues

### Issue: Import errors
**Solution**: Make sure you installed the package with `pip install -e .`

### Issue: Network errors when fetching weather data
**Solution**: Check your internet connection and try again. OpenMeteo API might be temporarily unavailable.

### Issue: TEASER errors
**Solution**: Make sure the TEASER library is installed correctly. The package should handle this automatically.

## Next Steps

1. Review the generated data in `data/timeseries/`
2. Use `data/objects.csv` with the ENTISE framework
3. Customize locations and building types for your research
4. Analyze the coupled time series for your paper

## Support

For issues or questions, please open an issue on GitHub.
