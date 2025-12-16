# Coupled Energy Time Series Data Generation

This repository contains the code for generating coupled energy time series data for a research paper about coupled time series methods. The generated data includes electricity consumption, occupancy, and HVAC (Heating, Ventilation, and Air Conditioning) time series for different building types and insulation qualities across multiple locations in Germany.

## Features

- **Location Data from OpenMeteo**: Fetches historical weather data for specified German locations
- **Building RC Calculations with TEASER**: Calculates thermal Resistance-Capacitance (RC) values for different building types and insulation qualities
- **Objects CSV Generation**: Creates objects.csv file compatible with the ENTISE framework
- **Coupled Time Series Generation**: Generates realistic electricity, occupancy, and HVAC time series that are coupled with weather conditions

## Project Structure

```
coupled_ts_paper/
├── src/
│   └── coupled_ts_paper/
│       ├── __init__.py
│       ├── location_reader.py              # OpenMeteo API integration
│       ├── building_rc_calculator.py       # TEASER integration for RC values
│       ├── objects_csv_generator.py        # Objects CSV generation for ENTISE
│       └── coupled_timeseries_generator.py # Time series generation
├── config/
│   └── germany_config.py                   # Configuration for locations and buildings
├── data/                                    # Generated data output directory
│   ├── rc_values.csv                       # Building RC values
│   ├── objects.csv                         # Objects file for ENTISE
│   └── timeseries/                         # Generated time series CSV files
├── generate_data.py                        # Main script to run the full pipeline
├── pyproject.toml                          # Python project configuration
└── README.md                               # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install Dependencies

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Quick Start

Run the main data generation script:

```bash
python generate_data.py
```

This will:
1. Fetch weather data from OpenMeteo for all configured German locations
2. Calculate RC values for all building type and insulation quality combinations
3. Generate the objects.csv file
4. Create coupled time series for each location-building-insulation combination

### Configuration

Edit `config/germany_config.py` to customize:

- **Locations**: Add or modify German cities (latitude/longitude)
- **Building Types**: Configure building types (office, residential, institute)
- **Insulation Qualities**: Set insulation levels (poor, moderate, good, excellent)
- **Time Period**: Define the date range for time series generation
- **Building Parameters**: Customize area, floors, occupancy per building type

### Output

Generated data is saved in the `data/` directory:

- `rc_values.csv`: RC values for all building configurations
- `objects.csv`: Objects file for use with ENTISE
- `timeseries/*.csv`: Coupled time series files (one per location-building-insulation combination)

Each time series file contains:
- `timestamp`: Hourly timestamps
- `occupancy`: Number of occupants
- `electricity_kw`: Electricity consumption in kW
- `hvac_heating_kw`: HVAC heating consumption in kW
- `hvac_cooling_kw`: HVAC cooling consumption in kW
- `hvac_total_kw`: Total HVAC consumption in kW
- `temperature_2m`: Outdoor temperature in °C

## Building Types

The repository supports three main building types:

1. **Office**: Commercial office buildings
2. **Residential**: Single family dwellings
3. **Institute**: Educational/research buildings

## Insulation Qualities

Four insulation quality levels are defined:

1. **Poor**: Buildings from ~1960 with minimal insulation
2. **Moderate**: Buildings from ~1980 with basic insulation
3. **Good**: Buildings from ~2000 with improved insulation
4. **Excellent**: Buildings from ~2015 with modern insulation standards

## Dependencies

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computations
- **requests**: HTTP requests for OpenMeteo API
- **teaser**: Building thermal modeling (RC calculations)
- **pytz**: Timezone handling

## Development

### Code Formatting

```bash
black src/
```

### Linting

```bash
ruff check src/
```

### Testing

```bash
pytest
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this code or data in your research, please cite:

[Citation information to be added]

## Contact

[Contact information to be added]
