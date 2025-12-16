"""
Configuration for German locations and building types.

This file contains example configurations for different locations in Germany
and building type/insulation quality combinations.
"""

# German cities with coordinates
GERMAN_LOCATIONS = [
    {
        "name": "Munich",
        "latitude": 48.1351,
        "longitude": 11.5820,
    },
    {
        "name": "Berlin",
        "latitude": 52.5200,
        "longitude": 13.4050,
    },
    {
        "name": "Hamburg",
        "latitude": 53.5511,
        "longitude": 9.9937,
    },
    {
        "name": "Frankfurt",
        "latitude": 50.1109,
        "longitude": 8.6821,
    },
    {
        "name": "Stuttgart",
        "latitude": 48.7758,
        "longitude": 9.1829,
    },
]

# Building types to analyze
BUILDING_TYPES = [
    "office",
    "residential",
    "institute",
]

# Insulation quality levels
INSULATION_QUALITIES = [
    "poor",
    "moderate",
    "good",
    "excellent",
]

# Building parameters for RC calculation
BUILDING_PARAMETERS = {
    "office": {
        "net_leased_area": 1500.0,
        "number_of_floors": 4,
        "height_of_floors": 3.5,
        "max_occupants": 150,
    },
    "residential": {
        "net_leased_area": 150.0,
        "number_of_floors": 2,
        "height_of_floors": 2.8,
        "max_occupants": 4,
    },
    "institute": {
        "net_leased_area": 2000.0,
        "number_of_floors": 5,
        "height_of_floors": 4.0,
        "max_occupants": 200,
    },
}

# Time period for time series generation
TIME_PERIOD = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}
