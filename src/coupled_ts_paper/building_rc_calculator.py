"""Building RC calculator module using TEASER library."""

import logging
from typing import Dict, List, Optional
import pandas as pd
from teaser.project import Project

logger = logging.getLogger(__name__)


class BuildingRCCalculator:
    """Calculator for building thermal RC parameters using TEASER."""

    # Common building types in TEASER
    BUILDING_TYPES = [
        "office",
        "institute",
        "institute4",
        "institute8",
        "residential",
        "single_family_dwelling",
    ]

    # Construction years for different insulation qualities
    CONSTRUCTION_YEARS = {
        "poor": 1960,
        "moderate": 1980,
        "good": 2000,
        "excellent": 2015,
    }

    def __init__(self):
        """Initialize the BuildingRCCalculator."""
        self.projects = {}

    def calculate_rc_values(
        self,
        building_name: str,
        building_type: str,
        insulation_quality: str,
        net_leased_area: float = 1000.0,
        number_of_floors: int = 3,
        height_of_floors: float = 3.5,
    ) -> Dict[str, any]:
        """
        Calculate RC values for a building using TEASER.

        Args:
            building_name: Name identifier for the building
            building_type: Type of building (office, residential, etc.)
            insulation_quality: Quality of insulation (poor, moderate, good, excellent)
            net_leased_area: Net leased area in m²
            number_of_floors: Number of floors
            height_of_floors: Height of each floor in meters

        Returns:
            Dictionary containing RC values and building parameters
        """
        logger.info(
            f"Calculating RC values for {building_name} "
            f"({building_type}, {insulation_quality})"
        )

        # Create TEASER project
        prj = Project(load_data=True)
        prj.name = f"{building_name}_project"

        # Get construction year based on insulation quality
        year_of_construction = self.CONSTRUCTION_YEARS.get(insulation_quality, 2000)

        # Add building based on type
        if building_type == "office":
            prj.add_non_residential(
                method="bmvbs",
                usage="office",
                name=building_name,
                year_of_construction=year_of_construction,
                number_of_floors=number_of_floors,
                height_of_floors=height_of_floors,
                net_leased_area=net_leased_area,
            )
        elif building_type in ["institute", "institute4", "institute8"]:
            prj.add_non_residential(
                method="bmvbs",
                usage="institute",
                name=building_name,
                year_of_construction=year_of_construction,
                number_of_floors=number_of_floors,
                height_of_floors=height_of_floors,
                net_leased_area=net_leased_area,
            )
        elif building_type in ["residential", "single_family_dwelling"]:
            prj.add_residential(
                method="iwu",
                usage="single_family_dwelling",
                name=building_name,
                year_of_construction=year_of_construction,
                number_of_floors=number_of_floors,
                height_of_floors=height_of_floors,
                net_leased_area=net_leased_area,
            )
        else:
            raise ValueError(f"Unknown building type: {building_type}")

        # Calculate building parameters
        prj.calc_all_buildings()

        # Store project for later use
        self.projects[building_name] = prj

        # Extract RC values from the building
        building = prj.buildings[0]
        
        # Get thermal zone data
        thermal_zone = building.thermal_zones[0] if building.thermal_zones else None
        
        rc_values = {
            "building_name": building_name,
            "building_type": building_type,
            "insulation_quality": insulation_quality,
            "year_of_construction": year_of_construction,
            "net_leased_area": net_leased_area,
            "number_of_floors": number_of_floors,
            "height_of_floors": height_of_floors,
        }

        if thermal_zone:
            rc_values.update({
                "volume": thermal_zone.volume,
                "area": thermal_zone.area,
                "c1_value": getattr(thermal_zone, "c1", None),
                "c2_value": getattr(thermal_zone, "c2", None),
                "r1_value": getattr(thermal_zone, "r1", None),
                "r2_value": getattr(thermal_zone, "r2", None),
                "r3_value": getattr(thermal_zone, "r3", None),
            })

        logger.info(f"RC values calculated for {building_name}")
        return rc_values

    def calculate_multiple_buildings(
        self,
        building_configs: List[Dict[str, any]]
    ) -> pd.DataFrame:
        """
        Calculate RC values for multiple building configurations.

        Args:
            building_configs: List of building configuration dictionaries

        Returns:
            DataFrame with RC values for all buildings
        """
        all_rc_values = []

        for config in building_configs:
            try:
                rc_values = self.calculate_rc_values(
                    building_name=config["building_name"],
                    building_type=config["building_type"],
                    insulation_quality=config["insulation_quality"],
                    net_leased_area=config.get("net_leased_area", 1000.0),
                    number_of_floors=config.get("number_of_floors", 3),
                    height_of_floors=config.get("height_of_floors", 3.5),
                )
                all_rc_values.append(rc_values)
            except Exception as e:
                logger.error(f"Error calculating RC for {config['building_name']}: {e}")
                continue

        return pd.DataFrame(all_rc_values)

    def export_project(self, building_name: str, output_path: str):
        """
        Export TEASER project for a building.

        Args:
            building_name: Name of the building
            output_path: Path to export the project
        """
        if building_name not in self.projects:
            raise ValueError(f"Building {building_name} not found")

        prj = self.projects[building_name]
        prj.export_aixlib(
            building_model="MultizoneEquipped",
            zone_model="ThermalZoneEquipped",
            corG=True,
            internal_id=None,
            path=output_path,
        )
        logger.info(f"Exported project for {building_name} to {output_path}")
