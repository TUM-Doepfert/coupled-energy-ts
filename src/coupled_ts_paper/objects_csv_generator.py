"""Generator for objects.csv file to be used with ENTISE."""

import logging
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectsCSVGenerator:
    """Generator for creating objects.csv file for ENTISE framework."""

    def __init__(self):
        """Initialize the ObjectsCSVGenerator."""
        self.objects_data = []

    def add_building(
        self,
        object_id: str,
        location: str,
        building_type: str,
        insulation_quality: str,
        rc_values: Dict[str, any],
        additional_params: Optional[Dict[str, any]] = None,
    ):
        """
        Add a building to the objects collection.

        Args:
            object_id: Unique identifier for the building object
            location: Location name
            building_type: Type of building
            insulation_quality: Quality of insulation
            rc_values: RC values from TEASER calculation
            additional_params: Additional parameters for the object
        """
        obj_data = {
            "object_id": object_id,
            "location": location,
            "building_type": building_type,
            "insulation_quality": insulation_quality,
            "year_of_construction": rc_values.get("year_of_construction"),
            "net_leased_area": rc_values.get("net_leased_area"),
            "number_of_floors": rc_values.get("number_of_floors"),
            "height_of_floors": rc_values.get("height_of_floors"),
            "volume": rc_values.get("volume"),
            "area": rc_values.get("area"),
            "c1": rc_values.get("c1_value"),
            "c2": rc_values.get("c2_value"),
            "r1": rc_values.get("r1_value"),
            "r2": rc_values.get("r2_value"),
            "r3": rc_values.get("r3_value"),
        }

        if additional_params:
            obj_data.update(additional_params)

        self.objects_data.append(obj_data)
        logger.info(f"Added building object: {object_id}")

    def add_multiple_buildings(
        self,
        locations: List[str],
        building_types: List[str],
        insulation_qualities: List[str],
        rc_values_df: pd.DataFrame,
    ):
        """
        Add multiple buildings based on combinations of parameters.

        Args:
            locations: List of location names
            building_types: List of building types
            insulation_qualities: List of insulation qualities
            rc_values_df: DataFrame with RC values
        """
        for location in locations:
            for building_type in building_types:
                for insulation_quality in insulation_qualities:
                    object_id = (
                        f"{location}_{building_type}_{insulation_quality}"
                    )
                    
                    # Find corresponding RC values
                    rc_row = rc_values_df[
                        (rc_values_df["building_type"] == building_type) &
                        (rc_values_df["insulation_quality"] == insulation_quality)
                    ]
                    
                    if not rc_row.empty:
                        rc_values = rc_row.iloc[0].to_dict()
                        self.add_building(
                            object_id=object_id,
                            location=location,
                            building_type=building_type,
                            insulation_quality=insulation_quality,
                            rc_values=rc_values,
                        )
                    else:
                        logger.warning(
                            f"No RC values found for {building_type}, "
                            f"{insulation_quality}"
                        )

    def generate_csv(self, output_path: str):
        """
        Generate and save the objects.csv file.

        Args:
            output_path: Path where to save the objects.csv file
        """
        if not self.objects_data:
            logger.warning("No objects data to save")
            return

        df = pd.DataFrame(self.objects_data)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved objects.csv with {len(df)} objects to {output_path}")
        
        return df

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the objects data as a DataFrame.

        Returns:
            DataFrame with all objects data
        """
        return pd.DataFrame(self.objects_data)
