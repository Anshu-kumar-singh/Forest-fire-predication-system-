"""
Grid-Based Spatial Logic
=========================
Defines forest regions and divides them into 12 grid cells each.
Each grid has geographic boundaries for localized fire risk prediction.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class GridCell:
    """Represents a single grid cell within a forest region."""
    id: str
    region: str
    row: int
    col: int
    center_lat: float
    center_lng: float
    bounds: Dict[str, float]  # north, south, east, west
    area_km2: float

@dataclass
class ForestRegion:
    """Represents a forest region with geographic boundaries."""
    id: str
    name: str
    description: str
    center_lat: float
    center_lng: float
    bounds: Dict[str, float]
    grid_rows: int = 3
    grid_cols: int = 4

# Define 4 major forest regions for demonstration
FOREST_REGIONS = {
    "amazon": ForestRegion(
        id="amazon",
        name="Amazon Rainforest",
        description="Brazil - World's largest tropical rainforest",
        center_lat=-3.4653,
        center_lng=-62.2159,
        bounds={"north": -2.0, "south": -5.0, "east": -60.0, "west": -65.0}
    ),
    "california": ForestRegion(
        id="california",
        name="California Forests",
        description="USA - Sierra Nevada and coastal forests",
        center_lat=37.5,
        center_lng=-119.5,
        bounds={"north": 39.0, "south": 36.0, "east": -118.0, "west": -121.0}
    ),
    "australia": ForestRegion(
        id="australia",
        name="Australian Bushland",
        description="Australia - Eastern forest regions",
        center_lat=-33.5,
        center_lng=150.5,
        bounds={"north": -32.0, "south": -35.0, "east": 152.0, "west": 149.0}
    ),
    "mediterranean": ForestRegion(
        id="mediterranean",
        name="Mediterranean Forests",
        description="Southern Europe - Portugal, Spain, Greece",
        center_lat=38.5,
        center_lng=-8.0,
        bounds={"north": 40.0, "south": 37.0, "east": -6.0, "west": -10.0}
    )
}

def calculate_grid_area(bounds: Dict[str, float]) -> float:
    """Calculate approximate area in km² for a grid cell."""
    lat_diff = abs(bounds["north"] - bounds["south"])
    lng_diff = abs(bounds["east"] - bounds["west"])
    # Approximate: 1 degree latitude ≈ 111 km, longitude varies with latitude
    avg_lat = (bounds["north"] + bounds["south"]) / 2
    km_per_lng = 111 * math.cos(math.radians(avg_lat))
    return lat_diff * 111 * lng_diff * km_per_lng

def generate_grids_for_region(region_id: str) -> List[GridCell]:
    """
    Divide a forest region into a 4x3 grid (12 cells).
    
    Grid Layout:
    ┌─────┬─────┬─────┬─────┐
    │ 0,0 │ 0,1 │ 0,2 │ 0,3 │  Row 0
    ├─────┼─────┼─────┼─────┤
    │ 1,0 │ 1,1 │ 1,2 │ 1,3 │  Row 1
    ├─────┼─────┼─────┼─────┤
    │ 2,0 │ 2,1 │ 2,2 │ 2,3 │  Row 2
    └─────┴─────┴─────┴─────┘
    """
    if region_id not in FOREST_REGIONS:
        raise ValueError(f"Unknown region: {region_id}")
    
    region = FOREST_REGIONS[region_id]
    grids = []
    
    lat_step = (region.bounds["north"] - region.bounds["south"]) / region.grid_rows
    lng_step = (region.bounds["east"] - region.bounds["west"]) / region.grid_cols
    
    for row in range(region.grid_rows):
        for col in range(region.grid_cols):
            grid_id = f"{region_id}_grid_{row}_{col}"
            
            # Calculate bounds for this cell
            north = region.bounds["north"] - (row * lat_step)
            south = north - lat_step
            west = region.bounds["west"] + (col * lng_step)
            east = west + lng_step
            
            cell_bounds = {
                "north": north,
                "south": south,
                "east": east,
                "west": west
            }
            
            # Center point
            center_lat = (north + south) / 2
            center_lng = (east + west) / 2
            
            # Calculate area
            area = calculate_grid_area(cell_bounds)
            
            grids.append(GridCell(
                id=grid_id,
                region=region_id,
                row=row,
                col=col,
                center_lat=center_lat,
                center_lng=center_lng,
                bounds=cell_bounds,
                area_km2=round(area, 2)
            ))
    
    return grids

def get_all_regions() -> List[Dict[str, Any]]:
    """Get list of all available forest regions."""
    return [
        {
            "id": region.id,
            "name": region.name,
            "description": region.description,
            "center": {"lat": region.center_lat, "lng": region.center_lng},
            "bounds": region.bounds,
            "grid_count": region.grid_rows * region.grid_cols
        }
        for region in FOREST_REGIONS.values()
    ]

def get_region_info(region_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific region."""
    if region_id not in FOREST_REGIONS:
        raise ValueError(f"Unknown region: {region_id}")
    
    region = FOREST_REGIONS[region_id]
    grids = generate_grids_for_region(region_id)
    
    return {
        "id": region.id,
        "name": region.name,
        "description": region.description,
        "center": {"lat": region.center_lat, "lng": region.center_lng},
        "bounds": region.bounds,
        "grids": [
            {
                "id": g.id,
                "row": g.row,
                "col": g.col,
                "center": {"lat": g.center_lat, "lng": g.center_lng},
                "bounds": g.bounds,
                "area_km2": g.area_km2
            }
            for g in grids
        ]
    }

def grid_cell_to_dict(grid: GridCell) -> Dict[str, Any]:
    """Convert GridCell to dictionary for JSON serialization."""
    return {
        "id": grid.id,
        "region": grid.region,
        "row": grid.row,
        "col": grid.col,
        "center": {"lat": grid.center_lat, "lng": grid.center_lng},
        "bounds": grid.bounds,
        "area_km2": grid.area_km2
    }
