"""
Forest Fire Early Warning System - FastAPI Backend
====================================================
REST API for grid-based fire risk prediction using ML and real-time weather data.

Endpoints:
- GET  /api/regions          - List all available forest regions
- GET  /api/grids/{region}   - Get grid cells for a specific region
- POST /api/predict          - Predict fire risk for all grids in a region
- GET  /api/grid/{region}/{grid_id} - Get detailed explanation for a grid
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio

from grid import get_all_regions, get_region_info, generate_grids_for_region, FOREST_REGIONS
from weather import get_weather_for_grid
from model import get_model

# Initialize FastAPI app
app = FastAPI(
    title="Forest Fire Early Warning System API",
    description="Grid-based fire risk prediction using ML and real-time satellite/weather data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class PredictionRequest(BaseModel):
    region_id: str
    
class GridPrediction(BaseModel):
    grid_id: str
    row: int
    col: int
    center: Dict[str, float]
    bounds: Dict[str, float]
    risk_score: float
    risk_category: str
    weather: Dict[str, Any]
    
class RegionPredictionResponse(BaseModel):
    region_id: str
    region_name: str
    grids: List[GridPrediction]
    summary: Dict[str, Any]

# API Endpoints

@app.get("/")
async def root():
    """API health check and welcome message."""
    return {
        "status": "online",
        "message": "Forest Fire Early Warning System API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "regions": "/api/regions",
            "predict": "/api/predict"
        }
    }

@app.get("/api/regions")
async def list_regions():
    """
    List all available forest regions.
    
    Returns:
        List of regions with name, description, and grid count.
    """
    regions = get_all_regions()
    return {
        "success": True,
        "count": len(regions),
        "regions": regions
    }

@app.get("/api/grids/{region_id}")
async def get_grids(region_id: str):
    """
    Get all grid cells for a specific region.
    
    Args:
        region_id: Region identifier (amazon, california, australia, mediterranean)
    
    Returns:
        Region info with all 12 grid cells and their boundaries.
    """
    try:
        region_info = get_region_info(region_id)
        return {
            "success": True,
            "region": region_info
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/predict")
async def predict_fire_risk(request: PredictionRequest):
    """
    Predict fire risk for all grids in a region.
    
    This is the main prediction endpoint that:
    1. Gets all grid cells for the region
    2. Fetches current weather for each grid
    3. Runs ML model to predict fire risk
    4. Returns color-coded risk for map visualization
    
    Args:
        request: Contains region_id
    
    Returns:
        All grid predictions with risk scores and categories.
    """
    region_id = request.region_id
    
    if region_id not in FOREST_REGIONS:
        raise HTTPException(
            status_code=404, 
            detail=f"Unknown region: {region_id}. Available: {list(FOREST_REGIONS.keys())}"
        )
    
    # Get region and grids
    region = FOREST_REGIONS[region_id]
    grids = generate_grids_for_region(region_id)
    
    # Get ML model
    model = get_model()
    
    # Fetch weather and predict for each grid
    predictions = []
    
    async def predict_for_grid(grid):
        # Get weather data
        weather = await get_weather_for_grid(
            grid.center_lat, 
            grid.center_lng, 
            grid.id
        )
        
        # Run prediction
        prediction = model.predict(weather)
        
        return {
            "grid_id": grid.id,
            "row": grid.row,
            "col": grid.col,
            "center": {"lat": grid.center_lat, "lng": grid.center_lng},
            "bounds": grid.bounds,
            "area_km2": grid.area_km2,
            "risk_score": prediction["risk_score"],
            "risk_category": prediction["risk_category"],
            "probability": prediction["probability"],
            "weather": weather,
            "model_type": prediction["model_type"]
        }
    
    # Run predictions concurrently
    predictions = await asyncio.gather(*[predict_for_grid(g) for g in grids])
    
    # Calculate summary statistics
    risk_scores = [p["risk_score"] for p in predictions]
    categories = [p["risk_category"] for p in predictions]
    
    # Count data sources
    sources = [p["weather"].get("source", "unknown") for p in predictions]
    real_data_count = sources.count("openweather_api")
    simulated_count = sources.count("simulated")
    
    summary = {
        "total_grids": len(predictions),
        "average_risk": round(sum(risk_scores) / len(risk_scores), 1),
        "max_risk": round(max(risk_scores), 1),
        "min_risk": round(min(risk_scores), 1),
        "high_risk_count": categories.count("High"),
        "medium_risk_count": categories.count("Medium"),
        "low_risk_count": categories.count("Low"),
        "data_source": {
            "real": real_data_count,
            "simulated": simulated_count
        },
        "alert_level": "CRITICAL" if categories.count("High") >= 3 else (
            "WARNING" if categories.count("High") >= 1 else "NORMAL"
        )
    }
    
    return {
        "success": True,
        "region_id": region_id,
        "region_name": region.name,
        "grids": predictions,
        "summary": summary
    }

@app.get("/api/grid/{region_id}/{grid_id}")
async def get_grid_explanation(region_id: str, grid_id: str):
    """
    Get detailed explanation for a specific grid's fire risk.
    
    This endpoint provides:
    - Current weather conditions
    - Risk score and category
    - Contributing factors with descriptions
    - Feature importance breakdown
    
    Args:
        region_id: Region identifier
        grid_id: Grid cell identifier
    
    Returns:
        Detailed explanation suitable for display in UI panel.
    """
    if region_id not in FOREST_REGIONS:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region_id}")
    
    # Find the grid
    grids = generate_grids_for_region(region_id)
    grid = next((g for g in grids if g.id == grid_id), None)
    
    if not grid:
        raise HTTPException(status_code=404, detail=f"Grid not found: {grid_id}")
    
    # Get weather and prediction
    weather = await get_weather_for_grid(grid.center_lat, grid.center_lng, grid.id)
    model = get_model()
    prediction = model.predict(weather)
    explanation = model.explain_prediction(weather, prediction)
    
    return {
        "success": True,
        "grid": {
            "id": grid.id,
            "row": grid.row,
            "col": grid.col,
            "center": {"lat": grid.center_lat, "lng": grid.center_lng},
            "bounds": grid.bounds,
            "area_km2": grid.area_km2
        },
        "weather": weather,
        "prediction": prediction,
        "explanation": explanation
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model.is_loaded,
        "regions_available": len(FOREST_REGIONS)
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    error_msg = str(exc)
    stack_trace = traceback.format_exc()
    
    # Log to file
    with open("error.log", "a") as f:
        f.write(f"\n[{request.method} {request.url}]\n")
        f.write(f"Error: {error_msg}\n")
        f.write(stack_trace)
        f.write("-" * 50 + "\n")
        
    print(f"ERROR: {error_msg}")  # Still print to console
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": error_msg,
            "debug_info": stack_trace, # Send to frontend for inspection if needed
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
