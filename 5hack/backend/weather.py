"""
Real-Time Weather Data Fetching
================================
Fetches current weather data from OpenWeather API for each grid cell.
Includes fallback to simulated data if API is unavailable.
"""

import os
import random
import httpx
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Typical weather ranges for forest fire conditions
WEATHER_RANGES = {
    "temperature": {"min": 20, "max": 45},  # Celsius
    "humidity": {"min": 20, "max": 90},     # Percentage
    "wind_speed": {"min": 0, "max": 30},    # km/h
    "rainfall": {"min": 0, "max": 15}       # mm (last 24h)
}

# Fire Weather Index simulation ranges
FWI_RANGES = {
    "ffmc": {"min": 40, "max": 96},   # Fine Fuel Moisture Code
    "dmc": {"min": 1, "max": 30},     # Duff Moisture Code
    "dc": {"min": 5, "max": 400},     # Drought Code
    "isi": {"min": 0, "max": 15},     # Initial Spread Index
    "bui": {"min": 1, "max": 40},     # Build-Up Index
    "fwi": {"min": 0, "max": 25}      # Fire Weather Index
}

async def fetch_weather_from_api(lat: float, lng: float) -> Optional[Dict[str, Any]]:
    """Fetch real weather data from OpenWeather API."""
    if not OPENWEATHER_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPENWEATHER_BASE_URL,
                params={
                    "lat": lat,
                    "lon": lng,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"] * 3.6,  # m/s to km/h
                    "rainfall": data.get("rain", {}).get("1h", 0) or 0,
                    "source": "openweather_api",
                    "weather_description": data["weather"][0]["description"] if data.get("weather") else "Unknown"
                }
    except Exception as e:
        print(f"❌ Weather API error: {e}")
    
    # If we get here, something failed
    print("⚠️ Falling back to simulated weather data")
    return None

def generate_simulated_weather(lat: float, lng: float, grid_id: str) -> Dict[str, Any]:
    """
    Generate realistic simulated weather data based on location.
    Uses grid_id as seed for reproducibility within a session.
    """
    # Create deterministic seed from grid_id
    seed = hash(grid_id) % 10000
    random.seed(seed)
    
    # Base temperature varies by latitude (higher temps near equator)
    lat_factor = 1 - (abs(lat) / 90)  # 0-1 scale
    base_temp = 20 + (lat_factor * 20)  # 20-40°C range
    
    # Add some variation
    temperature = base_temp + random.uniform(-5, 10)
    temperature = max(15, min(45, temperature))
    
    # Humidity inversely related to temperature with noise
    humidity = 90 - (temperature - 20) * 2 + random.uniform(-10, 10)
    humidity = max(20, min(95, humidity))
    
    # Wind and rain
    wind_speed = random.uniform(5, 25)
    rainfall = random.uniform(0, 5) if humidity > 60 else 0
    
    return {
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "wind_speed": round(wind_speed, 1),
        "rainfall": round(rainfall, 1),
        "source": "simulated",
        "weather_description": "Simulated conditions"
    }

def generate_fire_weather_indices(weather: Dict[str, Any]) -> Dict[str, float]:
    """
    Generate Fire Weather Index components based on weather conditions.
    These are simplified approximations for demonstration.
    
    Real FWI calculation requires previous day's values and complex formulas.
    This simplified version creates plausible values based on current weather.
    """
    temp = weather["temperature"]
    humidity = weather["humidity"]
    wind = weather["wind_speed"]
    rain = weather["rainfall"]
    
    # FFMC: Fine Fuel Moisture Code (surface litter moisture)
    # Higher temp + lower humidity = higher FFMC (drier fuels)
    ffmc_base = 60 + (temp - 25) * 1.5 - (humidity - 50) * 0.3
    ffmc = max(40, min(96, ffmc_base - rain * 3))
    
    # DMC: Duff Moisture Code (medium depth organic layer)
    dmc = max(1, min(30, 10 + (temp - 25) * 0.5 - humidity * 0.1))
    
    # DC: Drought Code (deep organic layer moisture)
    dc = max(5, min(400, 150 + (temp - 25) * 5 - rain * 10))
    
    # ISI: Initial Spread Index (fire spread rate)
    isi = max(0, min(15, (ffmc - 60) * 0.15 + wind * 0.2))
    
    # BUI: Build-Up Index (fuel available for combustion)
    bui = max(1, min(40, dmc * 0.8 + dc * 0.02))
    
    # FWI: Fire Weather Index (overall fire intensity)
    fwi = max(0, min(25, (isi * bui) ** 0.5))
    
    return {
        "ffmc": round(ffmc, 1),
        "dmc": round(dmc, 1),
        "dc": round(dc, 1),
        "isi": round(isi, 1),
        "bui": round(bui, 1),
        "fwi": round(fwi, 1)
    }

async def get_weather_for_grid(lat: float, lng: float, grid_id: str) -> Dict[str, Any]:
    """
    Get complete weather data for a grid cell.
    Tries real API first, falls back to simulation if unavailable.
    """
    # Try real API first
    weather = await fetch_weather_from_api(lat, lng)
    
    # Fall back to simulation if API fails
    if weather is None:
        weather = generate_simulated_weather(lat, lng, grid_id)
    
    # Generate fire weather indices
    fwi_indices = generate_fire_weather_indices(weather)
    
    return {
        **weather,
        **fwi_indices,
        "lat": lat,
        "lng": lng,
        "grid_id": grid_id
    }

def get_weather_sync(lat: float, lng: float, grid_id: str) -> Dict[str, Any]:
    """Synchronous version for non-async contexts."""
    weather = generate_simulated_weather(lat, lng, grid_id)
    fwi_indices = generate_fire_weather_indices(weather)
    
    return {
        **weather,
        **fwi_indices,
        "lat": lat,
        "lng": lng,
        "grid_id": grid_id
    }
