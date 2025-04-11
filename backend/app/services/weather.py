from fastapi import FastAPI, APIRouter
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from app.core.supabase_client import supabase
import asyncio


app = FastAPI()
router = APIRouter()
load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
SOLAR_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type/wind-and-solar"

def calculate_solar_generation(cloud_cover, panel_area=100, panel_efficiency=0.2):
    """
    Estimate solar generation based on cloud cover.
    """
    max_radiation = 1000  # W/m² on a clear day
    actual_radiation = max_radiation * (1 - cloud_cover / 100)
    power_output = actual_radiation * panel_area * panel_efficiency
    solar_kw = power_output / 1000  # Convert to kW
    print(f"[DEBUG] Calculated Solar Generation: {solar_kw:.2f} kW (Cloud Cover: {cloud_cover}%)")
    return solar_kw

@app.get("/weather/uk")
def get_uk_weather():
    """
    Fetch weather data for the UK and calculate solar generation based on cloud cover.
    """
    params = {
        "lat": 54.5,  # UK center latitude
        "lon": -2.0,  # UK center longitude
        "appid": API_KEY,
        "units": "metric"
    }
    print("[DEBUG] Fetching weather data with parameters:", params)

    response = requests.get(WEATHER_URL, params=params)
    print(f"[DEBUG] Weather API Response Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        cloud_cover = data["clouds"]["all"]
        solar_generation = calculate_solar_generation(cloud_cover)

        result = {
            "temperature": data["main"]["temp"],
            "wind_speed": data["wind"]["speed"],
            "cloud_cover": cloud_cover,
            "humidity": data["main"]["humidity"],
            "precipitation": data.get("rain", {}).get("1h", 0),
            "solar_generation": solar_generation
        }

        print("[DEBUG] Weather Data Fetched Successfully:", result)
        return result
    else:
        error_message = {"error": f"Failed to fetch weather data: {response.status_code}"}
        print("[ERROR]", error_message)
        return error_message

def fetch_solar_generation(settlement_date=None):
    """
    Fetch actual solar generation data from the UK grid.
    """
    if not settlement_date:
        settlement_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print("[DEBUG] Using settlement date:", settlement_date)

    params = {
        "from": settlement_date,
        "to": settlement_date,
        "settlementPeriodFrom": 3,
        "settlementPeriodTo": 47,
        "format": "json",
    }
    try:
        response = requests.get(SOLAR_URL, params=params, timeout=15)
        print(f"[DEBUG] Solar API Response Code: {response.status_code}")

        if response.status_code != 200:
            print("[ERROR] Failed to fetch solar data:", response.text)
            return {"error": f"API request failed with status {response.status_code}: {response.text}"}

        data = response.json()
        generation_data = data.get("data", [])

        solar_data = [entry for entry in generation_data if entry.get("businessType") == "Solar generation"]

        total_solar_generation = sum(entry["quantity"] for entry in solar_data)

        return int(total_solar_generation)

    except requests.exceptions.RequestException as e:
        print("[ERROR] Request failed:", e)
        return {"error": f"Request failed: {e}"}

@router.post("/store-weather-data/")
async def store_weather_data():
    weather = get_uk_weather()
    solar = fetch_solar_generation()
    # ✅ Check if weather data contains all required fields
    if not weather or "temperature" not in weather:
        print("[ERROR] Weather API returned invalid data:", weather)
        return {"error": "Weather data fetch failed. Not storing in Supabase."}

    timestamp = datetime.now(timezone.utc).isoformat()

    print("[DEBUG] Storing data:", {
        "timestamp": timestamp,
        "temperature": weather["temperature"],
        "wind_speed": weather["wind_speed"],
        "humidity": weather["humidity"],
        "cloud_coverage": weather["cloud_cover"],
        "solar_generation": weather["solar_generation"]
    })

    response = supabase.table("weather_solar_data").insert({
        "timestamp": timestamp,
        "temperature": weather["temperature"],
        "wind_speed": weather["wind_speed"],
        "humidity": weather["humidity"],
        "cloud_coverage": weather["cloud_cover"],
        "solar_generation": solar
    }).execute()

    print("[DEBUG] Supabase Response:", response)
    return {"message": "Weather Data Stored", "response": response}


async def schedule_weather_updates(interval_seconds=900):  # 900 seconds = 15 minutes
    """Run `store_weather_data()` automatically every 15 minutes."""
    while True:
        print("[AUTO] Fetching and storing weather + solar data...")
        await store_weather_data()
        await asyncio.sleep(interval_seconds)


@app.get("/solar")
def get_solar_generation(settlement_date: str = None):
    """
    Get real-time solar generation from the UK grid.
    """
    solar_data = fetch_solar_generation(settlement_date)
    return solar_data
