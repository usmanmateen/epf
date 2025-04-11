import requests
import os
import xml.etree.ElementTree as ET
from fastapi import HTTPException

# Define API URLs
UK_GRID_API = "https://api.carbonintensity.org.uk/intensity"
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "")
ENTSOE_API_URL = "https://transparency.entsoe.eu/api"

# Function to fetch UK grid demand
def get_uk_grid_demand():
    try:
        response = requests.get(UK_GRID_API, timeout=10)  # Add timeout for reliability
        response.raise_for_status()  # Raises exception for 4xx/5xx errors

        data = response.json()
        intensity = data["data"][0]["intensity"]

        return {
            "carbon_intensity": intensity["actual"],  # Actual carbon intensity (gCO2/kWh)
            "forecast": intensity["forecast"],  # Forecasted intensity
            "status": intensity["index"]  # Moderate, low, high, etc.
        }
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"UK Grid API Error: {str(e)}")

# Function to fetch ENTSO-E European grid demand
def get_entsoe_grid_demand():
    try:
        params = {
            "securityToken": ENTSOE_API_KEY,
            "documentType": "A65"  # Total Load data type
        }
        response = requests.get(ENTSOE_API_URL, params=params, timeout=10)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        # Extract relevant data (Example: Get latest grid demand in MW)
        data_values = root.findall(".//TimeSeries/Period/quantity")
        timestamps = root.findall(".//TimeSeries/Period/start")

        if not data_values or not timestamps:
            raise HTTPException(status_code=500, detail="ENTSO-E API returned empty response")

        demand_data = [
            {
                "timestamp": timestamps[i].text,
                "grid_demand_mw": data_values[i].text
            }
            for i in range(len(data_values))
        ]

        return {
            "source": "ENTSO-E",
            "latest_demand": demand_data[-1] if demand_data else {},
            "historical_demand": demand_data
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"ENTSO-E API Error: {str(e)}")

    except ET.ParseError:
        raise HTTPException(status_code=500, detail="Failed to parse ENTSO-E XML response")
