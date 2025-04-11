from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import requests
import os
from app.services import weather  # Ensure this file exists
import asyncio
from app.services.weather import router as weather_router, schedule_weather_updates 
from fastapi.responses import JSONResponse

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Energy Price Predictor API")

# CORS Settings
origins = [
    "http://localhost:5173",  # Local development
    "http://frontend:5173",   # Docker frontend
    "http://172.20.0.3:5173"  # Docker frontend IP
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# OPTIONS handler for preflight requests
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Handle CORS preflight OPTIONS requests."""
    headers = {
        "Access-Control-Allow-Origin": ", ".join(origins),
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE", 
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    return JSONResponse(status_code=200, content="Preflight OK", headers=headers)

# API Key Model
class ApiKeyModel(BaseModel):
    name: str
    key: str




app.include_router(weather_router)



@app.on_event("startup")
async def start_background_tasks():
    """Starts the automatic 15-minute weather data collection"""
    asyncio.create_task(schedule_weather_updates())


# ✅ Weather Route (Unchanged)
@app.get("/weather/uk")
def get_weather():
    return weather.get_uk_weather()

# ✅ Solar Route (Unchanged)
@app.get("/solar")
def get_solar():
    return weather.fetch_solar_generation()

# ✅ Home Route (Unchanged)
@app.get("/")
def home():
    return {"message": "Energy Price Prediction API is running"}

# ✅ Save API Keys to .env
@app.post("/api/save-api-keys")
async def save_api_keys(api_keys: list[ApiKeyModel]):
    try:
        with open(".env", "w") as env_file:
            for api_key in api_keys:
                env_file.write(f"{api_key.name.upper().replace(' ', '_')}_KEY={api_key.key}\n")
        return {"message": "API keys saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Get API Keys from .env
@app.get("/api/get-api-keys")
async def get_api_keys():
    try:
        keys = [
            {"name": "Weather API", "key": os.getenv("WEATHER_API_KEY", ""), "status": "unchecked"},
            {"name": "News API", "key": os.getenv("NEWS_API_KEY", ""), "status": "unchecked"},
            {"name": "Energy Prices API", "key": os.getenv("ENERGY_PRICES_API_KEY", ""), "status": "unchecked"},
        ]
        return keys
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-api-key")
async def test_api_key(name: str = Query(..., description="API Service Name"), key: str = Query(..., description="User API Key")):
    """
    Test API key by making a request to the corresponding service before saving.
    """
    try:
        # ✅ Define API endpoints for testing
        api_endpoints = {
            "Weather API": f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={key}",
            "News API": f"https://newsapi.org/v2/top-headlines?country=us&apiKey={key}",
            "Energy Prices API": f"https://api.example-energy.com/v1/prices?apiKey={key}"
        }

        print(f"[DEBUG] Testing API: {name} with key: {key}")                              

        # ✅ Check if API name is valid
        if name not in api_endpoints:
            return {"status": "invalid", "detail": "Invalid API name"}

        # ✅ Make a request to test API key
        response = requests.get(api_endpoints[name])

        print(f"[DEBUG] Response Status: {response.status_code}")

        if response.status_code == 200:
            return {"status": "valid"}

        # ✅ Handle API response errors properly
        try:
            error_detail = response.json()
        except:
            error_detail = {"error": "No JSON response"}

        return {"status": "invalid", "detail": error_detail}

    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


xgb_model = joblib.load("app/models/xgboost_model.pkl")
lstm_model = load_model("app/models/lstm_model.keras")
scaler = joblib.load("app/models/scaler.pkl")

# ✅ Request Model
class PredictionRequest(BaseModel):
    features: list[float]

# ✅ Prediction Endpoint
@app.post("/predict")
async def predict_energy_price(request: PredictionRequest):
    try:
        X_input = np.array(request.features).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # XGBoost Prediction
        xgb_pred = xgb_model.predict(X_scaled)[0]

        # LSTM Prediction
        lstm_pred = lstm_model.predict(np.expand_dims(X_scaled, axis=0))[0][0]

        return {
            "xgboost_prediction": round(float(xgb_pred), 4),
            "lstm_prediction": round(float(lstm_pred), 4),
            "confidence": random.randint(85, 95)  # Simulating confidence score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Real-Time Stats Endpoint
@app.get("/stats")
async def get_current_stats():
    """
    Simulates real-time energy price and external factors.
    This should be connected to a live API in production.
    """
    return {
        "current_price": "£0.178/kWh",
        "price_change": "+2.3%",
        "prediction_accuracy": "91%",
        "external_factors": {
            "solar_generation": f"{random.randint(0, 50)} MW",
            "wind_speed": f"{round(random.uniform(1.5, 15.0), 2)} mph",
            "cloud_cover": f"{random.randint(5, 80)}%",
            "precipitation": f"{random.randint(0, 20)}%"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print(asyncio.run(get_api_keys()))  
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
