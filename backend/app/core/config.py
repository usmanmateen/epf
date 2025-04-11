from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Energy Price Prediction"
    
    # Weather API Configuration
    WEATHER_API_KEY: str
    WEATHER_API_URL: str = "https://api.weatherapi.com/v1"
    
    # ML Model Configuration
    MODEL_PATH: str = "models/energy_price_model.pkl"
    
    class Config:
        env_file = ".env"

settings = Settings()
