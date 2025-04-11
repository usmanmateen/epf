from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionInput(BaseModel):
    datetime: datetime
    location: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None

class PredictionOutput(BaseModel):
    predicted_price: float
    timestamp: datetime
    confidence: float
