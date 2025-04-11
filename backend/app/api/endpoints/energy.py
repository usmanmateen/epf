from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from app.core.supabase_client import supabase

router = APIRouter()

class EnergyPriceData(BaseModel):
    timestamp: datetime
    price: float
    volatility: float

@router.post("/store-energy-price/")
async def store_energy_price(data: EnergyPriceData):
    response = supabase.table("energy_prices").insert({
        "timestamp": data.timestamp.isoformat(),
        "price": data.price,
        "volatility": data.volatility
    }).execute()
    
    return {"message": "Energy Price Stored", "response": response}
