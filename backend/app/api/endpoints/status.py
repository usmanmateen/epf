from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/status")
async def get_status():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }
