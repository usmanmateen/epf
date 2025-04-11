from fastapi import APIRouter
from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.get("/insights")
async def get_model_insights():
    return {
        "metrics": await prediction_service.get_model_metrics(),
        "predictions": await prediction_service.get_recent_predictions()
    } 