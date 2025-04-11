from fastapi import APIRouter, HTTPException
from app.models.prediction import PredictionInput, PredictionOutput
from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionOutput)
async def predict_price(input_data: PredictionInput):
    try:
        prediction = await prediction_service.predict(input_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
