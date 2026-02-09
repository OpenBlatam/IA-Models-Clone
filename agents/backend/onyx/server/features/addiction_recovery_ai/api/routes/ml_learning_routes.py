"""
ML learning routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict

try:
    from services.ml_learning_service import MLLearningService
except ImportError:
    from ...services.ml_learning_service import MLLearningService

router = APIRouter()

ml_learning = MLLearningService()


@router.post("/ml/train-model")
async def train_personalized_model(
    user_id: str = Body(...),
    training_data: List[Dict] = Body(...),
    model_type: str = Body("relapse_prediction")
):
    """Entrena modelo personalizado para usuario"""
    try:
        model = ml_learning.train_personalized_model(user_id, training_data, model_type)
        return JSONResponse(content=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")



