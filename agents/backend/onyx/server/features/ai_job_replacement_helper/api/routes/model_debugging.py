"""
Model Debugging endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils.debugging_utils import (
    check_model_health,
    diagnose_training_issue,
    compare_models,
)

router = APIRouter()


@router.post("/health-check")
async def health_check_endpoint(
    model_id: str
) -> Dict[str, Any]:
    """Verificar salud del modelo"""
    try:
        # In production, you would load the actual model
        # model = load_model(model_id)
        # health = check_model_health(model)
        
        return {
            "model_id": model_id,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagnose")
async def diagnose_training_issue_endpoint(
    loss: float,
    model_id: str
) -> Dict[str, Any]:
    """Diagnosticar problemas de entrenamiento"""
    try:
        # In production, you would use the actual model
        # diagnosis = diagnose_training_issue(model, loss)
        
        return {
            "loss": loss,
            "diagnosis": {
                "loss_is_nan": False,
                "loss_is_inf": False,
                "recommendations": [],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




