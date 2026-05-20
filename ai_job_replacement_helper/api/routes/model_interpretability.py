"""
Model Interpretability endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_interpretability import ModelInterpretabilityService

router = APIRouter()
interpretability_service = ModelInterpretabilityService()


@router.post("/explain")
async def explain_prediction(
    instance: List[float],
    method: str = "shap"
) -> Dict[str, Any]:
    """Explicar predicción"""
    try:
        import numpy as np
        instance_arr = np.array([instance])
        
        # In production, you would pass the actual model
        # explanation = interpretability_service.explain_prediction_shap(model, instance_arr)
        
        return {
            "method": method,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-importance")
async def get_feature_importance(
    feature_names: List[str],
    method: str = "permutation"
) -> Dict[str, Any]:
    """Obtener importancia de features"""
    try:
        return {
            "method": method,
            "feature_names": feature_names,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




