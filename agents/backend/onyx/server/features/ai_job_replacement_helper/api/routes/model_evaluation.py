"""
Model Evaluation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_evaluation import ModelEvaluationService

router = APIRouter()
evaluation_service = ModelEvaluationService()


@router.post("/classification")
async def evaluate_classification(
    y_true: List[float],
    y_pred: List[float],
    y_proba: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """Evaluar modelo de clasificación"""
    try:
        import numpy as np
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_proba_arr = np.array(y_proba) if y_proba else None
        
        metrics = evaluation_service.evaluate_classification(
            y_true_arr, y_pred_arr, y_proba_arr
        )
        
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "roc_auc": metrics.roc_auc,
            "confusion_matrix": metrics.confusion_matrix.tolist() if metrics.confusion_matrix is not None else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regression")
async def evaluate_regression(
    y_true: List[float],
    y_pred: List[float]
) -> Dict[str, Any]:
    """Evaluar modelo de regresión"""
    try:
        import numpy as np
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        metrics = evaluation_service.evaluate_regression(y_true_arr, y_pred_arr)
        
        return {
            "mse": metrics.mse,
            "mae": metrics.mae,
            "r2_score": metrics.r2_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

