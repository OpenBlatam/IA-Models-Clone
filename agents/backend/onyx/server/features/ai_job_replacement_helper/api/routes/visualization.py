"""
Visualization endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils.visualization_utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
)

router = APIRouter()


@router.post("/training-history")
async def plot_training_history_endpoint(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Graficar historial de entrenamiento"""
    try:
        image_bytes = plot_training_history(
            train_losses, val_losses, train_accs, val_accs
        )
        
        if image_bytes:
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return {
                "image": image_b64,
                "format": "png",
            }
        else:
            return {"error": "Could not generate plot"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confusion-matrix")
async def plot_confusion_matrix_endpoint(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Graficar matriz de confusión"""
    try:
        import numpy as np
        image_bytes = plot_confusion_matrix(
            np.array(y_true), np.array(y_pred), class_names
        )
        
        if image_bytes:
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return {
                "image": image_b64,
                "format": "png",
            }
        else:
            return {"error": "Could not generate plot"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-importance")
async def plot_feature_importance_endpoint(
    feature_names: List[str],
    importances: List[float],
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """Graficar importancia de features"""
    try:
        image_bytes = plot_feature_importance(
            feature_names, importances, top_k
        )
        
        if image_bytes:
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return {
                "image": image_b64,
                "format": "png",
            }
        else:
            return {"error": "Could not generate plot"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




