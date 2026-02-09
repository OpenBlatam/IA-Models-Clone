"""
Rutas para Automated Feature Engineering Advanced
==================================================

Endpoints para feature engineering avanzado.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.automated_feature_engineering_advanced import (
    get_automated_feature_engineering_advanced,
    AutomatedFeatureEngineeringAdvanced,
    FeatureTransformation
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/automated-feature-engineering-advanced",
    tags=["Automated Feature Engineering Advanced"]
)


class CreateTaskRequest(BaseModel):
    """Request para crear tarea"""
    input_features: List[str] = Field(..., description="Features de entrada")


@router.post("/tasks")
async def create_task(
    request: CreateTaskRequest,
    system: AutomatedFeatureEngineeringAdvanced = Depends(get_automated_feature_engineering_advanced)
):
    """Crear tarea de feature engineering"""
    try:
        task = system.create_task(request.input_features)
        
        return {
            "task_id": task.task_id,
            "input_features": task.input_features,
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/generate")
async def generate_features(
    task_id: str,
    transformations: Optional[List[str]] = Field(None, description="Transformaciones"),
    system: AutomatedFeatureEngineeringAdvanced = Depends(get_automated_feature_engineering_advanced)
):
    """Generar features"""
    try:
        trans_list = None
        if transformations:
            trans_list = [FeatureTransformation(t) for t in transformations]
        
        generated = system.generate_features(task_id, trans_list)
        
        return {
            "task_id": task_id,
            "generated_features": [
                {
                    "feature_id": f.feature_id,
                    "name": f.name,
                    "transformation": f.transformation.value,
                    "importance": f.importance
                }
                for f in generated
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generando features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/select-best")
async def select_best_features(
    task_id: str,
    top_k: int = Field(10, description="Top K"),
    system: AutomatedFeatureEngineeringAdvanced = Depends(get_automated_feature_engineering_advanced)
):
    """Seleccionar mejores features"""
    try:
        best = system.select_best_features(task_id, top_k)
        
        return {
            "task_id": task_id,
            "best_features": [
                {
                    "feature_id": f.feature_id,
                    "name": f.name,
                    "importance": f.importance
                }
                for f in best
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error seleccionando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


