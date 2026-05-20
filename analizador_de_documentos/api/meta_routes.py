"""
Rutas para Meta-Learning
==========================

Endpoints para meta-learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.meta_learning import (
    get_meta_learning,
    MetaLearningSystem,
    MetaLearningMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/meta-learning",
    tags=["Meta-Learning"]
)


class CreateMetaTaskRequest(BaseModel):
    """Request para crear tarea meta"""
    task_type: str = Field(..., description="Tipo de tarea")
    support_set: List[Dict[str, Any]] = Field(..., description="Conjunto de soporte")
    query_set: List[Dict[str, Any]] = Field(..., description="Conjunto de consulta")
    method: str = Field("maml", description="Método de meta-learning")


@router.post("/tasks")
async def create_meta_task(
    request: CreateMetaTaskRequest,
    system: MetaLearningSystem = Depends(get_meta_learning)
):
    """Crear tarea meta"""
    try:
        method = MetaLearningMethod(request.method)
        meta_task = system.create_meta_task(
            request.task_type,
            request.support_set,
            request.query_set,
            method
        )
        
        return {
            "meta_task_id": meta_task.meta_task_id,
            "task_type": meta_task.task_type,
            "method": meta_task.method.value,
            "support_samples": len(meta_task.support_set),
            "query_samples": len(meta_task.query_set)
        }
    except Exception as e:
        logger.error(f"Error creando tarea meta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{meta_task_id}/meta-train")
async def meta_train(
    meta_task_id: str,
    model_id: str = Field(None, description="ID del modelo"),
    meta_epochs: int = Field(10, description="Número de épocas meta"),
    system: MetaLearningSystem = Depends(get_meta_learning)
):
    """Entrenar meta-modelo"""
    try:
        result = system.meta_train(meta_task_id, model_id, meta_epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en meta-entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{meta_task_id}/fast-adapt")
async def fast_adapt(
    meta_task_id: str,
    new_task_data: List[Dict[str, Any]] = Field(..., description="Datos de nueva tarea"),
    adaptation_steps: int = Field(5, description="Pasos de adaptación"),
    system: MetaLearningSystem = Depends(get_meta_learning)
):
    """Adaptación rápida"""
    try:
        result = system.fast_adapt(meta_task_id, new_task_data, adaptation_steps)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en adaptación: {e}")
        raise HTTPException(status_code=500, detail=str(e))



