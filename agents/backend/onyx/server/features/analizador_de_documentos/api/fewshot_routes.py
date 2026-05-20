"""
Rutas para Few-Shot Learning
==============================

Endpoints para few-shot learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.few_shot_learning import (
    get_few_shot_learning,
    FewShotLearning,
    FewShotMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/few-shot",
    tags=["Few-Shot Learning"]
)


class CreateFewShotTaskRequest(BaseModel):
    """Request para crear tarea few-shot"""
    task_name: str = Field(..., description="Nombre de la tarea")
    examples: List[Dict[str, Any]] = Field(..., description="Pocos ejemplos")
    method: str = Field("prompt_based", description="Método")


@router.post("/tasks")
async def create_few_shot_task(
    request: CreateFewShotTaskRequest,
    system: FewShotLearning = Depends(get_few_shot_learning)
):
    """Crear tarea few-shot"""
    try:
        method = FewShotMethod(request.method)
        task = system.create_few_shot_task(
            request.task_name,
            request.examples,
            method
        )
        
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "num_examples": len(task.examples),
            "method": task.method.value,
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/learn")
async def learn_from_few_examples(
    task_id: str,
    model_id: str = Field(None, description="ID del modelo"),
    system: FewShotLearning = Depends(get_few_shot_learning)
):
    """Aprender de pocos ejemplos"""
    try:
        result = system.learn_from_few_examples(task_id, model_id)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error aprendiendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/evaluate")
async def evaluate_few_shot(
    task_id: str,
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba"),
    system: FewShotLearning = Depends(get_few_shot_learning)
):
    """Evaluar rendimiento few-shot"""
    try:
        performance = system.evaluate_few_shot_performance(task_id, test_data)
        
        return performance
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando: {e}")
        raise HTTPException(status_code=500, detail=str(e))



