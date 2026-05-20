"""
Rutas para Continual Learning
==============================

Endpoints para continual learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.continual_learning import (
    get_continual_learning,
    ContinualLearning,
    CLStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/continual-learning",
    tags=["Continual Learning"]
)


class AddTaskRequest(BaseModel):
    """Request para agregar tarea"""
    task_name: str = Field(..., description="Nombre de la tarea")
    data: List[Dict[str, Any]] = Field(..., description="Datos de la tarea")
    priority: int = Field(5, description="Prioridad")


class LearnTaskRequest(BaseModel):
    """Request para aprender tarea"""
    strategy: str = Field("ewc", description="Estrategia")
    epochs: int = Field(5, description="Número de épocas")


@router.post("/tasks")
async def add_task(
    request: AddTaskRequest,
    system: ContinualLearning = Depends(get_continual_learning)
):
    """Agregar nueva tarea"""
    try:
        task = system.add_task(
            request.task_name,
            request.data,
            request.priority
        )
        
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "priority": task.priority,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error agregando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/learn")
async def learn_task(
    task_id: str,
    model_id: str = Field(..., description="ID del modelo"),
    request: LearnTaskRequest = ...,
    system: ContinualLearning = Depends(get_continual_learning)
):
    """Aprender tarea"""
    try:
        strategy = CLStrategy(request.strategy)
        result = system.learn_task(task_id, model_id, strategy, request.epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error aprendiendo tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-retention")
async def evaluate_retention(
    model_id: str = Field(..., description="ID del modelo"),
    previous_tasks: List[str] = Field(..., description="IDs de tareas anteriores"),
    system: ContinualLearning = Depends(get_continual_learning)
):
    """Evaluar retención"""
    try:
        retention = system.evaluate_retention(model_id, previous_tasks)
        
        return retention
    except Exception as e:
        logger.error(f"Error evaluando retención: {e}")
        raise HTTPException(status_code=500, detail=str(e))



