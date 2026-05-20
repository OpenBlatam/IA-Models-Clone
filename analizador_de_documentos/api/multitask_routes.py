"""
Rutas para Multi-Task Learning
================================

Endpoints para multi-task learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.multi_task_learning import (
    get_multi_task_learning,
    MultiTaskLearning,
    MultiTaskMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/multi-task-learning",
    tags=["Multi-Task Learning"]
)


class AddTaskRequest(BaseModel):
    """Request para agregar tarea"""
    task_name: str = Field(..., description="Nombre de la tarea")
    task_type: str = Field(..., description="Tipo de tarea")
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    priority: int = Field(5, description="Prioridad")


@router.post("/tasks")
async def add_task(
    request: AddTaskRequest,
    system: MultiTaskLearning = Depends(get_multi_task_learning)
):
    """Agregar tarea"""
    try:
        task = system.add_task(
            request.task_name,
            request.task_type,
            request.data,
            request.priority
        )
        
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "priority": task.priority
        }
    except Exception as e:
        logger.error(f"Error agregando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_multi_task(
    task_ids: List[str] = Field(..., description="IDs de tareas"),
    method: str = Field("hard_parameter_sharing", description="Método"),
    epochs: int = Field(10, description="Número de épocas"),
    system: MultiTaskLearning = Depends(get_multi_task_learning)
):
    """Entrenar modelo multi-tarea"""
    try:
        mt_method = MultiTaskMethod(method)
        result = system.train_multi_task(task_ids, mt_method, epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/evaluate/{task_id}")
async def evaluate_task(
    model_id: str,
    task_id: str,
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba"),
    system: MultiTaskLearning = Depends(get_multi_task_learning)
):
    """Evaluar tarea específica"""
    try:
        evaluation = system.evaluate_task(model_id, task_id, test_data)
        
        return evaluation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


