"""
Rutas para Imitation Learning
===============================

Endpoints para aprendizaje por imitación.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.imitation_learning import (
    get_imitation_learning,
    ImitationLearning,
    ImitationMethod,
    ExpertDemonstration
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/imitation-learning",
    tags=["Imitation Learning"]
)


class CreateTaskRequest(BaseModel):
    """Request para crear tarea"""
    demonstrations: List[Dict[str, Any]] = Field(..., description="Demostraciones")
    method: str = Field("behavioral_cloning", description="Método")


@router.post("/tasks")
async def create_task(
    request: CreateTaskRequest,
    system: ImitationLearning = Depends(get_imitation_learning)
):
    """Crear tarea de imitación"""
    try:
        method = ImitationMethod(request.method)
        
        demos = [
            ExpertDemonstration(
                demo_id=demo.get("demo_id", f"demo_{i}"),
                state=demo.get("state", {}),
                action=demo.get("action"),
                reward=demo.get("reward")
            )
            for i, demo in enumerate(request.demonstrations)
        ]
        
        task = system.create_task(demos, method)
        
        return {
            "task_id": task.task_id,
            "method": task.method.value,
            "num_demos": len(task.expert_demos),
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/learn")
async def learn_from_demonstrations(
    task_id: str,
    epochs: int = Field(10, description="Número de épocas"),
    system: ImitationLearning = Depends(get_imitation_learning)
):
    """Aprender de demostraciones"""
    try:
        result = system.learn_from_demonstrations(task_id, epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error aprendiendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policies/{policy_id}/evaluate")
async def evaluate_imitation(
    policy_id: str,
    test_states: List[Dict[str, Any]] = Field(..., description="Estados de prueba"),
    system: ImitationLearning = Depends(get_imitation_learning)
):
    """Evaluar política aprendida"""
    try:
        evaluation = system.evaluate_imitation(policy_id, test_states)
        
        return evaluation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


