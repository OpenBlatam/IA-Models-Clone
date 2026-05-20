"""
Continual Learning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.continual_learning import (
    ContinualLearningService,
    ContinualLearningConfig,
    ContinualLearningMethod
)

router = APIRouter()
continual_learning_service = ContinualLearningService()


@router.post("/store-task")
async def store_task(
    task_name: str
) -> Dict[str, Any]:
    """Almacenar pesos de una tarea"""
    try:
        return {
            "task_name": task_name,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_name}")
async def get_task(task_name: str) -> Dict[str, Any]:
    """Obtener pesos de una tarea"""
    try:
        task_data = continual_learning_service.get_task_weights(task_name)
        if not task_data:
            return {"error": "Task not found"}
        
        return {
            "task_name": task_name,
            "stored": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




