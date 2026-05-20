"""
Rutas para Transfer Learning
==============================

Endpoints para transfer learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.transfer_learning import (
    get_transfer_learning,
    TransferLearningSystem,
    TransferMode
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/transfer-learning",
    tags=["Transfer Learning"]
)


class CreateTransferTaskRequest(BaseModel):
    """Request para crear tarea de transfer"""
    source_model: str = Field(..., description="Modelo fuente")
    target_domain: str = Field(..., description="Dominio objetivo")
    transfer_mode: str = Field("fine_tuning", description="Modo de transfer")


class ExecuteTransferRequest(BaseModel):
    """Request para ejecutar transfer"""
    training_data: List[Dict[str, Any]] = Field(..., description="Datos de entrenamiento")
    epochs: int = Field(5, description="Número de épocas")


@router.post("/tasks")
async def create_transfer_task(
    request: CreateTransferTaskRequest,
    system: TransferLearningSystem = Depends(get_transfer_learning)
):
    """Crear tarea de transfer learning"""
    try:
        transfer_mode = TransferMode(request.transfer_mode)
        task = system.create_transfer_task(
            request.source_model,
            request.target_domain,
            transfer_mode
        )
        
        return {
            "task_id": task.task_id,
            "source_model": task.source_model,
            "target_domain": task.target_domain,
            "transfer_mode": task.transfer_mode.value,
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/execute")
async def execute_transfer(
    task_id: str,
    request: ExecuteTransferRequest,
    system: TransferLearningSystem = Depends(get_transfer_learning)
):
    """Ejecutar transfer learning"""
    try:
        result = system.execute_transfer(
            task_id,
            request.training_data,
            request.epochs
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/evaluate")
async def evaluate_model(
    model_id: str,
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba"),
    system: TransferLearningSystem = Depends(get_transfer_learning)
):
    """Evaluar modelo transferido"""
    try:
        metrics = system.evaluate_transfer(model_id, test_data)
        
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))



