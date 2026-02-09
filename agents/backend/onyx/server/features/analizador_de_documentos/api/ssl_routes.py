"""
Rutas para Self-Supervised Learning
=====================================

Endpoints para self-supervised learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.self_supervised_learning import (
    get_self_supervised,
    SelfSupervisedLearning,
    SSLMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/self-supervised",
    tags=["Self-Supervised Learning"]
)


class CreateSSLTaskRequest(BaseModel):
    """Request para crear tarea SSL"""
    unlabeled_data: List[Dict[str, Any]] = Field(..., description="Datos no etiquetados")
    method: str = Field("masked_language", description="Método")


@router.post("/tasks")
async def create_ssl_task(
    request: CreateSSLTaskRequest,
    system: SelfSupervisedLearning = Depends(get_self_supervised)
):
    """Crear tarea de self-supervised learning"""
    try:
        method = SSLMethod(request.method)
        task = system.create_ssl_task(request.unlabeled_data, method)
        
        return {
            "task_id": task.task_id,
            "method": task.method.value,
            "samples": len(task.unlabeled_data),
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/pretrain")
async def pretrain(
    task_id: str,
    epochs: int = Field(10, description="Número de épocas"),
    system: SelfSupervisedLearning = Depends(get_self_supervised)
):
    """Pre-entrenar modelo"""
    try:
        result = system.pretrain(task_id, epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error pre-entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/transfer")
async def transfer_to_downstream(
    task_id: str,
    downstream_data: List[Dict[str, Any]] = Field(..., description="Datos downstream"),
    fine_tune_epochs: int = Field(5, description="Épocas de fine-tuning"),
    system: SelfSupervisedLearning = Depends(get_self_supervised)
):
    """Transferir a tarea downstream"""
    try:
        result = system.transfer_to_downstream(task_id, downstream_data, fine_tune_epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error transfiriendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))



