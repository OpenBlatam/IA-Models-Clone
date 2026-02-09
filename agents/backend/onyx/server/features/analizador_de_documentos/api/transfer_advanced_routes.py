"""
Rutas para Advanced Transfer Learning
=======================================

Endpoints para transfer learning avanzado.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.transfer_learning_advanced import (
    get_advanced_transfer_learning,
    AdvancedTransferLearning,
    TransferStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-transfer-learning",
    tags=["Advanced Transfer Learning"]
)


class CreateTransferTaskRequest(BaseModel):
    """Request para crear tarea de transfer"""
    source_domain: str = Field(..., description="Dominio fuente")
    target_domain: str = Field(..., description="Dominio objetivo")
    source_model: str = Field(..., description="Modelo fuente")
    strategy: str = Field("fine_tuning", description="Estrategia")


@router.post("/tasks")
async def create_transfer_task(
    request: CreateTransferTaskRequest,
    system: AdvancedTransferLearning = Depends(get_advanced_transfer_learning)
):
    """Crear tarea de transfer learning"""
    try:
        strategy = TransferStrategy(request.strategy)
        task = system.create_transfer_task(
            request.source_domain,
            request.target_domain,
            request.source_model,
            strategy
        )
        
        return {
            "task_id": task.task_id,
            "source_domain": task.source_domain,
            "target_domain": task.target_domain,
            "strategy": task.strategy.value,
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creando tarea: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/execute")
async def execute_transfer(
    task_id: str,
    target_data: List[Dict[str, Any]] = Field(..., description="Datos del dominio objetivo"),
    epochs: int = Field(10, description="Número de épocas"),
    system: AdvancedTransferLearning = Depends(get_advanced_transfer_learning)
):
    """Ejecutar transfer learning"""
    try:
        result = system.execute_transfer(task_id, target_data, epochs)
        
        return {
            "task_id": result.task_id,
            "transferred_model_id": result.transferred_model_id,
            "performance_source": result.performance_source,
            "performance_target": result.performance_target,
            "transfer_efficiency": result.transfer_efficiency
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando transfer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-similarity")
async def analyze_domain_similarity(
    source_domain: str = Field(..., description="Dominio fuente"),
    target_domain: str = Field(..., description="Dominio objetivo"),
    system: AdvancedTransferLearning = Depends(get_advanced_transfer_learning)
):
    """Analizar similitud entre dominios"""
    try:
        similarity = system.analyze_domain_similarity(source_domain, target_domain)
        
        return similarity
    except Exception as e:
        logger.error(f"Error analizando similitud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


