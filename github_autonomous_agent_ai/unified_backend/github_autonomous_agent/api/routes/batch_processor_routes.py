"""
Batch Processor Routes - Rutas para procesamiento en lote.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, Field

import asyncio
from api.utils import handle_api_errors
from core.services.batch_processor import BatchProcessor
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class ProcessBatchRequest(BaseModel):
    """Request para procesar lote de tareas."""
    tasks: List[Dict[str, Any]] = Field(..., min_length=1, description="Lista de tareas a procesar")
    max_concurrent: Optional[int] = Field(None, ge=1, le=50, description="Máximo concurrente (opcional)")
    batch_size: Optional[int] = Field(None, ge=1, le=100, description="Tamaño de lote (opcional)")


def get_batch_processor() -> BatchProcessor:
    """Obtener procesador de lotes."""
    try:
        return get_service("batch_processor")
    except Exception:
        raise HTTPException(status_code=503, detail="Batch processor no disponible")


@router.post("/process")
@handle_api_errors
async def process_batch(
    request: ProcessBatchRequest,
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Procesar lote de tareas.
    
    Args:
        request: Lote de tareas
        
    Returns:
        Resultado del procesamiento
    """
    # Configurar procesador si se proporcionan parámetros
    if request.max_concurrent:
        batch_processor.max_concurrent = request.max_concurrent
        batch_processor.semaphore = asyncio.Semaphore(request.max_concurrent)
    
    if request.batch_size:
        batch_processor.batch_size = request.batch_size
    
    result = await batch_processor.process_batch(request.tasks)
    
    return result


@router.get("/stats")
@handle_api_errors
async def get_batch_processor_stats(
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Obtener estadísticas del procesador de lotes.
    
    Returns:
        Estadísticas
    """
    return batch_processor.get_stats()

