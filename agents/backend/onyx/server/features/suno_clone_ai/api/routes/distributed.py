"""
API de Inferencia Distribuida

Endpoints para:
- Registrar workers
- Obtener worker disponible
- Estadísticas de inferencia distribuida
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.distributed_inference import get_distributed_inference
from middleware.auth_middleware import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/distributed",
    tags=["distributed"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/workers")
async def register_worker(
    worker_id: str = Body(..., description="ID del worker"),
    url: str = Body(..., description="URL del worker"),
    capacity: int = Body(10, description="Capacidad concurrente")
) -> Dict[str, Any]:
    """
    Registra un worker de inferencia.
    """
    try:
        inference_engine = get_distributed_inference()
        inference_engine.register_worker(
            worker_id=worker_id,
            url=url,
            capacity=capacity
        )
        
        return {
            "message": "Worker registered successfully",
            "worker_id": worker_id,
            "url": url,
            "capacity": capacity
        }
    except Exception as e:
        logger.error(f"Error registering worker: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering worker: {str(e)}"
        )


@router.get("/worker")
async def get_available_worker(
    service_name: str = Query("default", description="Nombre del servicio")
) -> Dict[str, Any]:
    """
    Obtiene un worker disponible para inferencia.
    """
    try:
        inference_engine = get_distributed_inference()
        worker = inference_engine.get_available_worker()
        
        if not worker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No workers available"
            )
        
        return {
            "worker_id": worker.id,
            "url": worker.url,
            "capacity": worker.capacity,
            "active_tasks": worker.active_tasks,
            "available_capacity": worker.capacity - worker.active_tasks,
            "healthy": worker.healthy,
            "avg_latency": worker.avg_latency
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting worker: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting worker: {str(e)}"
        )


@router.get("/stats")
async def get_distributed_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas de inferencia distribuida.
    """
    try:
        inference_engine = get_distributed_inference()
        stats = inference_engine.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting distributed stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

