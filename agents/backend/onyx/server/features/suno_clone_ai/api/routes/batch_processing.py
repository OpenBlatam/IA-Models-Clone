"""
API de Procesamiento por Lotes

Endpoints para:
- Crear batches
- Procesar batches
- Obtener estado de batches
- Cancelar batches
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.batch_processor import (
    get_batch_processor,
    BatchPriority
)
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/batch",
    tags=["batch"]
)


@router.post("/create")
async def create_batch(
    items: List[Any] = Body(..., description="Lista de items a procesar"),
    priority: str = Body("normal", description="Prioridad (low, normal, high, urgent)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea un nuevo batch para procesamiento.
    """
    try:
        # Validar prioridad
        try:
            batch_priority = BatchPriority[priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority: {priority}"
            )
        
        processor = get_batch_processor()
        batch_id = processor.create_batch(
            items=items,
            priority=batch_priority
        )
        
        return {
            "batch_id": batch_id,
            "message": "Batch created successfully",
            "items_count": len(items),
            "priority": priority
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating batch: {str(e)}"
        )


@router.get("/{batch_id}")
async def get_batch_status(batch_id: str) -> Dict[str, Any]:
    """
    Obtiene el estado de un batch.
    """
    try:
        processor = get_batch_processor()
        batch = processor.get_batch(batch_id)
        
        if not batch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        return {
            "batch_id": batch.id,
            "status": batch.status.value,
            "priority": batch.priority.value,
            "progress": batch.progress,
            "items_total": len(batch.items),
            "items_completed": sum(1 for item in batch.items if item.status == "completed"),
            "items_failed": sum(1 for item in batch.items if item.status == "failed"),
            "created_at": batch.created_at.isoformat(),
            "started_at": batch.started_at.isoformat() if batch.started_at else None,
            "completed_at": batch.completed_at.isoformat() if batch.completed_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving batch status: {str(e)}"
        )


@router.post("/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Cancela un batch pendiente.
    """
    try:
        processor = get_batch_processor()
        success = processor.cancel_batch(batch_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not cancel batch {batch_id} (may be processing or completed)"
            )
        
        return {
            "message": f"Batch {batch_id} cancelled successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling batch: {str(e)}"
        )


@router.get("/stats")
async def get_batch_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas de procesamiento por lotes.
    """
    try:
        processor = get_batch_processor()
        stats = processor.get_batch_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting batch stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

