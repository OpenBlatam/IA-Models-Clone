"""
Resource API Endpoints
======================

Endpoints para resource pool y batch processor.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
import logging

from ..core.resource_pool import get_resource_pool
from ..core.batch_processor_advanced import get_advanced_batch_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])


@router.get("/pools/{pool_name}/statistics")
async def get_resource_pool_statistics(pool_name: str) -> Dict[str, Any]:
    """Obtener estadísticas de pool de recursos."""
    try:
        pool = get_resource_pool(pool_name)
        if not pool:
            raise HTTPException(status_code=404, detail="Resource pool not found")
        
        stats = pool.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resource pool statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batches")
async def create_batch(
    batch_id: str,
    items: List[Any] = Body(...),
    batch_size: int = Query(10, ge=1, le=1000),
    max_workers: int = Query(5, ge=1, le=100)
) -> Dict[str, Any]:
    """Crear y procesar batch."""
    try:
        processor = get_advanced_batch_processor(
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # Función de ejemplo (en producción sería una función real)
        async def process_item(item):
            return {"processed": item}
        
        batch = await processor.process_batch(
            batch_id=batch_id,
            items=items,
            processor_func=process_item
        )
        
        return {
            "batch_id": batch.batch_id,
            "status": batch.status.value,
            "items_count": len(batch.items),
            "completed_at": batch.completed_at
        }
    except Exception as e:
        logger.error(f"Error creating batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str) -> Dict[str, Any]:
    """Obtener batch."""
    try:
        processor = get_advanced_batch_processor()
        batch = processor.get_batch(batch_id)
        
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        completed = sum(1 for item in batch.items if item.status.value == "completed")
        failed = sum(1 for item in batch.items if item.status.value == "failed")
        
        return {
            "batch_id": batch.batch_id,
            "status": batch.status.value,
            "items_count": len(batch.items),
            "completed_items": completed,
            "failed_items": failed,
            "created_at": batch.created_at,
            "started_at": batch.started_at,
            "completed_at": batch.completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batches/statistics")
async def get_batch_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de batches."""
    try:
        processor = get_advanced_batch_processor()
        stats = processor.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting batch statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






