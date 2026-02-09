"""
Pattern API Endpoints
=====================

Endpoints para CQRS y Saga patterns.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
import logging

from ..core.cqrs_pattern import get_cqrs_system
from ..core.saga_pattern import get_saga_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/patterns", tags=["patterns"])


@router.post("/cqrs/commands/register")
async def register_command_handler(
    command_type: str,
    handler_info: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Registrar manejador de comando."""
    try:
        # En producción, esto sería una función real
        async def handler(payload):
            return {"processed": payload}
        
        cqrs = get_cqrs_system()
        cqrs.register_command_handler(command_type, handler)
        
        return {
            "command_type": command_type,
            "registered": True
        }
    except Exception as e:
        logger.error(f"Error registering command handler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cqrs/commands/execute")
async def execute_command(
    command_type: str,
    payload: Dict[str, Any] = Body(...),
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ejecutar comando."""
    try:
        cqrs = get_cqrs_system()
        result = await cqrs.execute_command(
            command_type=command_type,
            payload=payload,
            metadata=metadata
        )
        
        return {
            "command_id": result.command_id,
            "success": result.success,
            "result": result.result,
            "error": result.error
        }
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cqrs/queries/register")
async def register_query_handler(
    query_type: str,
    handler_info: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Registrar manejador de query."""
    try:
        # En producción, esto sería una función real
        async def handler(parameters):
            return {"data": parameters}
        
        cqrs = get_cqrs_system()
        cqrs.register_query_handler(query_type, handler)
        
        return {
            "query_type": query_type,
            "registered": True
        }
    except Exception as e:
        logger.error(f"Error registering query handler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cqrs/queries/execute")
async def execute_query(
    query_type: str,
    parameters: Dict[str, Any] = Body(...),
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ejecutar query."""
    try:
        cqrs = get_cqrs_system()
        result = await cqrs.execute_query(
            query_type=query_type,
            parameters=parameters,
            metadata=metadata
        )
        
        return {
            "query_id": result.query_id,
            "data": result.data
        }
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cqrs/statistics")
async def get_cqrs_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de CQRS."""
    try:
        cqrs = get_cqrs_system()
        stats = cqrs.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting CQRS statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sagas/create")
async def create_saga(
    name: str,
    steps: List[Dict[str, Any]] = Body(...),
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear saga."""
    try:
        manager = get_saga_manager()
        saga_id = manager.create_saga(
            name=name,
            steps=steps,
            metadata=metadata
        )
        
        return {
            "saga_id": saga_id,
            "name": name,
            "steps_count": len(steps)
        }
    except Exception as e:
        logger.error(f"Error creating saga: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sagas/{saga_id}/execute")
async def execute_saga(saga_id: str) -> Dict[str, Any]:
    """Ejecutar saga."""
    try:
        manager = get_saga_manager()
        result = await manager.execute_saga(saga_id)
        return result
    except Exception as e:
        logger.error(f"Error executing saga: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sagas/{saga_id}")
async def get_saga(saga_id: str) -> Dict[str, Any]:
    """Obtener saga."""
    try:
        manager = get_saga_manager()
        saga = manager.get_saga(saga_id)
        
        if not saga:
            raise HTTPException(status_code=404, detail="Saga not found")
        
        return {
            "saga_id": saga.saga_id,
            "name": saga.name,
            "status": saga.status.value,
            "current_step": saga.current_step,
            "steps_count": len(saga.steps),
            "created_at": saga.created_at,
            "started_at": saga.started_at,
            "completed_at": saga.completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting saga: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sagas/statistics")
async def get_saga_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de sagas."""
    try:
        manager = get_saga_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting saga statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


