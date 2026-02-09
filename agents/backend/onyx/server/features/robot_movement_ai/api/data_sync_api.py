"""
Data Sync API Endpoints
=======================

Endpoints para data replication y data synchronization.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
import logging

from ..core.data_replication import get_data_replication_manager
from ..core.data_synchronization import (
    get_data_synchronization_manager,
    SyncDirection
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data-sync", tags=["data-sync"])


@router.post("/replication/targets/register")
async def register_replication_target(
    name: str,
    endpoint: str,
    connection_string: Optional[str] = None,
    credentials: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar objetivo de replicación."""
    try:
        manager = get_data_replication_manager()
        target_id = manager.register_target(
            name=name,
            endpoint=endpoint,
            connection_string=connection_string,
            credentials=credentials,
            metadata=metadata
        )
        
        return {
            "target_id": target_id,
            "name": name,
            "endpoint": endpoint
        }
    except Exception as e:
        logger.error(f"Error registering replication target: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replication/jobs/create")
async def create_replication_job(
    source: str,
    targets: List[str],
    data_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear trabajo de replicación."""
    try:
        manager = get_data_replication_manager()
        job_id = manager.create_replication_job(
            source=source,
            targets=targets,
            data_type=data_type,
            metadata=metadata
        )
        
        return {
            "job_id": job_id,
            "source": source,
            "targets": targets,
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Error creating replication job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replication/jobs/{job_id}/execute")
async def execute_replication(
    job_id: str,
    data: Any = Body(...)
) -> Dict[str, Any]:
    """Ejecutar replicación."""
    try:
        manager = get_data_replication_manager()
        result = await manager.execute_replication(job_id, data)
        return result
    except Exception as e:
        logger.error(f"Error executing replication: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replication/statistics")
async def get_replication_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de replicación."""
    try:
        manager = get_data_replication_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting replication statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synchronization/endpoints/register")
async def register_sync_endpoint(
    name: str,
    endpoint: str,
    credentials: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar endpoint de sincronización."""
    try:
        manager = get_data_synchronization_manager()
        endpoint_id = manager.register_endpoint(
            name=name,
            endpoint=endpoint,
            credentials=credentials,
            metadata=metadata
        )
        
        return {
            "endpoint_id": endpoint_id,
            "name": name,
            "endpoint": endpoint
        }
    except Exception as e:
        logger.error(f"Error registering sync endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synchronization/rules/create")
async def create_sync_rule(
    name: str,
    source_endpoint: str,
    target_endpoint: str,
    direction: str = "bidirectional",
    sync_interval: float = 60.0,
    conflict_resolution: str = "source_wins",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear regla de sincronización."""
    try:
        manager = get_data_synchronization_manager()
        direction_enum = SyncDirection(direction.lower())
        rule_id = manager.create_sync_rule(
            name=name,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            direction=direction_enum,
            sync_interval=sync_interval,
            conflict_resolution=conflict_resolution,
            metadata=metadata
        )
        
        return {
            "rule_id": rule_id,
            "name": name,
            "source_endpoint": source_endpoint,
            "target_endpoint": target_endpoint,
            "direction": direction
        }
    except Exception as e:
        logger.error(f"Error creating sync rule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synchronization/rules/{rule_id}/sync")
async def sync_data(
    rule_id: str,
    force: bool = False
) -> Dict[str, Any]:
    """Sincronizar datos según regla."""
    try:
        manager = get_data_synchronization_manager()
        result = await manager.sync(rule_id, force=force)
        
        return {
            "sync_id": result.sync_id,
            "rule_id": result.rule_id,
            "status": result.status.value,
            "items_synced": result.items_synced,
            "conflicts": result.conflicts,
            "error": result.error
        }
    except Exception as e:
        logger.error(f"Error syncing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/synchronization/statistics")
async def get_synchronization_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de sincronización."""
    try:
        manager = get_data_synchronization_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting synchronization statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


