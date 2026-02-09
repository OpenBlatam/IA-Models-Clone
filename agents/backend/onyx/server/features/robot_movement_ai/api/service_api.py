"""
Service Discovery API Endpoints
================================

Endpoints para service discovery y distributed locks.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.service_discovery import (
    get_service_discovery,
    ServiceStatus
)
from ..core.distributed_lock import get_distributed_lock_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/services", tags=["services"])


@router.post("/register")
async def register_service(
    service_id: str,
    name: str,
    address: str,
    port: int,
    version: str = "1.0.0",
    group: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar servicio."""
    try:
        discovery = get_service_discovery()
        service = discovery.register_service(
            service_id=service_id,
            name=name,
            address=address,
            port=port,
            version=version,
            group=group,
            metadata=metadata
        )
        return {
            "service_id": service.service_id,
            "name": service.name,
            "address": service.address,
            "port": service.port,
            "status": service.status.value
        }
    except Exception as e:
        logger.error(f"Error registering service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def find_services(
    name: Optional[str] = None,
    group: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """Buscar servicios."""
    try:
        discovery = get_service_discovery()
        service_status = ServiceStatus(status.lower()) if status else None
        services = discovery.find_services(name=name, group=group, status=service_status)
        return {
            "services": [
                {
                    "service_id": s.service_id,
                    "name": s.name,
                    "address": s.address,
                    "port": s.port,
                    "version": s.version,
                    "status": s.status.value
                }
                for s in services
            ],
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"Error finding services: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_service_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de servicios."""
    try:
        discovery = get_service_discovery()
        stats = discovery.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting service statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/locks/acquire")
async def acquire_lock(
    resource: str,
    owner: str,
    ttl: float = 60.0,
    wait_timeout: float = 10.0
) -> Dict[str, Any]:
    """Adquirir lock distribuido."""
    try:
        lock_manager = get_distributed_lock_manager()
        lock = await lock_manager.acquire(
            resource=resource,
            owner=owner,
            ttl=ttl,
            wait_timeout=wait_timeout
        )
        
        if not lock:
            raise HTTPException(status_code=408, detail="Failed to acquire lock (timeout)")
        
        return {
            "lock_id": lock.lock_id,
            "resource": lock.resource,
            "owner": lock.owner,
            "expires_at": lock.expires_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acquiring lock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/locks/release")
async def release_lock(
    resource: str,
    owner: str
) -> Dict[str, Any]:
    """Liberar lock distribuido."""
    try:
        lock_manager = get_distributed_lock_manager()
        success = await lock_manager.release(resource, owner)
        
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found or not owned")
        
        return {
            "resource": resource,
            "released": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing lock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/locks/{resource}")
async def get_lock(resource: str) -> Dict[str, Any]:
    """Obtener información de lock."""
    try:
        lock_manager = get_distributed_lock_manager()
        lock = lock_manager.get_lock(resource)
        
        if not lock:
            raise HTTPException(status_code=404, detail="Lock not found")
        
        return {
            "lock_id": lock.lock_id,
            "resource": lock.resource,
            "owner": lock.owner,
            "status": lock.status.value,
            "expires_at": lock.expires_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lock: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/locks/statistics")
async def get_lock_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de locks."""
    try:
        lock_manager = get_distributed_lock_manager()
        stats = lock_manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting lock statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






