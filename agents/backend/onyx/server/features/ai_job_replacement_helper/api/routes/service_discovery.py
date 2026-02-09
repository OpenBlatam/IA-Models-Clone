"""
Service Discovery endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.service_discovery import ServiceDiscoveryService

router = APIRouter()
discovery_service = ServiceDiscoveryService()


@router.post("/register")
async def register_service(
    service_name: str,
    host: str,
    port: int,
    protocol: str = "http",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar instancia de servicio"""
    try:
        instance = discovery_service.register_service(
            service_name, host, port, protocol, metadata
        )
        return {
            "service_id": instance.service_id,
            "service_name": instance.service_name,
            "host": instance.host,
            "port": instance.port,
            "status": instance.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heartbeat/{service_id}")
async def heartbeat(service_id: str) -> Dict[str, Any]:
    """Enviar heartbeat"""
    try:
        success = discovery_service.heartbeat(service_id)
        return {
            "service_id": service_id,
            "heartbeat_received": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service/{service_name}")
async def get_service_url(
    service_name: str,
    path: str = "",
    strategy: str = "round_robin"
) -> Dict[str, Any]:
    """Obtener URL de servicio"""
    try:
        url = discovery_service.get_service_url(service_name, path, strategy)
        return {
            "service_name": service_name,
            "url": url,
            "found": url is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_services() -> Dict[str, Any]:
    """Obtener todos los servicios"""
    try:
        services = discovery_service.get_all_services()
        return {
            "services": services,
            "total": len(services),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




