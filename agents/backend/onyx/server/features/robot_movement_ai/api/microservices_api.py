"""
Microservices API Endpoints
===========================

Endpoints para microservices orchestrator y API composition.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
import logging

from ..core.microservices_orchestrator import get_microservices_orchestrator
from ..core.api_composition import (
    get_api_composer,
    CompositionStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/microservices", tags=["microservices"])


@router.post("/services/register")
async def register_service(
    name: str,
    endpoint: str,
    version: str = "1.0.0",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar microservicio."""
    try:
        orchestrator = get_microservices_orchestrator()
        service_id = orchestrator.register_service(
            name=name,
            endpoint=endpoint,
            version=version,
            metadata=metadata
        )
        
        return {
            "service_id": service_id,
            "name": name,
            "endpoint": endpoint,
            "version": version
        }
    except Exception as e:
        logger.error(f"Error registering service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/call")
async def call_service(
    service_id: str,
    method: str = "GET",
    endpoint: str = "",
    payload: Dict[str, Any] = Body(...),
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Llamar a microservicio."""
    try:
        orchestrator = get_microservices_orchestrator()
        result = await orchestrator.call_service(
            service_id=service_id,
            method=method,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout
        )
        
        return {
            "service_id": service_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error calling service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service_id}/health-check")
async def health_check_service(service_id: str) -> Dict[str, Any]:
    """Verificar salud de microservicio."""
    try:
        orchestrator = get_microservices_orchestrator()
        is_healthy = await orchestrator.health_check(service_id)
        
        service = orchestrator.get_service(service_id)
        if not service:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return {
            "service_id": service_id,
            "name": service.name,
            "status": service.status.value,
            "healthy": is_healthy,
            "last_health_check": service.last_health_check
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking service health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/statistics")
async def get_services_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de servicios."""
    try:
        orchestrator = get_microservices_orchestrator()
        stats = orchestrator.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting services statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compositions/create")
async def create_composition(
    name: str,
    strategy: str,
    steps: List[Dict[str, Any]] = Body(...),
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear composición de API."""
    try:
        composer = get_api_composer()
        strategy_enum = CompositionStrategy(strategy.lower())
        composition_id = composer.create_composition(
            name=name,
            strategy=strategy_enum,
            steps=steps,
            metadata=metadata
        )
        
        return {
            "composition_id": composition_id,
            "name": name,
            "strategy": strategy,
            "steps_count": len(steps)
        }
    except Exception as e:
        logger.error(f"Error creating composition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compositions/{composition_id}/execute")
async def execute_composition(
    composition_id: str,
    context: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Ejecutar composición de API."""
    try:
        composer = get_api_composer()
        result = await composer.execute_composition(
            composition_id=composition_id,
            context=context
        )
        
        return {
            "composition_id": composition_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error executing composition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compositions/{composition_id}")
async def get_composition(composition_id: str) -> Dict[str, Any]:
    """Obtener composición."""
    try:
        composer = get_api_composer()
        composition = composer.get_composition(composition_id)
        
        if not composition:
            raise HTTPException(status_code=404, detail="Composition not found")
        
        return {
            "composition_id": composition.composition_id,
            "name": composition.name,
            "strategy": composition.strategy.value,
            "steps_count": len(composition.steps),
            "created_at": composition.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting composition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compositions/statistics")
async def get_compositions_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de composiciones."""
    try:
        composer = get_api_composer()
        stats = composer.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting compositions statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


