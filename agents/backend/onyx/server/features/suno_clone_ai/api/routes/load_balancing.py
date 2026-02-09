"""
API de Load Balancing

Endpoints para:
- Agregar backends
- Obtener backend
- Estadísticas de load balancer
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.load_balancer import (
    get_load_balancer,
    LoadBalancingStrategy
)
from middleware.auth_middleware import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/load-balancer",
    tags=["load-balancer"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/backends")
async def add_backend(
    backend_id: str = Body(..., description="ID del backend"),
    url: str = Body(..., description="URL del backend"),
    weight: int = Body(1, description="Peso para weighted round-robin"),
    health_check_url: Optional[str] = Body(None, description="URL de health check")
) -> Dict[str, Any]:
    """
    Agrega un backend al load balancer.
    """
    try:
        load_balancer = get_load_balancer()
        load_balancer.add_backend(
            backend_id=backend_id,
            url=url,
            weight=weight,
            health_check_url=health_check_url
        )
        
        return {
            "message": "Backend added successfully",
            "backend_id": backend_id,
            "url": url
        }
    except Exception as e:
        logger.error(f"Error adding backend: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding backend: {str(e)}"
        )


@router.get("/backend")
async def get_backend(
    service_name: str = Query("default", description="Nombre del servicio")
) -> Dict[str, Any]:
    """
    Obtiene un backend según la estrategia de load balancing.
    """
    try:
        load_balancer = get_load_balancer()
        backend = load_balancer.get_backend(service_name=service_name)
        
        if not backend:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No backends available"
            )
        
        return {
            "backend_id": backend.id,
            "url": backend.url,
            "weight": backend.weight,
            "healthy": backend.healthy,
            "active_connections": backend.active_connections,
            "success_rate": backend.get_success_rate(),
            "avg_response_time": backend.get_avg_response_time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backend: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting backend: {str(e)}"
        )


@router.get("/stats")
async def get_load_balancer_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del load balancer.
    """
    try:
        load_balancer = get_load_balancer()
        stats = load_balancer.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting load balancer stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


@router.delete("/backends/{backend_id}")
async def remove_backend(backend_id: str) -> Dict[str, Any]:
    """
    Elimina un backend del load balancer.
    """
    try:
        load_balancer = get_load_balancer()
        load_balancer.remove_backend(backend_id)
        
        return {
            "message": f"Backend {backend_id} removed successfully"
        }
    except Exception as e:
        logger.error(f"Error removing backend: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing backend: {str(e)}"
        )

