"""
Load Balancer API Endpoints
===========================

Endpoints para load balancer y circuit breaker.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.load_balancer import (
    create_load_balancer,
    get_load_balancer,
    LoadBalanceStrategy
)
from ..core.circuit_breaker import (
    get_circuit_breaker_manager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/load-balancer", tags=["load-balancer"])


@router.post("/balancers")
async def create_balancer(
    name: str,
    strategy: str = "round_robin"
) -> Dict[str, Any]:
    """Crear balanceador de carga."""
    try:
        strategy_enum = LoadBalanceStrategy(strategy.lower())
        balancer = create_load_balancer(name, strategy_enum)
        return {
            "name": balancer.name,
            "strategy": balancer.strategy.value
        }
    except Exception as e:
        logger.error(f"Error creating load balancer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/balancers/{name}/servers")
async def add_server(
    name: str,
    server_id: str,
    server_name: str,
    address: str,
    weight: int = 1
) -> Dict[str, Any]:
    """Agregar servidor al balanceador."""
    try:
        balancer = get_load_balancer(name)
        if not balancer:
            raise HTTPException(status_code=404, detail="Load balancer not found")
        
        server = balancer.add_server(
            server_id=server_id,
            name=server_name,
            address=address,
            weight=weight
        )
        
        return {
            "server_id": server.server_id,
            "name": server.name,
            "address": server.address
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balancers/{name}/server")
async def get_server(name: str) -> Dict[str, Any]:
    """Obtener servidor del balanceador."""
    try:
        balancer = get_load_balancer(name)
        if not balancer:
            raise HTTPException(status_code=404, detail="Load balancer not found")
        
        server = balancer.get_server()
        if not server:
            raise HTTPException(status_code=404, detail="No available servers")
        
        return {
            "server_id": server.server_id,
            "name": server.name,
            "address": server.address,
            "active": server.active
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting server: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balancers/{name}/statistics")
async def get_balancer_statistics(name: str) -> Dict[str, Any]:
    """Obtener estadísticas del balanceador."""
    try:
        balancer = get_load_balancer(name)
        if not balancer:
            raise HTTPException(status_code=404, detail="Load balancer not found")
        
        stats = balancer.get_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breakers")
async def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """Crear circuit breaker."""
    try:
        manager = get_circuit_breaker_manager()
        breaker = manager.create_breaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout
        )
        return {
            "name": breaker.name,
            "state": breaker.state.value,
            "failure_threshold": breaker.failure_threshold
        }
    except Exception as e:
        logger.error(f"Error creating circuit breaker: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit-breakers/{name}/statistics")
async def get_circuit_breaker_statistics(name: str) -> Dict[str, Any]:
    """Obtener estadísticas de circuit breaker."""
    try:
        manager = get_circuit_breaker_manager()
        stats = manager.get_statistics(name)
        return stats
    except Exception as e:
        logger.error(f"Error getting circuit breaker statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






