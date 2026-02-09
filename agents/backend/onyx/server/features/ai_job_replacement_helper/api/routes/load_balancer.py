"""
Load Balancer endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.load_balancer import LoadBalancerService, LoadBalancingAlgorithm

router = APIRouter()
lb_service = LoadBalancerService()


@router.post("/create")
async def create_load_balancer(
    name: str,
    algorithm: str,
    backends: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Crear balanceador de carga"""
    try:
        algorithm_enum = LoadBalancingAlgorithm(algorithm)
        balancer = lb_service.create_load_balancer(name, algorithm_enum, backends)
        return {
            "name": balancer.name,
            "algorithm": balancer.algorithm.value,
            "backends_count": len(balancer.backends),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/select/{balancer_name}")
async def select_backend(
    balancer_name: str,
    client_ip: Optional[str] = None
) -> Dict[str, Any]:
    """Seleccionar backend"""
    try:
        backend = lb_service.select_backend(balancer_name, client_ip)
        if not backend:
            raise HTTPException(status_code=404, detail="No healthy backends available")
        
        return {
            "backend_id": backend.id,
            "host": backend.host,
            "port": backend.port,
            "url": f"http://{backend.host}:{backend.port}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{balancer_name}")
async def get_balancer_stats(balancer_name: str) -> Dict[str, Any]:
    """Obtener estadísticas del balanceador"""
    try:
        stats = lb_service.get_balancer_stats(balancer_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




