"""
API Gateway endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.api_gateway import APIGatewayService, RouteMethod

router = APIRouter()
gateway_service = APIGatewayService()


@router.post("/register-route")
async def register_route(
    path: str,
    method: str,
    target_url: str,
    timeout: int = 30,
    retries: int = 3
) -> Dict[str, Any]:
    """Registrar ruta en gateway"""
    try:
        method_enum = RouteMethod(method.upper())
        route = gateway_service.register_route(
            path, method_enum, target_url, timeout, retries
        )
        return {
            "path": route.path,
            "method": route.method.value,
            "target_url": route.target_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routes")
async def get_routes() -> Dict[str, Any]:
    """Obtener todas las rutas"""
    try:
        routes = gateway_service.get_routes()
        return {
            "routes": routes,
            "total": len(routes),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

