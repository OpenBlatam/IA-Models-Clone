"""
Health Router - System health and status endpoints
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        registry = getattr(request.app.state, "services", None)
        services_status = {}
        
        if registry:
            health_status = registry.get_health_status()
            services_status = health_status.get("services", {})
        
        data = {
            "success": True,
            "data": {
                "status": "healthy",
                "timestamp": time.time(),
                "services": services_status,
                "api_version": "v1"
            },
            "error": None
        }
        # Ensure health is not cached
        response = JSONResponse(content=data)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": {
                    "message": "Health check failed",
                    "status_code": 500,
                    "type": "HealthCheckError"
                }
            }
        )


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check(request: Request) -> Dict[str, Any]:
    """Kubernetes readiness probe"""
    registry = getattr(request.app.state, "services", None)
    
    if not registry:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Services not initialized"}
        )
    
    health_status = registry.get_health_status()
    is_ready = health_status.get("initialized", False)
    
    if is_ready:
        resp = JSONResponse(content={"status": "ready"})
        resp.headers["Cache-Control"] = "no-store"
        return resp
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
            headers={"Cache-Control": "no-store"}
        )


@router.get("/live", response_model=Dict[str, Any])
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe"""
    return JSONResponse(content={"status": "alive"}, headers={"Cache-Control": "no-store"})

