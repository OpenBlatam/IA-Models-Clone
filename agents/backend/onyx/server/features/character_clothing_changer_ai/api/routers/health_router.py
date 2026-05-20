"""
Health Router
============

API endpoints for health checks and system status.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import logging

from ...core.clothing_changer_service import ClothingChangerService
from ...api.dependencies import get_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check(service: ClothingChangerService = Depends(get_service)):
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        model_info = service.get_model_info()
        is_initialized = model_info.get("status") != "not_initialized"
        is_deepseek = model_info.get("fallback_mode", False)
        
        return JSONResponse(content={
            "status": "healthy",
            "model_initialized": is_initialized,
            "using_deepseek_fallback": is_deepseek,
            "model_type": model_info.get("primary_model", "unknown"),
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

