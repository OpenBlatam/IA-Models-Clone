"""
API Version Router
=================

API versioning and information endpoints.
"""

from fastapi import APIRouter
from ..config import settings
from ..constants import MAX_LAYERS, MAX_BATCH_DELETE, MAX_BATCH_PREDICT

router = APIRouter(prefix="/v1", tags=["version"])


@router.get("/version")
async def get_version():
    """
    Get API version information.
    
    Returns current API version and compatibility information.
    """
    return {
        "version": settings.app_version,
        "api_version": "v1",
        "name": settings.app_name,
        "description": settings.app_description
    }


@router.get("/info")
async def get_api_info():
    """
    Get comprehensive API information.
    
    Returns detailed information about the API including:
    - Version
    - Available endpoints
    - Feature flags
    - Limits and constraints
    """
    return {
        "version": settings.app_version,
        "api_version": "v1",
        "name": settings.app_name,
        "description": settings.app_description,
        "features": {
            "batch_operations": True,
            "model_comparison": True,
            "operation_history": True,
            "model_export": True,
            "metrics": True,
            "caching": True,
            "rate_limiting": True
        },
        "limits": {
            "max_models": "unlimited",
            "max_layers_per_model": MAX_LAYERS,
            "max_batch_delete": MAX_BATCH_DELETE,
            "max_batch_predict": MAX_BATCH_PREDICT,
            "max_model_size_mb": settings.max_model_size_mb,
            "max_training_samples": settings.max_training_samples,
            "max_batch_size": settings.max_batch_size,
            "rate_limit_per_minute": settings.rate_limit_per_minute
        },
        "endpoints": {
            "models": "/models",
            "batch": "/models/batch",
            "utils": "/utils",
            "health": "/health",
            "metrics": "/metrics",
            "version": "/v1"
        }
    }

