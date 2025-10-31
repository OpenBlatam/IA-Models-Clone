"""
Unified FastAPI Integration for the ads feature.

This router provides lightweight endpoints to surface the legacy FastAPI docs reference
integration within the unified ads API.
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/fastapi-integration", tags=["ads-fastapi-integration"])

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "component": "fastapi_integration",
        "timestamp": datetime.now().isoformat(),
    }

@router.get("/capabilities")
async def capabilities():
    return {
        "routes": [
            "/health",
            "/capabilities",
            "/libraries",
        ],
        "description": "Legacy FastAPI docs reference endpoints exposed under unified API",
    }

@router.get("/libraries")
async def list_libraries():
    # Static placeholder mirroring legacy example
    return {
        "available_libraries": ["pytorch", "transformers", "diffusers", "gradio"],
        "count": 4,
    }

__all__ = ["router"]






