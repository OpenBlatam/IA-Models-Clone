"""
API v1 - Versión específica de la API
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/")
async def v1_info():
    """Información de la API v1"""
    return JSONResponse(content={
        "version": "1.0",
        "status": "active",
        "endpoints": {
            "analyze": "/dermatology/v1/analyze",
            "health": "/dermatology/v1/health"
        }
    })


@router.get("/health")
async def v1_health():
    """Health check v1"""
    return JSONResponse(content={
        "status": "healthy",
        "version": "1.0"
    })






