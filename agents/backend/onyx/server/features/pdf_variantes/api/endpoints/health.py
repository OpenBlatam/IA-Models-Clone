"""
PDF Variantes API - Health Check Endpoint
Simple health check for monitoring
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter(tags=["Health"], prefix="")


@router.get("/health")
async def health_check():
    """Health check endpoint - Simple version for quick checks"""
    return {
        "status": "healthy",
        "message": "PDF Variantes API is running",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "v1",
        "api_ready": True,
        "frontend_compatible": True
    }

