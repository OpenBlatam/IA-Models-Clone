"""
Health API
==========

Simple and clear health API for the Document Workflow Chain system.
"""

from __future__ import annotations
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_database
from ..core.config import settings

# Create router
router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check - simple and clear"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_database)):
    """Detailed health check - simple and clear"""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": db_status,
            "api": "healthy"
        }
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - simple and clear"""
    return {
        "status": "ready",
        "version": settings.VERSION
    }


@router.get("/live")
async def liveness_check():
    """Liveness check - simple and clear"""
    return {
        "status": "alive",
        "version": settings.VERSION
    }


