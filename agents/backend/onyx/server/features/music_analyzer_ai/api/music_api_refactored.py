"""
Refactored API endpoints for music analysis
Uses domain-based routers for better organization
"""

from fastapi import APIRouter
import logging

from .routes.main_router import create_main_router

logger = logging.getLogger(__name__)

# Create main router with prefix
router = APIRouter(prefix="/music", tags=["Music Analysis"])

# Include all domain routers
main_router = create_main_router()
router.include_router(main_router)

# Note: Cache and export routers are already included in main_router

# Root and health endpoints
@router.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Music Analyzer AI",
        "version": "2.21.0",
        "status": "running",
        "description": "Sistema de análisis musical con integración a Spotify"
    }


# Health endpoints are now in HealthRouter

