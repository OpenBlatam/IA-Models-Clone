"""
Base Endpoints
Root and health check endpoints for the FastAPI application
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI

from .lifespan import (
    get_content_generator_instance,
    get_collaboration_service_instance,
    get_analytics_service_instance
)
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

def setup_base_endpoints(app: FastAPI) -> None:
    """Configure base endpoints for the application"""
    settings = get_settings()
    
    @app.get("/", response_model=Dict[str, Any])
    async def root() -> Dict[str, Any]:
        """Root endpoint"""
        return {
            "message": f"Welcome to {settings.app_name} API",
            "version": settings.app_version,
            "status": "running",
            "environment": settings.environment,
            "docs": "/docs" if settings.debug else None,
            "redoc": "/redoc" if settings.debug else None
        }
    
    @app.get("/health", response_model=Dict[str, Any])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.app_version,
            "services": {
                "content_generator": "healthy" if get_content_generator_instance() else "unavailable",
                "collaboration_service": "healthy" if get_collaboration_service_instance() else "unavailable",
                "analytics_service": "healthy" if get_analytics_service_instance() else "unavailable"
            }
        }
    
    logger.info("Base endpoints configured")
