"""
Router Registration
Centralized router registration for the FastAPI application
"""

import logging
from fastapi import FastAPI

from .routes import content_router, collaboration_router, export_router, analytics_router
from .bul_routes import router as bul_router

logger = logging.getLogger(__name__)

def setup_routers(app: FastAPI) -> None:
    """Register all routers with the application"""
    app.include_router(content_router, prefix="/api/v1/content", tags=["content"])
    app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["collaboration"])
    app.include_router(export_router, prefix="/api/v1/export", tags=["export"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["analytics"])
    app.include_router(bul_router, prefix="/api/v1", tags=["bul"])
    
    logger.info("All routers registered successfully")
