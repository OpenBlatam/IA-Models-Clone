"""
Route configuration for FastAPI application
"""

import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_routes(app: FastAPI) -> None:
    """Setup all routes for the application"""
    setup_recovery_routes(app)
    setup_health_routes(app)


def setup_recovery_routes(app: FastAPI) -> None:
    """Setup recovery API routes"""
    try:
        from api import router
        app.include_router(router, prefix="/recovery", tags=["Recovery"])
        logger.info("Using modular recovery API routes")
    except ImportError:
        try:
            from api.recovery_api_refactored import router
            app.include_router(router, prefix="/recovery", tags=["Recovery"])
            logger.info("Using refactored recovery API routes")
        except ImportError:
            logger.warning("Recovery API router not available")


def setup_health_routes(app: FastAPI) -> None:
    """Setup health check routes"""
    try:
        from api.health import router as health_router
        from api.health_advanced import router as health_advanced_router
        app.include_router(health_router)
        app.include_router(health_advanced_router)
    except ImportError:
        try:
            from .api.health import router as health_router
            from .api.health_advanced import router as health_advanced_router
            app.include_router(health_router)
            app.include_router(health_advanced_router)
        except ImportError:
            logger.debug("Health check routers not available")

