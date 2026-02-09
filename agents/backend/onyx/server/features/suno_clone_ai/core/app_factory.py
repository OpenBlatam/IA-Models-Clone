"""
Application factory for creating and configuring the FastAPI application
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from config.settings import settings
from core.app_lifespan import application_lifespan
from core.middleware_setup import setup_middleware
from core.constants import API_ENDPOINTS
from utils.prometheus_metrics import metrics_endpoint

logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=application_lifespan,
        default_response_class=ORJSONResponse,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None
    )
    
    setup_middleware(app)
    
    logger.info("Application created successfully")
    
    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all API routes with the application.
    
    Note: song_api router already includes most routes via router_registry.
    Additional top-level routers are registered here.
    """
    from api.song_api import router as song_router
    from api.websocket_api import router as websocket_router
    from api.batch_api import router as batch_router
    from api.versioning import versions_router
    
    # Main API router (includes all sub-routers via registry)
    app.include_router(song_router)
    
    # Additional top-level routers
    app.include_router(websocket_router)
    app.include_router(batch_router)
    app.include_router(versions_router, prefix="/suno")
    
    # Metrics endpoint
    app.get("/metrics")(metrics_endpoint)
    
    logger.info("All routes registered successfully")


def register_endpoints(app: FastAPI) -> None:
    """
    Register root and health check endpoints.
    """
    from modules.registry import get_module_registry
    
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "endpoints": API_ENDPOINTS
        }
    
    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Quick health check"""
        return {"status": "healthy"}
    
    @app.get("/modules/health")
    async def modules_health() -> Dict[str, Any]:
        """Health check for all modules"""
        registry = get_module_registry()
        return registry.get_health_report()
    
    logger.info("Root and health endpoints registered")

