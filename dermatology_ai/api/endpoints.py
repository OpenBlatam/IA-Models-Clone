"""
API endpoints for the application.

Extracted from main.py for better organization and separation of concerns.
"""

import logging
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from config.settings import settings
from core.middleware_config import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)


def register_endpoints(app: FastAPI) -> None:
    """
    Register all API endpoints.
    
    Args:
        app: FastAPI application instance
    """
    app.get("/health")(health_check)
    app.get("/")(root)
    app.get("/metrics")(metrics_endpoint)


async def health_check() -> dict:
    """
    Minimal health check for load balancers and orchestrators.
    
    Returns:
        Dictionary with service health status
    """
    return {"status": "healthy", "service": "dermatology-ai"}


async def root() -> dict:
    """
    Root endpoint with service information.
    
    Returns:
        Dictionary with service metadata and configuration
    """
    return {
        "service": "Dermatology AI",
        "version": "7.1.0",
        "status": "running",
        "environment": settings.environment.value,
        "architecture": "hexagonal",
        "plugins_enabled": True,
        "docs": "/dermatology/docs" if settings.debug else None,
        "health": "/health",
    }


async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus metrics or error message
    """
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    return JSONResponse(
        status_code=503,
        content={"error": "Prometheus not available"}
    )


def register_exception_handler(app: FastAPI) -> None:
    """
    Register global exception handler.
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Global exception handler for unhandled errors.
        
        Args:
            request: FastAPI request object
            exc: Exception that was raised
            
        Returns:
            JSON response with error details
        """
        client_host = request.client.host if request.client else None
        request_id = getattr(request.state, "request_id", None)
        
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={
                "path": request.url.path,
                "method": request.method,
                "client": client_host,
                "request_id": request_id,
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            }
        )






