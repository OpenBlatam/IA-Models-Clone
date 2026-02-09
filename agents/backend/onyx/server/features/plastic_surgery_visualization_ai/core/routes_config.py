"""Routes configuration."""

from fastapi import FastAPI

from api.routes import visualization, health, metrics, comparison, batch, info
from core.constants import API_VERSION, SERVICE_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_routes(app: FastAPI) -> None:
    """Register API routes."""
    logger.info("Setting up routes...")
    
    # Health endpoints
    app.include_router(health.router, prefix="/health", tags=["Health"])
    
    # API v1 endpoints
    app.include_router(
        visualization.router,
        prefix="/api/v1",
        tags=["Visualization"]
    )
    app.include_router(
        comparison.router,
        prefix="/api/v1",
        tags=["Comparison"]
    )
    app.include_router(
        batch.router,
        prefix="/api/v1",
        tags=["Batch Processing"]
    )
    app.include_router(
        metrics.router,
        prefix="/api/v1/metrics",
        tags=["Metrics"]
    )
    app.include_router(
        info.router,
        prefix="/api/v1",
        tags=["Information"]
    )
    
    # Root endpoint
    setup_root_endpoint(app)
    
    logger.info("Routes setup complete")


def setup_root_endpoint(app: FastAPI) -> None:
    """Setup root endpoint."""
    @app.get("/")
    async def root():
        return {
            "service": SERVICE_NAME,
            "version": API_VERSION,
            "status": "operational",
            "endpoints": {
                "visualize": "/api/v1/visualize",
                "compare": "/api/v1/compare",
                "batch": "/api/v1/batch",
                "surgery_types": "/api/v1/surgery-types",
                "health": "/health/",
                "metrics": "/api/v1/metrics/",
                "docs": "/docs"
            }
        }

