"""Application factory."""

from fastapi import FastAPI

from core.lifespan import lifespan
from core.middleware_config import setup_middleware
from core.exceptions_config import setup_exception_handlers
from core.routes_config import setup_routes
from core.constants import API_VERSION, SERVICE_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    logger.info("Creating FastAPI application...")
    
    app_config = {
        "title": f"{SERVICE_NAME} API",
        "description": "AI system that visualizes how you'll look after plastic surgery procedures",
        "version": API_VERSION,
        "lifespan": lifespan,
    }
    
    app = FastAPI(**app_config)
    
    # Setup components
    setup_middleware(app)
    setup_exception_handlers(app)
    setup_routes(app)
    
    logger.info("FastAPI application created successfully")
    
    return app

