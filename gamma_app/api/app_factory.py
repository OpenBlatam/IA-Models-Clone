"""
Application Factory
Factory function to create and configure the FastAPI application
"""

import logging
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from .lifespan import lifespan
from .middleware import setup_middleware
from .rate_limiting import setup_rate_limiting
from .router_registry import setup_routers
from .endpoints import setup_base_endpoints
from .exceptions import setup_exception_handlers
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="AI-Powered Content Generation System",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
        default_response_class=ORJSONResponse
    )
    
    try:
        setup_middleware(app)
        setup_rate_limiting(app)
        setup_routers(app)
        setup_base_endpoints(app)
        setup_exception_handlers(app)
        logger.info("Application configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure application: {e}", exc_info=True)
        raise
    
    return app

