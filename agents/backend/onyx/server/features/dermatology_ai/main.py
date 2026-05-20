"""
Dermatology AI - FastAPI Server Optimized for Microservices & Serverless.

Version: 7.1.0 - Fully Modular Architecture with Plugin System
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
from fastapi import FastAPI

from config.settings import settings
from core.startup import initialize_application, shutdown_application
from core.middleware_config import configure_middleware
from api.endpoints import register_endpoints, register_exception_handler
from utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging(log_level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    
    Fully modular with plugin system and lazy loading.
    Optimized for serverless environments with minimal cold start time.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None: Control flow for application lifecycle
    """
    await initialize_application(app)
    yield
    await shutdown_application()


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Dermatology AI API",
        description=(
            "Sistema de análisis de piel y recomendaciones de skincare - "
            "Arquitectura Hexagonal v7.1 con Plugin System"
        ),
        version="7.1.0",
        lifespan=lifespan,
        docs_url="/dermatology/docs" if settings.debug else None,
        redoc_url="/dermatology/redoc" if settings.debug else None,
        openapi_url="/dermatology/openapi.json" if settings.debug else None,
    )
    
    configure_middleware(app)
    register_exception_handler(app)
    register_endpoints(app)
    
    return app


# Create application instance
app = create_application()


def main() -> None:
    """Main entry point for running the application."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        loop="uvloop",
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )


if __name__ == "__main__":
    main()
