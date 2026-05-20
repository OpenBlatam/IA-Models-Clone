from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from .shared.config import get_settings
from .shared.logging import setup_logging
from .shared.metrics import setup_metrics
from .shared.tracing import setup_tracing
from .infrastructure.caching import setup_cache, cleanup_cache
from .infrastructure.messaging import setup_event_bus, cleanup_event_bus
from .presentation.api.routes import create_api_routes
from .presentation.middleware import create_middleware_stack
from typing import Any, List, Dict, Optional
import logging
"""
AI Video System - Refactored Main Application
============================================

Main entry point for the refactored AI Video system with clean architecture.
"""





@asynccontextmanager
async def create_lifespan_manager(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.logging)
    
    # Setup metrics
    setup_metrics(settings.metrics)
    
    # Setup tracing
    setup_tracing(settings.tracing)
    
    # Initialize infrastructure
    await setup_cache(settings.cache)
    await setup_event_bus(settings.messaging)
    
    # Store in app state
    app.state.settings = settings
    
    logger = app.state.logger
    logger.info("AI Video System started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Video System")
    
    await cleanup_cache()
    await cleanup_event_bus()
    
    logger.info("AI Video System shutdown complete")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Video System - Refactored",
        version="3.0.0",
        description="Modern AI Video Generation System with Clean Architecture",
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
        default_response_class=ORJSONResponse,
        lifespan=create_lifespan_manager,
    )
    
    # Configure middleware
    create_middleware_stack(app)
    
    # Include API routes
    create_api_routes(app)
    
    return app


# Create application instance
app = create_application()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Video System - Refactored",
        "version": "3.0.0",
        "status": "running",
        "architecture": "clean-architecture",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "3.0.0",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        loop="uvloop",
        http="httptools",
    ) 