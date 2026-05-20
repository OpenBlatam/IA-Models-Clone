from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from .middleware.error_middleware import create_error_handler
from .middleware.logging_middleware import create_logging_middleware
from .middleware.performance_middleware import create_performance_middleware
from .middleware.security_middleware import create_security_middleware
from .routers import health_router, video_router, metrics_router, template_router
from .services.cache_service import initialize_cache, cleanup_cache
from .services.monitoring_service import initialize_monitoring, cleanup_monitoring
from .utils.config import get_settings
from typing import Any, List, Dict, Optional
import logging
"""
Modern FastAPI Application for AI Video System
=============================================

Clean, scalable FastAPI implementation following best practices:
- Functional programming with minimal classes
- RORO pattern (Receive Object, Return Object)
- Type hints for all functions
- Early error returns
- Async/await optimization
- Pydantic v2 models
- Performance-first architecture
"""





@asynccontextmanager
async def create_lifespan_manager(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event decorators.
    """
    # Startup
    settings = get_settings()
    
    # Initialize services
    await initialize_cache(settings.cache)
    await initialize_monitoring(settings.monitoring)
    
    # Store in app state for access
    app.state.settings = settings
    
    yield
    
    # Shutdown
    await cleanup_cache()
    await cleanup_monitoring()


def create_middleware_stack(app: FastAPI) -> None:
    """Configure middleware stack with proper ordering."""
    settings = get_settings()
    
    # CORS (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Security middleware
    app.middleware("http")(create_security_middleware())
    
    # Performance monitoring
    app.middleware("http")(create_performance_middleware())
    
    # Logging middleware
    app.middleware("http")(create_logging_middleware())
    
    # Error handling (innermost)
    app.exception_handler(Exception)(create_error_handler())


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Video API with Templates & Avatars",
        version="2.0.0",
        description="Modern AI Video Generation API with Template Selection and AI Avatars",
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
        default_response_class=ORJSONResponse,  # Faster JSON serialization
        lifespan=create_lifespan_manager,
    )
    
    # Configure middleware
    create_middleware_stack(app)
    
    # Include routers
    app.include_router(health_router.router, prefix="/api/v1", tags=["Health"])
    app.include_router(video_router.router, prefix="/api/v1", tags=["Video"])
    app.include_router(template_router.router, prefix="/api/v1", tags=["Templates & Avatars"])
    app.include_router(metrics_router.router, prefix="/api/v1", tags=["Metrics"])
    
    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    uvicorn.run(
        "improved_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        loop="uvloop",  # Performance optimization
        http="httptools",  # Performance optimization
    ) 