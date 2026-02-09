"""API Routes and FastAPI App"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .openapi import custom_openapi
from .middleware_enhanced import TimingMiddleware, RequestIDMiddleware
from ..middleware import error_handler_middleware, logging_middleware
from ...config.app_settings import get_settings

def create_app() -> FastAPI:
    """Create FastAPI application with all middleware."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="API for quality control and defect detection",
        version=settings.app_version,
        debug=settings.debug,
    )
    
    # Custom OpenAPI schema
    app.openapi = custom_openapi
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add enhanced middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    
    # Add custom middleware
    app.middleware("http")(error_handler_middleware)
    app.middleware("http")(logging_middleware)
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Setup logging if configured
    if settings.log_file:
        from ...infrastructure.logging import setup_logging
        setup_logging(
            level=settings.log_level,
            format=settings.log_format,
            file=settings.log_file
        )
    
    return app

__all__ = ["create_app", "router"]

