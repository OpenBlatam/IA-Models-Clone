"""Optimized main application following FastAPI best practices."""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Core dependencies
from .enhanced_config import get_ultra_fast_config
from .routers import optimized_router
from .enhanced_dependencies import get_config, get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    start_time = time.time()
    
    # Startup
    logger.info("🚀 Starting Optimized PDF Variantes API")
    
    # Load configuration
    config = get_ultra_fast_config()
    app.state.config = config
    app.state.start_time = start_time
    
    logger.info(f"📋 Configuration loaded for {config.environment.value} environment")
    logger.info(f"🔧 Features enabled: {sum(config.features.values())}/{len(config.features)}")
    
    yield
    
    # Shutdown
    uptime = time.time() - start_time
    logger.info(f"🛑 Shutting down Optimized PDF Variantes API (uptime: {uptime:.2f}s)")


def create_app() -> FastAPI:
    """Create optimized FastAPI application."""
    config = get_ultra_fast_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="Optimized PDF Variantes API",
        description="High-performance PDF processing with functional patterns",
        version="4.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include optimized router
    app.include_router(optimized_router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with proper logging."""
        logger.error(
            f"Unhandled exception: {exc}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": type(exc).__name__
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["System"], summary="Health Check")
    async def health_check():
        """System health check."""
        config = get_ultra_fast_config()
        uptime = time.time() - app.state.start_time
        
        return {
            "status": "healthy",
            "service": "optimized-pdf-variantes",
            "version": "4.0.0",
            "environment": config.environment.value,
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "enabled": sum(config.features.values()),
                "total": len(config.features)
            }
        }
    
    # Root endpoint
    @app.get("/", tags=["System"], summary="API Information")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Optimized PDF Variantes API",
            "version": "4.0.0",
            "description": "High-performance PDF processing with functional patterns",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "pdf": "/pdf"
            },
            "features": [
                "Optimized PDF Processing",
                "Functional Programming Patterns",
                "Early Error Handling",
                "Async Operations",
                "Intelligent Caching",
                "Batch Processing"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Configuration endpoint
    @app.get("/config", tags=["System"], summary="Get Configuration")
    async def get_configuration(
        config = Depends(get_config),
        current_user = Depends(get_current_user)
    ):
        """Get current configuration."""
        config_dict = config.to_dict()
        
        # Remove sensitive information
        sensitive_keys = ["api_key", "password", "secret_key"]
        for key in sensitive_keys:
            if key in str(config_dict):
                config_dict = str(config_dict).replace(key, "***")
        
        return config_dict
    
    # Metrics endpoint
    @app.get("/metrics", tags=["System"], summary="System Metrics")
    async def get_metrics():
        """Get system metrics."""
        uptime = time.time() - app.state.start_time
        
        return {
            "uptime_seconds": uptime,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational"
        }
    
    # Request logging middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Request logging middleware."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        # Add response headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-API-Version"] = "4.0.0"
        
        return response
    
    return app


# Create app instance
app = create_app()

# Export for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
