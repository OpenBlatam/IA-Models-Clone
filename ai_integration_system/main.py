"""
AI Integration System - Main Application Entry Point
FastAPI application with all endpoints and middleware configured
"""

import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .api_endpoints import router
from .advanced_api_endpoints import router as advanced_router
from .integration_engine import integration_engine, initialize_engine
from .workflow_engine import initialize_workflow_engine
from .middleware import (
    RateLimitMiddleware, AuthenticationMiddleware, RequestLoggingMiddleware,
    SecurityHeadersMiddleware, WebhookSignatureMiddleware
)
from .config import settings, get_environment_config

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format=settings.logging.format
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AI Integration System...")
    
    try:
        # Initialize the integration engine
        await initialize_engine()
        logger.info("✅ Integration engine initialized successfully")
        
        # Initialize workflow engine
        initialize_workflow_engine()
        logger.info("✅ Workflow engine initialized successfully")
        
        # Start the engine
        # Note: In production, you might want to run this in a separate process
        # await integration_engine.start_engine()
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize integration engine: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI Integration System...")
        await integration_engine.stop_engine()
        logger.info("✅ Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="AI Integration System",
    description="A comprehensive system for integrating AI-generated content with CMS, CRM, and marketing platforms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Get environment-specific configuration
env_config = get_environment_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=env_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100, burst_size=20)

# Add authentication middleware for production
if settings.environment.value == "production":
    app.add_middleware(AuthenticationMiddleware)

# Add trusted host middleware for production
if settings.environment.value == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {response.status_code} - {process_time:.3f}s")
    
    return response

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Include API routes
app.include_router(router)
app.include_router(advanced_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "AI Integration System",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment.value,
        "available_platforms": integration_engine.get_available_platforms(),
        "docs_url": "/docs",
        "health_url": "/ai-integration/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Import time for middleware
import time
from datetime import datetime

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.logging.level.lower()
    )
