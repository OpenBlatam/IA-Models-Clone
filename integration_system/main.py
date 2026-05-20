"""
Integration System - Main Application
=====================================

Unified system that integrates all features: Content Redundancy Detector, BUL, Gamma App, Business Agents, and Export IA.
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import asyncio
from datetime import datetime

# Import all feature systems
from ..content_redundancy_detector.app import app as content_redundancy_app
from ..bulk.bul_realistic import app as bulk_app
from ..gamma_app.api.main import app as gamma_app
from ..business_agents.main import app as business_agents_app
from ..export_ia.main import app as export_ia_app

# Import integration components
from .core.integration_manager import IntegrationManager
from .api.gateway_router import router as gateway_router
from .api.health_router import router as health_router
from .api.metrics_router import router as metrics_router
from .config.settings import get_settings
from .middleware.cors_middleware import setup_cors
from .middleware.auth_middleware import AuthMiddleware
from .middleware.rate_limit_middleware import RateLimitMiddleware
from .services.health_service import HealthService
from .services.metrics_service import MetricsService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global instances
integration_manager = None
health_service = None
metrics_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global integration_manager, health_service, metrics_service
    
    # Startup
    logger.info("Starting Integration System...")
    
    try:
        # Initialize integration manager
        integration_manager = IntegrationManager()
        await integration_manager.initialize()
        
        # Initialize services
        health_service = HealthService(integration_manager)
        metrics_service = MetricsService(integration_manager)
        
        # Store in app state
        app.state.integration_manager = integration_manager
        app.state.health_service = health_service
        app.state.metrics_service = metrics_service
        
        logger.info("Integration System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Integration System: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Integration System...")
        if integration_manager:
            await integration_manager.shutdown()

# Create FastAPI application
app = FastAPI(
    title="Blatam Academy Integration System",
    description="Unified system integrating all features: Content Redundancy Detector, BUL, Gamma App, Business Agents, and Export IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup middleware
setup_cors(app)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": datetime.now().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all integrated systems."""
    try:
        if not health_service:
            raise HTTPException(status_code=503, detail="Health service not initialized")
        
        health_status = await health_service.get_comprehensive_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with comprehensive system information."""
    return {
        "name": "Blatam Academy Integration System",
        "version": "1.0.0",
        "description": "Unified system integrating all business features",
        "integrated_systems": [
            {
                "name": "Content Redundancy Detector",
                "status": "integrated",
                "endpoint": "/api/v1/content-redundancy",
                "description": "Detects redundancy in content and analyzes similarity"
            },
            {
                "name": "BUL (Business Unlimited)",
                "status": "integrated", 
                "endpoint": "/api/v1/bul",
                "description": "AI-powered document generation for SMEs"
            },
            {
                "name": "Gamma App",
                "status": "integrated",
                "endpoint": "/api/v1/gamma",
                "description": "AI-powered content generation system"
            },
            {
                "name": "Business Agents",
                "status": "integrated",
                "endpoint": "/api/v1/business-agents",
                "description": "Comprehensive agent system for business areas"
            },
            {
                "name": "Export IA",
                "status": "integrated",
                "endpoint": "/api/v1/export-ia",
                "description": "Advanced document export and analytics"
            }
        ],
        "features": [
            "Unified API Gateway",
            "Cross-system Integration",
            "Real-time Health Monitoring",
            "Comprehensive Metrics",
            "Authentication & Authorization",
            "Rate Limiting",
            "Request Routing",
            "Error Handling",
            "Logging & Monitoring"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "gateway": "/api/v1/gateway",
            "metrics": "/api/v1/metrics",
            "systems": "/api/v1/systems"
        }
    }

# Include routers
app.include_router(gateway_router, prefix="/api/v1/gateway")
app.include_router(health_router, prefix="/api/v1/health")
app.include_router(metrics_router, prefix="/api/v1/metrics")

# Mount sub-applications
app.mount("/api/v1/content-redundancy", content_redundancy_app)
app.mount("/api/v1/bul", bulk_app)
app.mount("/api/v1/gamma", gamma_app)
app.mount("/api/v1/business-agents", business_agents_app)
app.mount("/api/v1/export-ia", export_ia_app)

# Mount static files if they exist
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "integration_system.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

