"""
Export IA - Main Application
============================

Advanced document export and analytics system with AI-powered content analysis.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Dict, Any
import os
from pathlib import Path

# Import core components
from .core.export_engine import ExportEngine
from .analytics.content_analytics import ContentAnalyticsEngine
from .quality.quality_validator import QualityValidator
from .api.routes import router as api_router
from .config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global instances
export_engine = None
analytics_engine = None
quality_validator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global export_engine, analytics_engine, quality_validator
    
    # Startup
    logger.info("Starting Export IA System...")
    
    try:
        # Initialize core components
        export_engine = ExportEngine()
        analytics_engine = ContentAnalyticsEngine()
        quality_validator = QualityValidator()
        
        # Store in app state
        app.state.export_engine = export_engine
        app.state.analytics_engine = analytics_engine
        app.state.quality_validator = quality_validator
        
        logger.info("Export IA System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Export IA System: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Export IA System...")

# Create FastAPI application
app = FastAPI(
    title="Export IA",
    description="Advanced document export and analytics system with AI-powered content analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

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
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "Export IA",
            "version": "1.0.0",
            "components": {
                "export_engine": "healthy",
                "analytics_engine": "healthy",
                "quality_validator": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Export IA",
        "version": "1.0.0",
        "description": "Advanced document export and analytics system",
        "features": [
            "Document Export",
            "Content Analytics",
            "Quality Validation",
            "AI-Powered Analysis",
            "Multi-Format Support",
            "Real-time Processing"
        ],
        "supported_formats": [
            "PDF", "DOCX", "HTML", "Markdown", "JSON", "XML"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "api": "/api/v1"
        }
    }

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Mount static files if they exist
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "export_ia.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


