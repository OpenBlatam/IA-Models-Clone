from .settings import settings
from .ultimate_quantum_ai_app import create_app

app = create_app(settings)

"""
Business Agents System - Main Application
=========================================

FastAPI application entry point for the Business Agents system.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Dict, Any

from .api import agents_router, workflows_router, documents_router, system_router
from .config import config, is_development, is_production
from .business_agents import BusinessAgentManager
from .web_interface import get_index_html_path, get_static_files_path
from .middleware import RequestIDMiddleware, LoggingMiddleware, setup_cors_middleware, SecurityMiddleware
from .services import HealthService, SystemInfoService, MetricsService
from .docs.openapi import get_openapi_schema
from .docs.tags import get_tags_metadata

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format=config.log_format,
    filename=config.log_file if config.log_file else None
)

logger = logging.getLogger(__name__)

# Global business agent manager instance
agent_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_manager
    
    # Startup
    logger.info("Starting Business Agents System...")
    
    try:
        # Initialize business agent manager
        agent_manager = BusinessAgentManager()
        app.state.agent_manager = agent_manager
        
        # Initialize services
        app.state.health_service = HealthService(agent_manager)
        app.state.system_info_service = SystemInfoService(agent_manager)
        app.state.metrics_service = MetricsService(agent_manager)
        
        logger.info("Business Agents System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Business Agents System: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Business Agents System...")

# Create FastAPI application
app = FastAPI(
    title="Business Agents System",
    description="Comprehensive agent system for all business areas with workflow management and document generation",
    version="1.0.0",
    docs_url="/docs" if is_development() else None,
    redoc_url="/redoc" if is_development() else None,
    lifespan=lifespan,
    openapi_tags=get_tags_metadata()
)

# Override OpenAPI schema with enhanced version
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi_schema(app)
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Setup middleware
setup_cors_middleware(app)
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

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
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        health_service = app.state.health_service
        result = await health_service.get_health_status()
        
        if result["status"] == "healthy":
            return result
        else:
            return JSONResponse(
                status_code=503,
                content=result
            )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Business Agents System",
        "version": "1.0.0",
        "description": "Comprehensive agent system for all business areas",
        "features": [
            "Workflow Management",
            "Document Generation",
            "Agent Coordination",
            "Real-time Execution",
            "Multi-business Area Support"
        ],
        "business_areas": [
            "marketing",
            "sales", 
            "operations",
            "hr",
            "finance",
            "legal",
            "technical",
            "content"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs" if is_development() else "disabled",
            "api": "/business-agents"
        }
    }

# Include all routers
app.include_router(agents_router, prefix="/business-agents")
app.include_router(workflows_router, prefix="/business-agents")
app.include_router(documents_router, prefix="/business-agents")
app.include_router(system_router, prefix="/business-agents")

# Mount static files
app.mount("/static", StaticFiles(directory=get_static_files_path()), name="static")

# Serve web interface
@app.get("/", include_in_schema=False)
async def serve_web_interface():
    """Serve the web interface."""
    return FileResponse(get_index_html_path())

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "business_agents.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=is_development(),
        log_level=config.log_level.lower(),
        access_log=True
    )
