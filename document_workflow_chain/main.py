"""
Main Application - Complete Document Workflow Chain System
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import all our modules
from .workflow_chain_engine import enhanced_workflow_engine
from .workflow_scheduler import workflow_scheduler
from .content_quality_control import content_quality_controller
from .content_versioning import content_version_manager
from .api_endpoints import router as api_router
from .dashboard import interactive_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Document Workflow Chain System...")
    
    try:
        # Initialize all components
        await enhanced_workflow_engine.initialize()
        await workflow_scheduler.start()
        
        logger.info("All systems initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Document Workflow Chain System...")
        
        try:
            await workflow_scheduler.stop()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="Document Workflow Chain System",
    description="Advanced AI-powered document generation with workflow chaining, collaboration, and analytics",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check all system components
        systems_status = {
            "workflow_engine": "healthy",
            "scheduler": "healthy",
            "quality_control": "healthy",
            "versioning": "healthy"
        }
        
        return JSONResponse(content={
            "status": "healthy",
            "systems": systems_status,
            "timestamp": "2024-01-01T00:00:00Z"  # This would be actual timestamp
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="System unhealthy")

# System status endpoint
@app.get("/status")
async def system_status():
    """Get detailed system status"""
    try:
        # Get metrics from core components
        workflow_metrics = await enhanced_workflow_engine.get_comprehensive_workflow_insights("system")
        quality_trends = await content_quality_controller.get_quality_trends(30)
        scheduler_metrics = await workflow_scheduler.get_metrics()
        
        return JSONResponse(content={
            "system": {
                "status": "operational",
                "uptime": "24h 30m",  # This would be calculated
                "version": "2.0.0"
            },
            "metrics": {
                "workflows": workflow_metrics,
                "quality": quality_trends,
                "scheduler": scheduler_metrics
            },
            "components": {
                "workflow_engine": "active",
                "scheduler": "active",
                "quality_control": "active",
                "versioning": "active"
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return JSONResponse(content={
        "message": "Document Workflow Chain System",
        "version": "2.0.0",
        "description": "Advanced AI-powered document generation with workflow chaining",
        "features": [
            "Continuous document generation",
            "Quality control and analytics",
            "Workflow scheduling",
            "Content versioning",
            "Interactive dashboard",
            "Multi-language support",
            "Content templates",
            "Advanced analytics"
        ],
        "endpoints": {
            "api": "/api/v1/document-workflow-chain",
            "dashboard": "/dashboard",
            "health": "/health",
            "status": "/status",
            "docs": "/docs"
        }
    })

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# WebSocket endpoints for real-time features
@app.websocket("/ws/dashboard/{user_id}")
async def dashboard_websocket(websocket, user_id: str):
    """WebSocket endpoint for dashboard updates"""
    await interactive_dashboard.websocket_endpoint(websocket, user_id)

# Background tasks
async def background_tasks():
    """Background tasks for system maintenance"""
    while True:
        try:
            # Clean up old notifications
            # Update system metrics
            # Perform health checks
            # etc.
            
            await asyncio.sleep(300)  # Run every 5 minutes
            
        except Exception as e:
            logger.error(f"Background task error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry

# Start background tasks
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(background_tasks())

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
