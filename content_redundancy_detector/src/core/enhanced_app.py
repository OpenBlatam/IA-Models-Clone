"""
Enhanced Content Redundancy Detector - Main FastAPI Application
Following best practices: functional programming, RORO pattern, async operations
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from config import settings
from middleware import (
    LoggingMiddleware, ErrorHandlingMiddleware, 
    SecurityMiddleware, PerformanceMiddleware
)
from routers import router
from .api.analytics_routes import router as analytics_router
from .api.websocket_routes import router as websocket_router
from .real_time_processor import initialize_processor, shutdown_processor

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup structured logging with enhanced configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", encoding="utf-8")
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Enhanced lifespan context manager for startup and shutdown events"""
    
    # Startup
    setup_logging()
    logger.info("🚀 Enhanced Content Redundancy Detector starting up...")
    logger.info(f"Server will run on {settings.host}:{settings.port}")
    
    try:
        # Initialize real-time processor
        await initialize_processor()
        logger.info("✅ Real-time processor initialized")
        
        # Initialize other services here
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Content Redundancy Detector...")
    
    try:
        # Shutdown real-time processor
        await shutdown_processor()
        logger.info("✅ Real-time processor shutdown")
        
        # Shutdown other services here
        logger.info("✅ All services shutdown successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


def create_enhanced_app() -> FastAPI:
    """Create and configure enhanced FastAPI application"""
    
    app = FastAPI(
        title="Enhanced Content Redundancy Detector",
        description="""
        ## 🚀 Enhanced Content Redundancy Detector
        
        **Advanced content analysis system with real-time processing capabilities**
        
        ### 🌟 Features:
        
        #### 📊 **Advanced Analytics**
        - **Similarity Analysis**: TF-IDF, Jaccard, and Cosine similarity
        - **Redundancy Detection**: Clustering-based duplicate detection
        - **Content Metrics**: Readability, sentiment, and quality analysis
        - **Batch Processing**: Efficient processing of large content collections
        
        #### ⚡ **Real-time Processing**
        - **WebSocket Support**: Real-time updates and notifications
        - **Streaming Analysis**: Process content in batches with progress updates
        - **Job Queue**: Priority-based processing with worker pools
        - **Live Metrics**: Real-time processing statistics
        
        #### 🔧 **Core Functionality**
        - **Content Deduplication**: Hash-based and similarity-based detection
        - **Text Analysis**: Feature extraction and content profiling
        - **Caching**: Redis-based result caching for performance
        - **API Documentation**: Comprehensive OpenAPI/Swagger documentation
        
        ### 🎯 **Use Cases:**
        - Content management systems
        - Plagiarism detection
        - SEO content optimization
        - Document management
        - Quality assurance workflows
        
        ### 🔗 **API Endpoints:**
        - `/api/v1/analytics/*` - Advanced content analytics
        - `/api/v1/websocket/*` - Real-time processing and updates
        - `/api/v1/content/*` - Core content processing
        - `/docs` - Interactive API documentation
        
        ---
        
        **Built with ❤️ using FastAPI, following functional programming best practices.**
        """,
        version="2.0.0",
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware (order matters - last added is first executed)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Include routers
    app.include_router(router)  # Core content processing
    app.include_router(analytics_router)  # Advanced analytics
    app.include_router(websocket_router)  # Real-time processing
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        logger.warning("Static files directory not found, skipping mount")
    
    return app


# Create enhanced app instance
app = create_enhanced_app()


# Enhanced global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Enhanced HTTP exception handler with detailed logging"""
    logger.warning(f"HTTP error: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Enhanced global exception handler with comprehensive error tracking"""
    logger.error(f"Unexpected error: {exc} - Path: {request.url.path}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Internal server error",
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None),
            "error_type": exc.__class__.__name__,
        }
    )


# Enhanced root endpoint
@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """Enhanced root endpoint with comprehensive system information"""
    return {
        "message": "🚀 Enhanced Content Redundancy Detector",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "features": [
            "Advanced Content Analytics",
            "Real-time Processing",
            "WebSocket Support",
            "Batch Processing",
            "Content Deduplication",
            "Similarity Analysis",
            "Performance Monitoring",
            "Comprehensive API"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "analytics": "/api/v1/analytics/health",
            "websocket": "/api/v1/websocket/health",
            "demo": "/api/v1/websocket/demo"
        },
        "capabilities": {
            "similarity_types": ["tfidf", "jaccard", "cosine"],
            "processing_modes": ["batch", "streaming", "realtime"],
            "supported_formats": ["text", "json", "csv"],
            "max_content_size": "100KB",
            "concurrent_jobs": 1000,
            "real_time_workers": 10
        }
    }


# Enhanced health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Enhanced health check with detailed system status"""
    try:
        # Check real-time processor status
        from .real_time_processor import get_processor_metrics
        processor_metrics = await get_processor_metrics()
        
        # Check Redis connection
        redis_status = "unknown"
        try:
            from .advanced_analytics import get_redis_client
            redis_client = await get_redis_client()
            await redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "api": "healthy",
                "real_time_processor": "healthy" if processor_metrics["total_jobs"] >= 0 else "unhealthy",
                "redis_cache": redis_status,
                "analytics_engine": "healthy"
            },
            "metrics": {
                "uptime_seconds": time.time() - getattr(health_check, "start_time", time.time()),
                "processor": processor_metrics,
                "active_connections": processor_metrics.get("active_connections", 0)
            },
            "system_info": {
                "python_version": sys.version,
                "fastapi_version": "0.104.1",
                "log_level": settings.log_level,
                "debug_mode": settings.debug
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "version": "2.0.0"
        }


# Custom OpenAPI schema
def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Enhanced Content Redundancy Detector API",
        version="2.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "System",
            "description": "System health and information endpoints"
        },
        {
            "name": "Analytics",
            "description": "Advanced content analytics and similarity analysis"
        },
        {
            "name": "WebSocket",
            "description": "Real-time processing and WebSocket endpoints"
        },
        {
            "name": "Content",
            "description": "Core content processing and redundancy detection"
        }
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://{settings.host}:{settings.port}",
            "description": "Development server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting Enhanced Content Redundancy Detector on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "enhanced_app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=True,
        use_colors=True
    )




