"""
Ultimate Content Redundancy Detector - Complete AI-powered system
Following FastAPI best practices: functional programming, RORO pattern, async operations
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
from .api.ai_routes import router as ai_router
from .api.optimization_routes import router as optimization_router
from .real_time_processor import initialize_processor, shutdown_processor
from .ai_content_analyzer import initialize_ai_analyzer
from .content_optimizer import initialize_content_optimizer

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup structured logging with enhanced configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ultimate_app.log", encoding="utf-8")
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ultimate lifespan context manager for startup and shutdown events"""
    
    # Startup
    setup_logging()
    logger.info("🚀 Ultimate Content Redundancy Detector starting up...")
    logger.info(f"Server will run on {settings.host}:{settings.port}")
    
    try:
        # Initialize real-time processor
        logger.info("⚡ Initializing real-time processor...")
        await initialize_processor()
        logger.info("✅ Real-time processor initialized")
        
        # Initialize AI content analyzer
        logger.info("🤖 Initializing AI content analyzer...")
        await initialize_ai_analyzer()
        logger.info("✅ AI content analyzer initialized")
        
        # Initialize content optimizer
        logger.info("🔧 Initializing content optimizer...")
        await initialize_content_optimizer()
        logger.info("✅ Content optimizer initialized")
        
        logger.info("🌟 All AI systems initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Ultimate Content Redundancy Detector...")
    
    try:
        # Shutdown real-time processor
        await shutdown_processor()
        logger.info("✅ Real-time processor shutdown")
        
        logger.info("✅ All systems shutdown successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


def create_ultimate_app() -> FastAPI:
    """Create and configure ultimate FastAPI application"""
    
    app = FastAPI(
        title="Ultimate Content Redundancy Detector",
        description="""
        ## 🚀 Ultimate Content Redundancy Detector
        
        **The most advanced AI-powered content analysis and optimization system**
        
        ### 🌟 Core Features:
        
        #### 📊 **Advanced Analytics Engine**
        - **Similarity Analysis**: TF-IDF, Jaccard, and Cosine similarity
        - **Redundancy Detection**: Clustering-based duplicate detection
        - **Content Metrics**: Readability, sentiment, and quality analysis
        - **Batch Processing**: Efficient processing of large content collections
        - **Caching**: Redis-based result caching for performance
        
        #### ⚡ **Real-time Processing**
        - **WebSocket Support**: Real-time updates and notifications
        - **Streaming Analysis**: Process content in batches with progress updates
        - **Job Queue**: Priority-based processing with worker pools
        - **Live Metrics**: Real-time processing statistics
        - **Connection Management**: Advanced WebSocket connection handling
        
        #### 🤖 **AI Content Analysis**
        - **Sentiment Analysis**: Advanced emotion and sentiment detection
        - **Topic Classification**: Automatic topic identification and categorization
        - **Named Entity Recognition**: Extract entities, locations, and organizations
        - **Language Detection**: Multi-language content analysis
        - **Content Insights**: AI-generated summaries and recommendations
        - **Engagement Prediction**: Predict content engagement potential
        
        #### 🔧 **Content Optimization**
        - **Readability Enhancement**: Improve content accessibility and clarity
        - **SEO Optimization**: Optimize for search engines and discoverability
        - **Engagement Boosting**: Enhance reader interaction and retention
        - **Grammar & Style**: Improve writing quality and consistency
        - **Brand Voice Alignment**: Ensure consistent brand messaging
        - **Performance Metrics**: Track optimization improvements
        
        #### 🌐 **Comprehensive API**
        - **RESTful Endpoints**: Complete API for all functionality
        - **WebSocket Support**: Real-time communication and updates
        - **Batch Processing**: Handle large volumes of content efficiently
        - **Streaming Responses**: Process and return results in real-time
        - **Comprehensive Documentation**: Auto-generated OpenAPI/Swagger docs
        
        ### 🎯 **Use Cases:**
        - **Content Management**: Detect duplicates and optimize content quality
        - **SEO Optimization**: Improve search rankings and visibility
        - **Content Marketing**: Enhance engagement and conversion rates
        - **Quality Assurance**: Ensure consistent, high-quality content
        - **Plagiarism Detection**: Identify similar or duplicate content
        - **Content Strategy**: Data-driven content planning and optimization
        
        ### 🔗 **API Endpoints:**
        - `/api/v1/analytics/*` - Advanced content analytics and similarity analysis
        - `/api/v1/websocket/*` - Real-time processing and WebSocket communication
        - `/api/v1/ai/*` - AI-powered content analysis and insights
        - `/api/v1/optimization/*` - Content optimization and enhancement
        - `/api/v1/content/*` - Core content processing and redundancy detection
        - `/docs` - Interactive API documentation
        - `/redoc` - Alternative API documentation
        
        ### 🛠️ **Technologies:**
        - **FastAPI**: Modern, fast web framework for building APIs
        - **AI/ML**: Transformers, scikit-learn, sentence-transformers
        - **Real-time**: WebSockets, asyncio, Redis
        - **Optimization**: Advanced algorithms for content enhancement
        - **Monitoring**: Comprehensive health checks and metrics
        
        ---
        
        **Built with ❤️ using the most advanced AI and web technologies.**
        """,
        version="3.0.0",
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
    
    # Include all routers
    app.include_router(router)  # Core content processing
    app.include_router(analytics_router)  # Advanced analytics
    app.include_router(websocket_router)  # Real-time processing
    app.include_router(ai_router)  # AI content analysis
    app.include_router(optimization_router)  # Content optimization
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        logger.warning("Static files directory not found, skipping mount")
    
    return app


# Create ultimate app instance
app = create_ultimate_app()


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


# Ultimate root endpoint
@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """Ultimate root endpoint with comprehensive system information"""
    return {
        "message": "🚀 Ultimate Content Redundancy Detector",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "features": [
            "Advanced Content Analytics",
            "Real-time Processing with WebSockets",
            "AI-powered Content Analysis",
            "Content Optimization Engine",
            "Batch Processing & Streaming",
            "Comprehensive API Documentation",
            "Performance Monitoring",
            "Multi-language Support"
        ],
        "ai_capabilities": [
            "Sentiment Analysis",
            "Topic Classification",
            "Named Entity Recognition",
            "Language Detection",
            "Content Summarization",
            "Engagement Prediction",
            "SEO Optimization",
            "Readability Enhancement"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "analytics": "/api/v1/analytics/health",
            "websocket": "/api/v1/websocket/health",
            "ai": "/api/v1/ai/health",
            "optimization": "/api/v1/optimization/health",
            "demo": "/api/v1/websocket/demo"
        },
        "capabilities": {
            "similarity_types": ["tfidf", "jaccard", "cosine"],
            "processing_modes": ["batch", "streaming", "realtime"],
            "ai_models": ["sentiment", "emotion", "topic", "ner", "summarization"],
            "optimization_goals": ["readability", "seo", "engagement", "grammar", "style"],
            "supported_formats": ["text", "json", "csv"],
            "max_content_size": "20KB",
            "concurrent_jobs": 1000,
            "real_time_workers": 10,
            "ai_confidence_threshold": 0.7
        },
        "performance": {
            "response_time_target": "< 2s",
            "throughput_target": "1000 requests/min",
            "availability_target": "99.9%",
            "cache_hit_rate_target": "> 80%"
        }
    }


# Ultimate health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Ultimate health check with detailed system status"""
    try:
        # Check real-time processor status
        from .real_time_processor import get_processor_metrics
        processor_metrics = await get_processor_metrics()
        
        # Check AI analyzer status
        from .ai_content_analyzer import get_ai_analyzer_health
        ai_health = await get_ai_analyzer_health()
        
        # Check content optimizer status
        from .content_optimizer import get_content_optimizer_health
        optimizer_health = await get_content_optimizer_health()
        
        # Check Redis connection
        redis_status = "unknown"
        try:
            from .advanced_analytics import get_redis_client
            redis_client = await get_redis_client()
            await redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "disconnected"
        
        # Calculate overall health
        services_healthy = all([
            processor_metrics["total_jobs"] >= 0,
            ai_health["status"] == "healthy",
            optimizer_health["status"] == "healthy"
        ])
        
        return {
            "status": "healthy" if services_healthy else "degraded",
            "timestamp": time.time(),
            "version": "3.0.0",
            "services": {
                "api": "healthy",
                "real_time_processor": "healthy" if processor_metrics["total_jobs"] >= 0 else "unhealthy",
                "ai_analyzer": ai_health["status"],
                "content_optimizer": optimizer_health["status"],
                "redis_cache": redis_status,
                "analytics_engine": "healthy"
            },
            "metrics": {
                "uptime_seconds": time.time() - getattr(health_check, "start_time", time.time()),
                "processor": processor_metrics,
                "ai_models_loaded": ai_health.get("models_loaded", False),
                "optimizer_models_loaded": optimizer_health.get("models_loaded", False),
                "active_connections": processor_metrics.get("active_connections", 0)
            },
            "ai_capabilities": {
                "sentiment_analysis": ai_health.get("available_models", {}).get("sentiment_analyzer", False),
                "emotion_analysis": ai_health.get("available_models", {}).get("emotion_analyzer", False),
                "topic_classification": ai_health.get("available_models", {}).get("topic_classifier", False),
                "named_entity_recognition": ai_health.get("available_models", {}).get("ner_pipeline", False),
                "text_summarization": ai_health.get("available_models", {}).get("summarizer", False),
                "language_detection": ai_health.get("available_models", {}).get("language_detector", False)
            },
            "optimization_capabilities": {
                "grammar_checking": optimizer_health.get("available_models", {}).get("grammar_checker", False),
                "style_improvement": optimizer_health.get("available_models", {}).get("style_improver", False),
                "seo_optimization": optimizer_health.get("available_models", {}).get("seo_optimizer", False),
                "readability_enhancement": optimizer_health.get("available_models", {}).get("readability_enhancer", False),
                "engagement_boosting": optimizer_health.get("available_models", {}).get("engagement_booster", False)
            },
            "system_info": {
                "python_version": sys.version,
                "fastapi_version": "0.104.1",
                "log_level": settings.log_level,
                "debug_mode": settings.debug,
                "device": ai_health.get("device", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "version": "3.0.0"
        }


# Custom OpenAPI schema
def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Ultimate Content Redundancy Detector API",
        version="3.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "System",
            "description": "System health, information, and monitoring endpoints"
        },
        {
            "name": "Analytics",
            "description": "Advanced content analytics, similarity analysis, and redundancy detection"
        },
        {
            "name": "WebSocket",
            "description": "Real-time processing, WebSocket communication, and live updates"
        },
        {
            "name": "AI Analysis",
            "description": "AI-powered content analysis, sentiment analysis, and insights generation"
        },
        {
            "name": "Content Optimization",
            "description": "Content optimization, SEO analysis, and enhancement recommendations"
        },
        {
            "name": "Content",
            "description": "Core content processing, redundancy detection, and basic analysis"
        }
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://{settings.host}:{settings.port}",
            "description": "Ultimate Content Redundancy Detector API Server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key for authentication"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting Ultimate Content Redundancy Detector on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "ultimate_app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=True,
        use_colors=True
    )




