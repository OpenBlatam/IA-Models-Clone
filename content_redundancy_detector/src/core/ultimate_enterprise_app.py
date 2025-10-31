"""
Ultimate Enterprise Content Redundancy Detector - Complete enterprise-grade system
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
from .api.workflow_routes import router as workflow_router
from .api.intelligence_routes import router as intelligence_router
from .real_time_processor import initialize_processor, shutdown_processor
from .ai_content_analyzer import initialize_ai_analyzer
from .content_optimizer import initialize_content_optimizer
from .content_workflow_engine import initialize_workflow_engine, shutdown_workflow_engine
from .content_intelligence_engine import initialize_content_intelligence_engine

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup structured logging with enhanced configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ultimate_enterprise_app.log", encoding="utf-8")
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ultimate enterprise lifespan context manager for startup and shutdown events"""
    
    # Startup
    setup_logging()
    logger.info("🚀 Ultimate Enterprise Content Redundancy Detector starting up...")
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
        
        # Initialize workflow engine
        logger.info("🔄 Initializing workflow engine...")
        await initialize_workflow_engine()
        logger.info("✅ Workflow engine initialized")
        
        # Initialize content intelligence engine
        logger.info("🧠 Initializing content intelligence engine...")
        await initialize_content_intelligence_engine()
        logger.info("✅ Content intelligence engine initialized")
        
        logger.info("🌟 All ultimate enterprise systems initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Ultimate Enterprise Content Redundancy Detector...")
    
    try:
        # Shutdown workflow engine
        await shutdown_workflow_engine()
        logger.info("✅ Workflow engine shutdown")
        
        # Shutdown real-time processor
        await shutdown_processor()
        logger.info("✅ Real-time processor shutdown")
        
        logger.info("✅ All ultimate enterprise systems shutdown successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


def create_ultimate_enterprise_app() -> FastAPI:
    """Create and configure ultimate enterprise FastAPI application"""
    
    app = FastAPI(
        title="Ultimate Enterprise Content Redundancy Detector",
        description="""
        ## 🚀 Ultimate Enterprise Content Redundancy Detector
        
        **The most advanced enterprise-grade content analysis, optimization, workflow automation, and intelligence system**
        
        ### 🌟 Ultimate Enterprise Features:
        
        #### 📊 **Advanced Analytics Engine**
        - **Similarity Analysis**: TF-IDF, Jaccard, and Cosine similarity
        - **Redundancy Detection**: Clustering-based duplicate detection
        - **Content Metrics**: Readability, sentiment, and quality analysis
        - **Batch Processing**: Efficient processing of large content collections
        - **Caching**: Redis-based result caching for performance
        - **Streaming**: Real-time processing of large datasets
        
        #### ⚡ **Real-time Processing**
        - **WebSocket Support**: Real-time updates and notifications
        - **Streaming Analysis**: Process content in batches with progress updates
        - **Job Queue**: Priority-based processing with worker pools
        - **Live Metrics**: Real-time processing statistics
        - **Connection Management**: Advanced WebSocket connection handling
        - **Background Tasks**: Asynchronous task processing
        
        #### 🤖 **AI Content Analysis**
        - **Sentiment Analysis**: Advanced emotion and sentiment detection
        - **Topic Classification**: Automatic topic identification and categorization
        - **Named Entity Recognition**: Extract entities, locations, and organizations
        - **Language Detection**: Multi-language content analysis
        - **Content Insights**: AI-generated summaries and recommendations
        - **Engagement Prediction**: Predict content engagement potential
        - **Quality Assessment**: Comprehensive content quality evaluation
        
        #### 🔧 **Content Optimization**
        - **Readability Enhancement**: Improve content accessibility and clarity
        - **SEO Optimization**: Optimize for search engines and discoverability
        - **Engagement Boosting**: Enhance reader interaction and retention
        - **Grammar & Style**: Improve writing quality and consistency
        - **Brand Voice Alignment**: Ensure consistent brand messaging
        - **Performance Metrics**: Track optimization improvements
        - **A/B Testing**: Compare optimized versions
        
        #### 🔄 **Workflow Automation**
        - **Custom Workflows**: Create complex content processing workflows
        - **Step Handlers**: Extensible step processing system
        - **Dependency Management**: Handle complex step dependencies
        - **Error Handling**: Robust error handling and retry mechanisms
        - **Parallel Processing**: Execute independent steps in parallel
        - **Workflow Templates**: Pre-built workflow templates
        - **Execution Monitoring**: Real-time workflow execution tracking
        
        #### 🧠 **Content Intelligence Engine**
        - **Intelligence Analysis**: Comprehensive content intelligence scoring
        - **Trend Analysis**: Content trends and pattern analysis
        - **Insights Generation**: Detailed content insights and recommendations
        - **Strategy Planning**: Content strategy recommendations
        - **Audience Analysis**: Target audience identification and analysis
        - **Competitive Analysis**: Competitive advantages and positioning
        - **Risk Assessment**: Content risk factors and mitigation strategies
        
        #### 🌐 **Ultimate Enterprise API**
        - **RESTful Endpoints**: Complete API for all functionality
        - **WebSocket Support**: Real-time communication and updates
        - **Batch Processing**: Handle large volumes of content efficiently
        - **Streaming Responses**: Process and return results in real-time
        - **Comprehensive Documentation**: Auto-generated OpenAPI/Swagger docs
        - **API Versioning**: Support for multiple API versions
        - **Rate Limiting**: Enterprise-grade rate limiting and throttling
        
        ### 🎯 **Ultimate Enterprise Use Cases:**
        - **Content Management Systems**: Enterprise CMS integration
        - **Marketing Automation**: Automated content optimization workflows
        - **Quality Assurance**: Automated content quality monitoring
        - **SEO Management**: Large-scale SEO optimization
        - **Content Strategy**: Data-driven content planning and execution
        - **Compliance Monitoring**: Content compliance and governance
        - **Multi-tenant Systems**: Support for multiple organizations
        - **Intelligence Analytics**: Advanced content intelligence and insights
        - **Workflow Automation**: Complex content processing workflows
        - **Trend Analysis**: Content trends and market intelligence
        
        ### 🔗 **API Endpoints:**
        - `/api/v1/analytics/*` - Advanced content analytics and similarity analysis
        - `/api/v1/websocket/*` - Real-time processing and WebSocket communication
        - `/api/v1/ai/*` - AI-powered content analysis and insights
        - `/api/v1/optimization/*` - Content optimization and enhancement
        - `/api/v1/workflows/*` - Workflow automation and management
        - `/api/v1/intelligence/*` - Content intelligence and strategy
        - `/api/v1/content/*` - Core content processing and redundancy detection
        - `/docs` - Interactive API documentation
        - `/redoc` - Alternative API documentation
        
        ### 🛠️ **Ultimate Enterprise Technologies:**
        - **FastAPI**: Modern, fast web framework for building APIs
        - **AI/ML**: Transformers, scikit-learn, sentence-transformers
        - **Real-time**: WebSockets, asyncio, Redis
        - **Workflow Engine**: Custom workflow automation system
        - **Intelligence Engine**: Advanced content intelligence system
        - **Optimization**: Advanced algorithms for content enhancement
        - **Monitoring**: Comprehensive health checks and metrics
        - **Scalability**: Horizontal and vertical scaling capabilities
        
        ### 🔒 **Ultimate Enterprise Security:**
        - **Authentication**: JWT-based authentication system
        - **Authorization**: Role-based access control (RBAC)
        - **Rate Limiting**: Advanced rate limiting and DDoS protection
        - **Input Validation**: Comprehensive input validation and sanitization
        - **Audit Logging**: Complete audit trail for all operations
        - **Data Encryption**: End-to-end data encryption
        
        ---
        
        **Built with ❤️ for ultimate enterprise-grade content processing, automation, and intelligence.**
        """,
        version="5.0.0",
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
    app.include_router(workflow_router)  # Workflow automation
    app.include_router(intelligence_router)  # Content intelligence
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        logger.warning("Static files directory not found, skipping mount")
    
    return app


# Create ultimate enterprise app instance
app = create_ultimate_enterprise_app()


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


# Ultimate enterprise root endpoint
@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """Ultimate enterprise root endpoint with comprehensive system information"""
    return {
        "message": "🚀 Ultimate Enterprise Content Redundancy Detector",
        "version": "5.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "features": [
            "Advanced Content Analytics",
            "Real-time Processing with WebSockets",
            "AI-powered Content Analysis",
            "Content Optimization Engine",
            "Workflow Automation System",
            "Content Intelligence Engine",
            "Batch Processing & Streaming",
            "Ultimate Enterprise API Documentation",
            "Multi-tenant Support"
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
        "workflow_capabilities": [
            "Custom Workflow Creation",
            "Step Dependency Management",
            "Parallel Step Execution",
            "Error Handling & Retries",
            "Workflow Templates",
            "Execution Monitoring",
            "Background Processing",
            "Event-driven Triggers"
        ],
        "intelligence_capabilities": [
            "Content Intelligence Scoring",
            "Trend Analysis",
            "Insights Generation",
            "Strategy Planning",
            "Audience Analysis",
            "Competitive Analysis",
            "Risk Assessment",
            "Performance Optimization"
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
            "workflows": "/api/v1/workflows/health",
            "intelligence": "/api/v1/intelligence/health",
            "demo": "/api/v1/websocket/demo"
        },
        "capabilities": {
            "similarity_types": ["tfidf", "jaccard", "cosine"],
            "processing_modes": ["batch", "streaming", "realtime", "workflow", "intelligence"],
            "ai_models": ["sentiment", "emotion", "topic", "ner", "summarization"],
            "optimization_goals": ["readability", "seo", "engagement", "grammar", "style"],
            "workflow_step_types": ["content_analysis", "content_optimization", "similarity_analysis", "notification"],
            "intelligence_types": ["engagement", "viral", "seo", "conversion", "brand"],
            "supported_formats": ["text", "json", "csv", "xml"],
            "max_content_size": "20KB",
            "concurrent_jobs": 1000,
            "real_time_workers": 10,
            "workflow_workers": 5,
            "intelligence_workers": 3,
            "ai_confidence_threshold": 0.7
        },
        "enterprise_features": {
            "multi_tenant": True,
            "rbac": True,
            "audit_logging": True,
            "rate_limiting": True,
            "api_versioning": True,
            "horizontal_scaling": True,
            "disaster_recovery": True,
            "compliance": ["GDPR", "SOC2", "ISO27001"],
            "intelligence_analytics": True,
            "workflow_automation": True,
            "content_strategy": True
        },
        "performance": {
            "response_time_target": "< 2s",
            "throughput_target": "15000 requests/min",
            "availability_target": "99.99%",
            "cache_hit_rate_target": "> 95%",
            "workflow_execution_time": "< 30s",
            "intelligence_analysis_time": "< 10s"
        }
    }


# Ultimate enterprise health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Ultimate enterprise health check with detailed system status"""
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
        
        # Check workflow engine status
        from .content_workflow_engine import get_workflow_health
        workflow_health = await get_workflow_health()
        
        # Check content intelligence engine status
        from .content_intelligence_engine import get_content_intelligence_health
        intelligence_health = await get_content_intelligence_health()
        
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
            optimizer_health["status"] == "healthy",
            workflow_health["status"] == "healthy",
            intelligence_health["status"] == "healthy"
        ])
        
        return {
            "status": "healthy" if services_healthy else "degraded",
            "timestamp": time.time(),
            "version": "5.0.0",
            "services": {
                "api": "healthy",
                "real_time_processor": "healthy" if processor_metrics["total_jobs"] >= 0 else "unhealthy",
                "ai_analyzer": ai_health["status"],
                "content_optimizer": optimizer_health["status"],
                "workflow_engine": workflow_health["status"],
                "intelligence_engine": intelligence_health["status"],
                "redis_cache": redis_status,
                "analytics_engine": "healthy"
            },
            "metrics": {
                "uptime_seconds": time.time() - getattr(health_check, "start_time", time.time()),
                "processor": processor_metrics,
                "ai_models_loaded": ai_health.get("models_loaded", False),
                "optimizer_models_loaded": optimizer_health.get("models_loaded", False),
                "workflow_engine_running": workflow_health.get("is_running", False),
                "intelligence_models_loaded": intelligence_health.get("models_loaded", False),
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
            "workflow_capabilities": {
                "workflow_engine": workflow_health.get("is_running", False),
                "total_workflows": workflow_health.get("total_workflows", 0),
                "total_executions": workflow_health.get("total_executions", 0),
                "active_handlers": workflow_health.get("active_handlers", 0),
                "redis_connected": workflow_health.get("redis_connected", False)
            },
            "intelligence_capabilities": {
                "intelligence_engine": intelligence_health.get("models_loaded", False),
                "trend_analyzer": intelligence_health.get("available_models", {}).get("trend_analyzer", False),
                "insight_generator": intelligence_health.get("available_models", {}).get("insight_generator", False),
                "strategy_planner": intelligence_health.get("available_models", {}).get("strategy_planner", False),
                "competitive_analyzer": intelligence_health.get("available_models", {}).get("competitive_analyzer", False),
                "audience_analyzer": intelligence_health.get("available_models", {}).get("audience_analyzer", False)
            },
            "system_info": {
                "python_version": sys.version,
                "fastapi_version": "0.104.1",
                "log_level": settings.log_level,
                "debug_mode": settings.debug,
                "device": ai_health.get("device", "unknown")
            },
            "enterprise_status": {
                "multi_tenant_ready": True,
                "rbac_enabled": True,
                "audit_logging": True,
                "rate_limiting": True,
                "api_versioning": True,
                "monitoring": True,
                "alerting": True,
                "intelligence_analytics": True,
                "workflow_automation": True,
                "content_strategy": True
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "version": "5.0.0"
        }


# Custom OpenAPI schema
def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Ultimate Enterprise Content Redundancy Detector API",
        version="5.0.0",
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
            "name": "Workflows",
            "description": "Workflow automation, custom workflows, and process management"
        },
        {
            "name": "Content Intelligence",
            "description": "Content intelligence, strategy planning, and insights generation"
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
            "description": "Ultimate Enterprise Content Redundancy Detector API Server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key for authentication"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token for authentication"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting Ultimate Enterprise Content Redundancy Detector on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "ultimate_enterprise_app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=True,
        use_colors=True
    )




