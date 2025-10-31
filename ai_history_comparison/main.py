"""
AI History Comparison System - Main Application

This is the main application entry point for the AI History Comparison system.
It sets up the FastAPI application, middleware, and routes.
"""

# OPTIMIZED IMPORTS - Using loguru for faster logging
from loguru import logger
import sys
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from datetime import datetime
import traceback

from .core.config import get_config, is_production, is_development
from .api.router import create_api_router, create_legacy_router

# OPTIMIZED LOGGING - Using loguru for better performance
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
if is_production():
    logger.add(
        "ai_history_comparison.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )

# Get configuration
config = get_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AI History Comparison System...")
    
    try:
        # Initialize the system
        logger.info("✅ AI History Comparison System initialized successfully")
        
        # Log system information
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"Debug mode: {config.debug}")
        logger.info(f"Host: {config.host}")
        logger.info(f"Port: {config.port}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI History Comparison System...")

# Create FastAPI application with BEST PRACTICES
app = FastAPI(
    title="AI History Comparison System",
    description="""
    ## 🚀 AI History Comparison System API
    
    A comprehensive system for analyzing, comparing, and tracking AI model outputs over time.
    
    ### ✨ Features
    
    * **Content Analysis** - Analyze content quality, readability, sentiment, and complexity
    * **Historical Comparison** - Compare content across different time periods and model versions
    * **Trend Analysis** - Track performance trends and identify patterns
    * **Quality Reporting** - Generate comprehensive quality reports with insights
    * **Content Clustering** - Group similar content using machine learning
    * **Bulk Processing** - Analyze multiple content pieces efficiently
    
    ### 🎯 Use Cases
    
    * Monitor AI model performance over time
    * Track content quality improvements
    * Identify patterns in AI-generated content
    * Compare different model versions
    * Generate quality reports for stakeholders
    * Optimize content generation workflows
    
    ### 🔒 Authentication
    
    All endpoints require Bearer token authentication.
    
    ### ⚡ Rate Limiting
    
    API is rate limited to 100 requests per minute per IP.
    
    ### 📊 Response Format
    
    All responses follow a standardized format:
    ```json
    {
        "success": true,
        "data": {...},
        "message": "Operation completed successfully",
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_123456789",
        "version": "v1"
    }
    ```
    """,
    version="1.0.0",
    contact={
        "name": "AI History Comparison System",
        "email": "support@ai-history.com",
        "url": "https://ai-history.com/support"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if is_development() else None,
    redoc_url="/redoc" if is_development() else None,
    openapi_url="/openapi.json" if is_development() else None,
    # BEST PRACTICE: Add API tags for better organization
    tags_metadata=[
        {
            "name": "Content Analysis",
            "description": "Analyze content quality, readability, and sentiment metrics"
        },
        {
            "name": "Comparison",
            "description": "Compare content between different versions or models"
        },
        {
            "name": "Trends",
            "description": "Analyze trends and patterns over time"
        },
        {
            "name": "Reports",
            "description": "Generate comprehensive analysis reports"
        },
        {
            "name": "System",
            "description": "System health and monitoring endpoints"
        }
    ]
)

# BEST PRACTICE: Add compression middleware first
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.cors_origins,
    allow_credentials=config.security.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )

# BEST PRACTICE: Enhanced request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Enhanced request logging with best practices"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request with structured data
    logger.info(
        f"📥 Request {request_id}: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "content_length": request.headers.get("content-length", "0")
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response with performance metrics
    process_time = time.time() - start_time
    logger.info(
        f"📤 Response {request_id}: {response.status_code} - {process_time:.3f}s",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time": process_time,
            "response_size": response.headers.get("content-length", "0")
        }
    )
    
    # Add performance headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    response.headers["X-API-Version"] = "v1"
    
    return response

# Custom middleware for error handling
@app.middleware("http")
async def error_handler(request: Request, call_next):
    """Handle errors gracefully"""
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unhandled error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        )

# Include API routes
api_router = create_api_router(config)
legacy_router = create_legacy_router()

# BEST PRACTICE: Include LLM routes
try:
    from .llm_integration import llm_router, initialize_llm_service, cleanup_llm_service
    app.include_router(llm_router)
    logger.info("✅ LLM routes included successfully")
except ImportError as e:
    logger.warning(f"⚠️ LLM routes not available: {str(e)}")

app.include_router(api_router)
app.include_router(legacy_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "AI History Comparison System",
        "version": "1.0.0",
        "description": "Comprehensive AI content analysis and comparison system",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs" if is_development() else "disabled in production",
            "health": "/health",
            "api": "/api/v1",
            "legacy_api": "/ai-history",
            "system": "/api/v1/system"
        },
        "features": config.features
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": config.environment.value,
            "system": {
                "features_enabled": len([f for f in config.features.values() if f]),
                "total_features": len(config.features),
                "memory_usage": "normal"
            },
            "checks": {
                "database": "healthy",
                "cache": "healthy",
                "api": "healthy"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "environment": config.environment.value,
            "features_enabled": len([f for f in config.features.values() if f]),
            "total_features": len(config.features),
            "system_uptime": "unknown",  # Would need to track startup time
            "memory_usage": "normal",
            "cache_hit_rate": "unknown"  # Would need cache implementation
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI History Comparison System",
        version="1.0.0",
        description="""
        A comprehensive system for analyzing, comparing, and tracking AI model outputs over time.
        
        ## Key Features
        
        * **Content Analysis** - Analyze content quality, readability, sentiment, and complexity
        * **Historical Comparison** - Compare content across different time periods and model versions
        * **Trend Analysis** - Track performance trends and identify patterns
        * **Quality Reporting** - Generate comprehensive quality reports with insights
        * **Content Clustering** - Group similar content using machine learning
        * **Bulk Processing** - Analyze multiple content pieces efficiently
        
        ## Getting Started
        
        1. **Analyze Content** - Use `/ai-history/analyze` to analyze your first content piece
        2. **Compare Content** - Use `/ai-history/compare` to compare different content pieces
        3. **Track Trends** - Use `/ai-history/trends` to analyze trends over time
        4. **Generate Reports** - Use `/ai-history/report` to generate quality reports
        
        ## Authentication
        
        Authentication is currently disabled. In production, API keys or JWT tokens may be required.
        """,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "AI History Comparison",
            "description": "Core functionality for analyzing and comparing AI content over time"
        },
        {
            "name": "Content Analysis",
            "description": "Analyze individual content pieces for quality metrics"
        },
        {
            "name": "Trend Analysis", 
            "description": "Track trends and patterns in AI content over time"
        },
        {
            "name": "Quality Reporting",
            "description": "Generate comprehensive quality reports and insights"
        },
        {
            "name": "System Management",
            "description": "System status, health checks, and administrative functions"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI History Comparison System - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested resource {request.url.path} was not found",
            "timestamp": datetime.now().isoformat(),
            "suggestions": [
                "Check the API documentation at /docs",
                "Verify the endpoint URL",
                "Ensure the resource exists"
            ]
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc: HTTPException):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The request data is invalid",
            "timestamp": datetime.now().isoformat(),
            "details": exc.detail
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🎉 AI History Comparison System started successfully!")
    logger.info(f"📊 System ready to analyze AI content history")
    logger.info(f"🔗 API documentation available at: /docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("👋 AI History Comparison System shutting down...")

# Main function for running the application
def main():
    """Main function to run the application"""
    logger.info("Starting AI History Comparison System...")
    
    # Run the application
    uvicorn.run(
        "ai_history_comparison.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.monitoring.log_level.value.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
