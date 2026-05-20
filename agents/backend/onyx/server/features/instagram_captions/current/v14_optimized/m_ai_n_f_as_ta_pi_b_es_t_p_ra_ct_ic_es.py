from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
from models.fastapi_best_practices import (
from routes.fastapi_best_practices import router as best_practices_router
from middleware.fastapi_best_practices import (
from core.optimized_engine import (
from core.exceptions import (
from dependencies import CoreDependencies, AdvancedDependencies
from config import settings
import logging
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices Implementation - Main Application

This module implements a complete FastAPI application following best practices:
- Proper data models with Pydantic v2
- Comprehensive path operations with proper HTTP methods
- Advanced middleware stack for security, monitoring, and performance
- Structured error handling and validation
- Performance optimization and caching
- Security best practices
- Comprehensive documentation and examples
"""



# Import FastAPI best practices components
    CaptionGenerationRequest, CaptionGenerationResponse,
    BatchCaptionRequest, BatchCaptionResponse,
    UserPreferences, ErrorResponse, HealthResponse,
    CaptionAnalytics, ServiceStatus, ErrorDetail
)


    create_middleware_stack, create_cors_middleware,
    create_gzip_middleware, get_request_id, get_processing_time
)

# Import core components
    engine, OptimizedRequest, OptimizedResponse,
    generate_caption_optimized, generate_batch_captions_optimized,
    get_engine_stats
)

    ValidationError, AIGenerationError, CacheError,
    handle_validation_error, handle_ai_error, handle_cache_error
)

# Import dependencies

# Import configuration


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper resource management"""
    # Startup
    logger.info("🚀 Starting Instagram Captions API with FastAPI Best Practices")
    
    try:
        # Initialize AI engine
        await engine._initialize_models()
        logger.info("✅ AI engine initialized")
        
        # Initialize other services
        logger.info("✅ All services initialized")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Instagram Captions API")
        
        try:
            # Cleanup engine
            await engine.cleanup()
            logger.info("✅ AI engine cleaned up")
            
            # Cleanup other services
            logger.info("✅ All services cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI application with best practices
app = FastAPI(
    title="Instagram Captions API - FastAPI Best Practices",
    description="""
    Ultra-optimized Instagram captions generation API implementing FastAPI best practices.
    
    ## Features
    
    * **Advanced Data Models**: Pydantic v2 models with comprehensive validation
    * **Proper HTTP Methods**: RESTful API design with correct status codes
    * **Security Middleware**: Comprehensive security headers and rate limiting
    * **Performance Monitoring**: Request tracking and performance metrics
    * **Error Handling**: Structured error responses with proper HTTP status codes
    * **Caching**: Multi-level caching for optimal performance
    * **Documentation**: Comprehensive OpenAPI documentation with examples
    
    ## Authentication
    
    This API uses API key authentication. Include your API key in the header:
    `Authorization: Bearer your-api-key-here`
    
    ## Rate Limiting
    
    * 100 requests per minute per user
    * 1000 requests per hour per user
    
    ## Error Handling
    
    All errors return structured JSON responses with:
    * Error type and message
    * Request ID for tracking
    * Timestamp
    * Additional details when available
    """,
    version="14.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "email": "support@instagramcaptions.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.instagramcaptions.com", "description": "Production server"},
    ]
)


# =============================================================================
# CUSTOM OPENAPI SCHEMA
# =============================================================================

def custom_openapi():
    """Custom OpenAPI schema with additional information"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom components
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "API key for authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"]
)

# Add GZip middleware for compression
app.add_middleware(create_gzip_middleware())

# Add custom middleware stack
middleware_stack = create_middleware_stack()
for middleware in middleware_stack:
    app.add_middleware(middleware)


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors with structured response"""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        error="validation_error",
        message=exc.message,
        details=[
            ErrorDetail(
                field=exc.field,
                message=exc.message,
                code="VALIDATION_ERROR"
            )
        ],
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump()
    )


@app.exception_handler(AIGenerationError)
async def ai_generation_exception_handler(request: Request, exc: AIGenerationError):
    """Handle AI generation errors"""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        error="ai_generation_error",
        message="AI service temporarily unavailable",
        details=[
            ErrorDetail(
                message=exc.message,
                code="AI_SERVICE_ERROR"
            )
        ],
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_response.model_dump()
    )


@app.exception_handler(CacheError)
async def cache_exception_handler(request: Request, exc: CacheError):
    """Handle cache errors"""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        error="cache_error",
        message="Cache service error",
        details=[
            ErrorDetail(
                message=exc.message,
                code="CACHE_ERROR"
            )
        ],
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response"""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        error="http_error",
        message=exc.detail,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = get_request_id(request)
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        details=[
            ErrorDetail(
                message=str(exc),
                code="INTERNAL_ERROR"
            )
        ],
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


# =============================================================================
# INCLUDE ROUTERS
# =============================================================================

# Include best practices router
app.include_router(best_practices_router)


# =============================================================================
# ROOT AND HEALTH ENDPOINTS
# =============================================================================

@app.get(
    "/",
    summary="API Information",
    description="Get API information and available endpoints",
    response_description="API information and feature list"
)
async def root():
    """Root endpoint with comprehensive API information"""
    return {
        "message": "Instagram Captions API - FastAPI Best Practices",
        "version": "14.0.0",
        "status": "running",
        "features": [
            "Pydantic v2 data models with comprehensive validation",
            "RESTful API design with proper HTTP methods",
            "Advanced middleware stack for security and monitoring",
            "Structured error handling with proper status codes",
            "Performance optimization with caching and async operations",
            "Comprehensive OpenAPI documentation with examples",
            "Rate limiting and security headers",
            "Request tracking and performance metrics"
        ],
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "health_check": "/health",
            "captions": "/api/v14/captions",
            "analytics": "/api/v14/analytics"
        },
        "contact": {
            "email": "support@instagramcaptions.com",
            "documentation": "https://docs.instagramcaptions.com"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Comprehensive health check for all services",
    response_description="Health status of all services with detailed metrics"
)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check AI engine
        engine_stats = get_engine_stats()
        ai_engine_healthy = engine_stats.get("status", "unknown") == "healthy"
        
        # Create service statuses
        services = {
            "ai_engine": ServiceStatus(
                service="ai_engine",
                status="healthy" if ai_engine_healthy else "unhealthy",
                version="1.0.0",
                uptime=engine_stats.get("uptime", 0.0)
            ),
            "api": ServiceStatus(
                service="api",
                status="healthy",
                version="14.0.0",
                uptime=time.time()  # Simplified for demo
            ),
            "database": ServiceStatus(
                service="database",
                status="healthy",  # Simplified for demo
                version="1.0.0",
                uptime=3600.0
            ),
            "cache": ServiceStatus(
                service="cache",
                status="healthy",  # Simplified for demo
                version="1.0.0",
                uptime=3600.0
            )
        }
        
        # Determine overall status
        overall_status = "healthy"
        if any(service.status == "unhealthy" for service in services.values()):
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            version="14.0.0",
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="14.0.0",
            services={}
        )


@app.get(
    "/metrics",
    summary="Performance Metrics",
    description="Get detailed performance metrics and statistics",
    response_description="Comprehensive performance metrics"
)
async def get_metrics():
    """Get detailed performance metrics"""
    try:
        # Get engine stats
        engine_stats = get_engine_stats()
        
        # Get processing time from request state (simplified)
        processing_time = time.time()
        
        return {
            "timestamp": time.time(),
            "engine_stats": engine_stats,
            "performance": {
                "average_response_time": 1.23,  # Simplified for demo
                "requests_per_second": 45.67,
                "error_rate": 0.02,
                "cache_hit_rate": 0.85
            },
            "system": {
                "memory_usage": "256MB",
                "cpu_usage": "15%",
                "active_connections": 25
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving metrics"
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get(
    "/debug/request-info",
    summary="Debug Request Information",
    description="Get detailed information about the current request (for debugging)",
    response_description="Request details and headers"
)
async def debug_request_info(request: Request):
    """Debug endpoint to show request information"""
    return {
        "request_id": get_request_id(request),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": {
            "host": request.client.host if request.client else None,
            "port": request.client.port if request.client else None
        },
        "query_params": dict(request.query_params),
        "path_params": dict(request.path_params)
    }


@app.get(
    "/debug/performance",
    summary="Debug Performance Information",
    description="Get performance information for the current request",
    response_description="Performance metrics and timing information"
)
async def debug_performance(request: Request):
    """Debug endpoint to show performance information"""
    processing_time = get_processing_time(request)
    
    return {
        "request_id": get_request_id(request),
        "processing_time": processing_time,
        "performance_metrics": {
            "is_slow_request": processing_time > 5.0,
            "performance_level": "excellent" if processing_time < 1.0 else "good" if processing_time < 3.0 else "slow"
        }
    }


# =============================================================================
# MIDDLEWARE FOR PROCESSING TIME
# =============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    response.headers["X-Processing-Time"] = str(round(processing_time, 3))
    
    return response


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main_fastapi_best_practices:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 