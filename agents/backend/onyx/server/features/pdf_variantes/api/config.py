"""
PDF Variantes API - Application Configuration
Centralized configuration for API setup
"""

from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from .middleware import (
    setup_cors_middleware,
    setup_trusted_host_middleware,
    rate_limit_middleware,
    request_logging_middleware,
    performance_monitoring_middleware,
    security_middleware,
)
from .optimized_middleware import setup_optimized_middleware
from .dependencies import get_services


def create_app_config() -> Dict[str, Any]:
    """Get application configuration"""
    return {
        "title": "PDF Variantes API",
        "description": "Advanced PDF processing with AI capabilities, real-time collaboration, and enterprise features",
        "version": "2.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
    }


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application - ULTRA-OPTIMIZED"""
    # Ultra-optimized middleware stack (minimal overhead, maximum performance)
    optimize_config = {
        "max_requests": 100,
        "window_seconds": 60,
        "compress_min_size": 500,
        "compress_level": 6,
        "log_slow_only": True,  # Only log slow requests for performance
        "slow_threshold": 1.0
    }
    setup_optimized_middleware(app, optimize_config)
    
    # Security middleware (first)
    setup_trusted_host_middleware(app)
    
    # CORS middleware (early, before other middleware)
    setup_cors_middleware(app)
    
    # Optional: Full request logging (commented out for performance)
    # Uncomment only if detailed logging is needed:
    # app.middleware("http")(request_logging_middleware)


def setup_middleware_with_services(app: FastAPI) -> None:
    """Setup middleware that requires services (after services are initialized)"""
    # Performance monitoring (requires performance_service)
    async def perf_middleware(request, call_next):
        try:
            services = get_services()
        except:
            services = {}
        return await performance_monitoring_middleware(request, call_next, services)
    
    app.middleware("http")(perf_middleware)
    
    # Security middleware (requires security_service)
    async def sec_middleware(request, call_next):
        try:
            services = get_services()
        except:
            services = {}
        return await security_middleware(request, call_next, services)
    
    app.middleware("http")(sec_middleware)


def create_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Create custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="PDF Variantes API",
        version="2.0.0",
        description="""
        Advanced PDF processing API with AI capabilities, real-time collaboration, and enterprise features.
        
        ## Features
        - **PDF Processing**: Upload, validate, and process PDF documents
        - **AI-Powered Variants**: Generate multiple variants of PDFs using AI
        - **Topic Extraction**: Extract and analyze topics from PDFs
        - **Brainstorming**: Generate brainstorming ideas from PDF content
        - **Real-time Collaboration**: WebSocket-based collaboration
        - **Export Options**: Export to multiple formats (PDF, DOCX, TXT, HTML, JSON, etc.)
        - **Analytics**: Comprehensive analytics and reporting
        - **Search**: Full-text search across documents
        
        ## Authentication
        Most endpoints require authentication via JWT token in the Authorization header.
        
        ## Error Handling
        All errors follow a consistent format with error codes:
        - `PDF_INVALID_FORMAT`: Invalid PDF format
        - `PDF_TOO_LARGE`: File exceeds maximum size
        - `SERVICE_UNAVAILABLE`: Service temporarily unavailable
        - `RATE_LIMIT_EXCEEDED`: Too many requests
        
        ## Rate Limiting
        API requests are rate-limited per user/IP. Default: 100 requests per minute.
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "PDF Variantes API Support",
        "email": "support@pdfvariantes.com"
    }
    
    # Add error codes documentation
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["x-code-samples"] = {
        "errorResponse": {
            "success": False,
            "error": {
                "code": "PDF_INVALID_FORMAT",
                "message": "Invalid PDF format",
                "timestamp": "2024-01-01T00:00:00Z",
                "details": {}
            },
            "request_id": "uuid-here"
        }
    }
    
    # Add servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.yourdomain.com", "description": "Production server"},
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return openapi_schema

