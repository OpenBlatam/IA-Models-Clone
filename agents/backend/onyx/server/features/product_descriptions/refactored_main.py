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
from contextlib import asynccontextmanager
from typing import Dict, Any
import structlog
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from .config_refactored import config
from .services_refactored import services
    from .routers_refactored import (
    import subprocess
        from .schemas_refactored import ProductCreateRequest, AIDescriptionRequest
        from .schemas_refactored import ProductSearchRequest
    import sys
from typing import Any, List, Dict, Optional
import logging
"""
Refactored Main Application
==========================

Clean Architecture FastAPI application with proper separation of concerns.
Professional, maintainable, and production-ready implementation.
"""




# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# MIDDLEWARE - Cross-cutting concerns
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware with structured logging."""
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
start_time = time.time()
        
        # Log request
        logger.info("Request started", 
                   method=request.method,
                   path=request.url.path,
                   query_params=str(request.query_params))
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info("Request completed",
                       method=request.method,
                       path=request.url.path,
                       status_code=response.status_code,
                       duration_ms=round(duration * 1000, 2))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Request failed",
                        method=request.method,
                        path=request.url.path,
                        error=str(e),
                        duration_ms=round(duration * 1000, 2))
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if config.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware (demo implementation)."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        
    """__init__ function."""
super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
if not config.rate_limit_enabled:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < 60  # Keep requests from last minute
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "retry_after": 60}
            )
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        return await call_next(request)


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting Refactored Product API...")
    
    try:
        # Initialize services
        await _startup_services()
        logger.info("✅ All services initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down services...")
        await _shutdown_services()
        logger.info("✅ Shutdown completed")


async def _startup_services():
    """Initialize application services."""
    # Initialize cache service
    cache_service = await services.get_cache_service()
    logger.info("Cache service initialized")
    
    # Initialize other services
    await services.get_product_repository()
    await services.get_product_service()
    await services.get_ai_service()
    await services.get_health_service()
    
    logger.info("All services ready")


async def _shutdown_services():
    """Cleanup application services."""
    await services.cleanup()


def create_refactored_app() -> FastAPI:
    """Create refactored FastAPI application."""
    
    # Create app with lifespan
    app = FastAPI(
        title=config.name,
        description="Refactored Product API with Clean Architecture",
        version=config.version,
        lifespan=lifespan,
        docs_url=config.docs_url if not config.is_production else None,
        redoc_url="/redoc" if not config.is_production else None,
        openapi_url="/openapi.json" if not config.is_production else None,
    )
    
    # Add middleware (order matters - last added runs first)
    _setup_middleware(app)
    
    # Add routers
    _setup_routers(app)
    
    # Add exception handlers
    _setup_exception_handlers(app)
    
    logger.info("Refactored FastAPI application created")
    return app


def _setup_middleware(app: FastAPI):
    """Setup application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.is_development else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=config.requests_per_minute)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware configured")


def _setup_routers(app: FastAPI):
    """Setup application routers."""
    
    # Import routers (inline to avoid circular imports)
        products_router,
        ai_router,
        health_router,
        admin_router
    )
    
    # Add routers with prefix
    app.include_router(health_router, prefix="")  # No prefix for health
    app.include_router(products_router, prefix=config.api_prefix)
    app.include_router(ai_router, prefix=config.api_prefix)
    
    # Admin router only in development
    if config.is_development:
        app.include_router(admin_router, prefix="")
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "service": config.name,
            "version": config.version,
            "environment": config.environment,
            "status": "operational",
            "docs": config.docs_url,
            "health": "/health",
            "api": config.api_prefix
        }
    
    logger.info("Routers configured")


def _setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.warning("Validation error",
                      path=request.url.path,
                      errors=exc.errors())
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.info("HTTP exception",
                   path=request.url.path,
                   status_code=exc.status_code,
                   detail=exc.detail)
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error("Unexpected exception",
                    path=request.url.path,
                    error=str(exc),
                    exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "path": str(request.url.path)
            }
        )
    
    logger.info("Exception handlers configured")


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

# Create the application instance
app = create_refactored_app()


# =============================================================================
# CLI COMMANDS
# =============================================================================

def run_development():
    """Run development server."""
    logger.info("Starting development server...")
    
    uvicorn.run(
        "refactored_main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.lower(),
        access_log=True,
        use_colors=not config.is_production
    )


def run_production():
    """Run production server with Gunicorn."""
    
    logger.info("Starting production server...")
    
    cmd = [
        "gunicorn",
        "refactored_main:app",
        "-w", "4",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{config.host}:{config.port}",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", config.log_level.lower(),
        "--timeout", "120",
        "--keep-alive", "5",
        "--max-requests", "1000",
        "--max-requests-jitter", "100"
    ]
    
    subprocess.run(cmd)


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

async def demo_refactored_api():
    """Demonstrate the refactored API functionality."""
    print("\n🏗️ REFACTORED API DEMO")
    print("=====================")
    
    # Initialize services
    await _startup_services()
    
    try:
        # Get services
        product_service = await services.get_product_service()
        ai_service = await services.get_ai_service()
        health_service = await services.get_health_service()
        
        print("\n✅ Services initialized")
        
        # Test product creation
        
        product_request = ProductCreateRequest(
            name="Refactored Widget Pro",
            sku="REF-WIDGET-001",
            description="A professionally refactored widget",
            base_price=199.99,
            quantity=50,
            tags=["refactored", "professional", "widget"]
        )
        
        product = await product_service.create_product(product_request)
        print(f"\n📦 Product created: {product.name} (ID: {product.id})")
        
        # Test AI description
        ai_request = AIDescriptionRequest(
            product_name="Refactored Widget Pro",
            features=["clean architecture", "maintainable code", "high performance"],
            tone="professional"
        )
        
        ai_description = await ai_service.generate_description(ai_request)
        print(f"\n🤖 AI Description: {ai_description.description}")
        
        # Test health check
        health = await health_service.get_health_status()
        print(f"\n🏥 Health Status: {health.status}")
        print(f"   Services: {health.services}")
        
        # Test search
        
        search_request = ProductSearchRequest(query="widget", per_page=5)
        results = await product_service.search_products(search_request)
        print(f"\n🔍 Search Results: {results.total} products found")
        
        print("\n✅ Refactored API demo completed successfully!")
        
    finally:
        await _shutdown_services()


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "dev":
            run_development()
        elif command == "prod":
            run_production()
        elif command == "demo":
            asyncio.run(demo_refactored_api())
        else:
            print("Commands: dev, prod, demo")
    else:
        # Default to development
        print("🚀 Refactored Product API")
        print("========================")
        print("Features:")
        print("  ✅ Clean Architecture")
        print("  ✅ Separation of Concerns")
        print("  ✅ Professional Code Structure")
        print("  ✅ Comprehensive Error Handling")
        print("  ✅ Structured Logging")
        print("  ✅ Type Safety")
        print("  ✅ Production Ready")
        print("")
        print("Usage:")
        print("  python refactored_main.py dev   - Development server")
        print("  python refactored_main.py prod  - Production server")
        print("  python refactored_main.py demo  - Run demo")
        print("")
        
        run_development() 