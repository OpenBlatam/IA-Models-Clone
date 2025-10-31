from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import structlog
from ..utils.error_system import (
from .notifications.api import router as notifications_router
from .integrated.api import router as integrated_router
from .professional_documents.api import router as professional_documents_router
from typing import Any, List, Dict, Optional
"""
FastAPI Application Factory - Onyx Integration
Centralized application factory for creating optimized FastAPI applications.
"""



    error_factory,
    ErrorContext,
    ValidationError,
    SystemError,
    handle_errors,
    ErrorCategory
)

logger = structlog.get_logger(__name__)

class FastAPIConfig(BaseSettings):
    """Configuration for FastAPI applications."""
    
    # Application settings
    title: str = Field(default="Blatam Academy API", description="Application title")
    version: str = Field(default="1.0.0", description="Application version")
    description: str = Field(default="Blatam Academy Backend API", description="Application description")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Documentation
    docs_url: Optional[str] = Field(default="/docs", description="Swagger docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_default: str = Field(default="100/minute", description="Default rate limit")
    
    # Compression
    compression_enabled: bool = Field(default=True, description="Enable response compression")
    compression_min_size: int = Field(default=1000, description="Minimum size for compression")
    
    # Monitoring
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    
    # Security
    security_headers_enabled: bool = Field(default=True, description="Enable security headers")
    
    class Config:
        env_prefix = "FASTAPI_"
        case_sensitive = False

class FastAPIFactory:
    """Factory for creating optimized FastAPI applications."""
    
    def __init__(self, config: Optional[FastAPIConfig] = None):
        
    """__init__ function."""
self.config = config or FastAPIConfig()
        self.rate_limiter = None
        self.startup_events: List[Callable] = []
        self.shutdown_events: List[Callable] = []
        self.routers: List[Dict[str, Any]] = []
        
    def add_startup_event(self, event: Callable) -> 'FastAPIFactory':
        """Add a startup event handler."""
        self.startup_events.append(event)
        return self
    
    def add_shutdown_event(self, event: Callable) -> 'FastAPIFactory':
        """Add a shutdown event handler."""
        self.shutdown_events.append(event)
        return self
    
    def add_router(self, router, prefix: str = "", tags: Optional[List[str]] = None) -> 'FastAPIFactory':
        """Add a router to the application."""
        self.routers.append({
            "router": router,
            "prefix": prefix,
            "tags": tags or []
        })
        return self
    
    def create_lifespan(self) -> Callable:
        """Create lifespan context manager for the application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            
    """lifespan function."""
# Startup
            logger.info("Starting FastAPI application", 
                       title=self.config.title, 
                       version=self.config.version)
            
            try:
                # Run startup events
                for event in self.startup_events:
                    if asyncio.iscoroutinefunction(event):
                        await event()
                    else:
                        event()
                
                logger.info("FastAPI application started successfully")
                yield
                
            except Exception as e:
                logger.error("Error during startup", error=str(e))
                raise
            finally:
                # Shutdown
                logger.info("Shutting down FastAPI application")
                
                try:
                    # Run shutdown events
                    for event in self.shutdown_events:
                        if asyncio.iscoroutinefunction(event):
                            await event()
                        else:
                            event()
                    
                    logger.info("FastAPI application shut down successfully")
                    
                except Exception as e:
                    logger.error("Error during shutdown", error=str(e))
        
        return lifespan
    
    def setup_middleware(self, app: FastAPI) -> None:
        """Setup middleware for the application."""
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=self.config.cors_credentials,
            allow_methods=self.config.cors_methods,
            allow_headers=self.config.cors_headers,
        )
        
        # Compression middleware
        if self.config.compression_enabled:
            app.add_middleware(
                GZipMiddleware,
                minimum_size=self.config.compression_min_size
            )
        
        # Security headers middleware
        if self.config.security_headers_enabled:
            @app.middleware("http")
            async def add_security_headers(request: Request, call_next):
                
    """add_security_headers function."""
response = await call_next(request)
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                return response
    
    def setup_rate_limiting(self, app: FastAPI) -> None:
        """Setup rate limiting for the application."""
        if not self.config.rate_limit_enabled:
            return
        
        self.rate_limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = self.rate_limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    def setup_monitoring(self, app: FastAPI) -> None:
        """Setup monitoring and metrics for the application."""
        if not self.config.monitoring_enabled:
            return
        
        # Prometheus metrics
        Instrumentator().instrument(app).expose(app)
        
        # Request logging middleware
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            
    """log_requests function."""
start_time = asyncio.get_event_loop().time()
            
            # Log request
            logger.info("Incoming request",
                       method=request.method,
                       url=str(request.url),
                       client_ip=request.client.host if request.client else None)
            
            try:
                response = await call_next(request)
                
                # Log response
                process_time = asyncio.get_event_loop().time() - start_time
                logger.info("Request completed",
                           method=request.method,
                           url=str(request.url),
                           status_code=response.status_code,
                           process_time=process_time)
                
                return response
                
            except Exception as e:
                # Log error
                process_time = asyncio.get_event_loop().time() - start_time
                logger.error("Request failed",
                            method=request.method,
                            url=str(request.url),
                            error=str(e),
                            process_time=process_time)
                raise
    
    def setup_error_handlers(self, app: FastAPI) -> None:
        """Setup error handlers for the application."""
        
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Handle Pydantic validation errors."""
            context = ErrorContext(
                operation="request_validation",
                additional_data={"errors": exc.errors()}
            )
            
            error = error_factory.create_validation_error(
                "Request validation failed",
                validation_errors=[str(error) for error in exc.errors()],
                context=context
            )
            
            return ORJSONResponse(
                status_code=422,
                content=error.to_dict()
            )
        
        @app.exception_handler(ValidationError)
        async def onyx_validation_exception_handler(request: Request, exc: ValidationError):
            """Handle Onyx validation errors."""
            return ORJSONResponse(
                status_code=400,
                content=exc.to_dict()
            )
        
        @app.exception_handler(SystemError)
        async def system_exception_handler(request: Request, exc: SystemError):
            """Handle Onyx system errors."""
            return ORJSONResponse(
                status_code=500,
                content=exc.to_dict()
            )
        
        # Override default HTTP exception handler
        app.add_exception_handler(Exception, http_exception_handler)
    
    def setup_health_checks(self, app: FastAPI) -> None:
        """Setup health check endpoints."""
        if not self.config.health_check_enabled:
            return
        
        @app.get("/health", tags=["health"])
        async def health_check():
            """Basic health check."""
            return {
                "status": "healthy",
                "service": self.config.title,
                "version": self.config.version
            }
        
        @app.get("/health/ready", tags=["health"])
        async def readiness_check():
            """Readiness check for Kubernetes."""
            try:
                # Add your readiness checks here
                # e.g., database connectivity, external service health
                return {"status": "ready"}
            except Exception as e:
                logger.error("Readiness check failed", error=str(e))
                return {"status": "not_ready", "error": str(e)}
        
        @app.get("/health/live", tags=["health"])
        async def liveness_check():
            """Liveness check for Kubernetes."""
            return {"status": "alive"}
    
    def setup_routes(self, app: FastAPI) -> None:
        """Setup routes and routers."""
        
        # Root endpoint
        @app.get("/", tags=["root"])
        async def root():
            """Root endpoint with service information."""
            return {
                "service": self.config.title,
                "version": self.config.version,
                "description": self.config.description,
                "docs_url": self.config.docs_url,
                "health_url": "/health"
            }
        
        # Include routers
        for router_config in self.routers:
            app.include_router(
                router_config["router"],
                prefix=router_config["prefix"],
                tags=router_config["tags"]
            )
    
    def create_app(self) -> FastAPI:
        """Create and configure a FastAPI application."""
        
        # Create FastAPI app
        app = FastAPI(
            title=self.config.title,
            version=self.config.version,
            description=self.config.description,
            debug=self.config.debug,
            docs_url=self.config.docs_url,
            redoc_url=self.config.redoc_url,
            lifespan=self.create_lifespan(),
            default_response_class=ORJSONResponse
        )
        
        # Setup components
        self.setup_middleware(app)
        self.setup_rate_limiting(app)
        self.setup_monitoring(app)
        self.setup_error_handlers(app)
        self.setup_health_checks(app)
        self.setup_routes(app)
        
        logger.info("FastAPI application created successfully",
                   title=self.config.title,
                   version=self.config.version)
        
        return app

# Convenience functions for common configurations

def create_production_app(
    title: str = "Blatam Academy API",
    version: str = "1.0.0",
    routers: Optional[List[Dict[str, Any]]] = None
) -> FastAPI:
    """Create a production-ready FastAPI application."""
    
    config = FastAPIConfig(
        title=title,
        version=version,
        debug=False,
        cors_origins=["https://yourdomain.com"],  # Configure for your domain
        rate_limit_enabled=True,
        monitoring_enabled=True,
        health_check_enabled=True,
        security_headers_enabled=True
    )
    
    factory = FastAPIFactory(config)
    
    # Add routers if provided
    if routers:
        for router_config in routers:
            factory.add_router(**router_config)
    
    return factory.create_app()

def create_development_app(
    title: str = "Blatam Academy API (Dev)",
    version: str = "1.0.0-dev",
    routers: Optional[List[Dict[str, Any]]] = None
) -> FastAPI:
    """Create a development FastAPI application."""
    
    config = FastAPIConfig(
        title=title,
        version=version,
        debug=True,
        cors_origins=["*"],  # Allow all origins in development
        rate_limit_enabled=False,  # Disable rate limiting in development
        monitoring_enabled=True,
        health_check_enabled=True,
        security_headers_enabled=False  # Disable security headers in development
    )
    
    factory = FastAPIFactory(config)
    
    # Add routers if provided
    if routers:
        for router_config in routers:
            factory.add_router(**router_config)
    
    return factory.create_app()

def create_test_app(
    title: str = "Blatam Academy API (Test)",
    version: str = "1.0.0-test"
) -> FastAPI:
    """Create a test FastAPI application."""
    
    config = FastAPIConfig(
        title=title,
        version=version,
        debug=True,
        cors_origins=["*"],
        rate_limit_enabled=False,
        monitoring_enabled=False,
        health_check_enabled=False,
        security_headers_enabled=False
    )
    
    factory = FastAPIFactory(config)
    return factory.create_app()

# Example usage:
"""
# Create a production app with routers

app = create_production_app(
    title="Blatam Academy API",
    version="1.0.0",
    routers=[
        {"router": notifications_router, "prefix": "/api/v1/notifications", "tags": ["notifications"]},
        {"router": integrated_router, "prefix": "/api/v1/integrated", "tags": ["integrated"]},
        {"router": professional_documents_router, "prefix": "/api/v1", "tags": ["professional-documents"]}
    ]
)

# Or use the factory directly for more control
factory = FastAPIFactory(FastAPIConfig(title="Custom API"))
factory.add_startup_event(database_connect)
factory.add_shutdown_event(database_disconnect)
factory.add_router(notifications_router, prefix="/api/v1/notifications")
app = factory.create_app()
""" 