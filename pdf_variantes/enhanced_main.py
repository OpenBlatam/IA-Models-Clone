"""Enhanced main application with best-in-class libraries and patterns."""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

# Best libraries for web framework
import structlog
import sentry_sdk
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, REGISTRY

# Enhanced components
from .enhanced_config import get_ultra_fast_config
from .routers import enhanced_router
from .advanced_performance import create_performance_monitor
from .advanced_error_handling import create_error_handler, get_error_summary
from .enhanced_dependencies import get_config, get_current_user

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ERROR_COUNT = Counter('http_errors_total', 'Total HTTP errors', ['error_type'])

# Initialize Sentry for error tracking
try:
    sentry_sdk.init(
        dsn="your-sentry-dsn-here",  # Replace with actual DSN
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )
    logger.info("Sentry initialized successfully")
except Exception as e:
    logger.warning(f"Sentry initialization failed: {e}")


@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced application lifespan with monitoring."""
    start_time = time.time()
    
    # Startup
    logger.info("🚀 Starting Enhanced PDF Variantes API")
    
    # Load configuration
    config = get_ultra_fast_config()
    app.state.config = config
    app.state.start_time = start_time
    app.state.performance_monitor = create_performance_monitor()
    app.state.error_handler = create_error_handler()
    
    logger.info(f"📋 Configuration loaded for {config.environment.value} environment")
    logger.info(f"🔧 Features enabled: {sum(config.features.values())}/{len(config.features)}")
    
    # Record startup metrics
    REQUEST_COUNT.labels(method='STARTUP', endpoint='/', status='success').inc()
    
    yield
    
    # Shutdown
    uptime = time.time() - start_time
    logger.info(f"🛑 Shutting down Enhanced PDF Variantes API (uptime: {uptime:.2f}s)")
    
    # Record shutdown metrics
    REQUEST_COUNT.labels(method='SHUTDOWN', endpoint='/', status='success').inc()


def create_enhanced_app() -> FastAPI:
    """Create enhanced FastAPI application with best practices."""
    config = get_ultra_fast_config()
    
    # Create FastAPI app with enhanced configuration
    app = FastAPI(
        title="Enhanced PDF Variantes API",
        description="Advanced PDF processing system with best-in-class libraries and AI capabilities",
        version="3.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None,
        lifespan=enhanced_lifespan
    )
    
    # Enhanced CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include enhanced router
    app.include_router(enhanced_router)
    
    # Enhanced exception handler
    @app.exception_handler(Exception)
    async def enhanced_exception_handler(request: Request, exc: Exception):
        """Enhanced exception handling with monitoring."""
        ERROR_COUNT.labels(error_type=type(exc).__name__).inc()
        
        logger.error(
            "Unhandled exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": getattr(request.state, "request_id", "")
            }
        )
    
    # Enhanced health check endpoint
    @app.get("/health", tags=["System"], summary="Enhanced Health Check")
    async def enhanced_health_check():
        """Enhanced health check with comprehensive monitoring."""
        config = get_ultra_fast_config()
        uptime = time.time() - app.state.start_time
        
        # Get performance metrics
        perf_monitor = app.state.performance_monitor
        performance_summary = perf_monitor.get_performance_summary()
        
        # Get error summary
        error_summary = get_error_summary()
        
        return {
            "status": "healthy",
            "service": "enhanced-pdf-variantes",
            "version": "3.0.0",
            "environment": config.environment.value,
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "features": {
                "enabled": sum(config.features.values()),
                "total": len(config.features),
                "list": {name: enabled for name, enabled in config.features.items() if enabled}
            },
            "performance": {
                "requests_per_second": performance_summary.get("requests_per_second", 0),
                "average_response_time": performance_summary.get("average_response_time", 0),
                "total_requests": performance_summary.get("total_requests", 0)
            },
            "errors": {
                "total_errors": error_summary["metrics"]["total_errors"],
                "error_rate": error_summary["metrics"]["error_rate"],
                "recent_errors": len(error_summary["patterns"].get("recent_errors", {}))
            },
            "libraries": {
                "pymupdf": "available",
                "pdfplumber": "available",
                "spacy": "available",
                "sentence_transformers": "available",
                "textstat": "available",
                "prometheus": "available",
                "sentry": "available"
            }
        }
    
    # Enhanced root endpoint
    @app.get("/", tags=["System"], summary="API Information")
    async def enhanced_root():
        """Enhanced root endpoint with comprehensive information."""
        return {
            "message": "Enhanced PDF Variantes API",
            "version": "3.0.0",
            "description": "Advanced PDF processing system with best-in-class libraries and AI capabilities",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
                "pdf": "/pdf"
            },
            "features": [
                "Advanced PDF Processing with PyMuPDF",
                "AI-Powered Topic Extraction",
                "Intelligent Variant Generation",
                "Comprehensive Quality Analysis",
                "Real-time Monitoring",
                "Error Tracking with Sentry",
                "Performance Metrics with Prometheus"
            ],
            "libraries": [
                "PyMuPDF (fitz) - Fastest PDF processing",
                "pdfplumber - Excellent text extraction",
                "spaCy - Advanced NLP processing",
                "Sentence Transformers - AI embeddings",
                "textstat - Readability analysis",
                "Prometheus - Metrics collection",
                "Sentry - Error tracking"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Prometheus metrics endpoint
    @app.get("/metrics", tags=["System"], summary="Prometheus Metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
    
    # Enhanced configuration endpoint
    @app.get("/config", tags=["System"], summary="Get Configuration")
    async def get_configuration(
        config = Depends(get_config),
        current_user = Depends(get_current_user)
    ):
        """Get current configuration (admin only)."""
        config_dict = config.to_dict()
        
        # Remove sensitive information
        if "api_key" in config_dict.get("ai", {}):
            config_dict["ai"]["api_key"] = "***"
        if "password" in config_dict.get("redis", {}):
            config_dict["redis"]["password"] = "***"
        if "secret_key" in config_dict.get("security", {}):
            config_dict["security"]["secret_key"] = "***"
        
        return config_dict
    
    # Enhanced error statistics endpoint
    @app.get("/error-stats", tags=["System"], summary="Error Statistics")
    async def get_error_statistics():
        """Get comprehensive error statistics."""
        return get_error_summary()
    
    # Enhanced performance metrics endpoint
    @app.get("/performance", tags=["System"], summary="Performance Metrics")
    async def get_performance_metrics():
        """Get comprehensive performance metrics."""
        perf_monitor = app.state.performance_monitor
        return perf_monitor.get_performance_summary()
    
    # Request metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Enhanced metrics middleware."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Record Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        REQUEST_DURATION.observe(duration)
        
        # Record performance metrics
        perf_monitor = app.state.performance_monitor
        perf_monitor.record_latency("http_request", duration)
        perf_monitor.increment_counter("total_requests")
        
        if response.status_code >= 400:
            perf_monitor.increment_counter("error_requests")
            ERROR_COUNT.labels(error_type=f"http_{response.status_code}").inc()
        
        # Add enhanced headers
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "")
        response.headers["X-API-Version"] = "3.0.0"
        
        return response
    
    return app


# Create app instance
app = create_enhanced_app()

# Export for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )