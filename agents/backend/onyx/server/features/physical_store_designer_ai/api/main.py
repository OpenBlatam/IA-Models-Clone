"""
FastAPI application for Physical Store Designer AI

This module sets up the main FastAPI application with all middleware,
routers, and configuration.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.logging_config import setup_logging, get_logger
from ..core.middleware import (
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    CompressionMiddleware,
    TimeoutMiddleware
)
from ..config.settings import settings
from ..services.health import get_health_checker

# Lazy import routers for better startup performance
# Routers are imported when app is created to reduce initial load time

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Physical Store Designer AI API",
    description="API para diseñar locales físicos completos con IA",
    version="1.46.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "store-designer", "description": "Operaciones principales de diseño de stores"},
        {"name": "analysis", "description": "Análisis de stores (competencia, financiero, KPIs)"},
        {"name": "advanced", "description": "Funcionalidades avanzadas (comparación, versionado, feedback)"},
        {"name": "premium", "description": "Funcionalidades premium (reportes, colaboración, dashboard)"},
        {"name": "enterprise", "description": "Funcionalidades enterprise (auth, optimización, predicción)"},
        {"name": "health", "description": "Health checks y monitoreo"},
        {"name": "monitoring", "description": "Métricas y observabilidad"},
    ],
)

# Add middleware in order (last added is first executed)
# Timeout (should be early to catch long-running requests)
app.add_middleware(TimeoutMiddleware, timeout_seconds=30.0)

# Compression (should be early to compress responses)
app.add_middleware(CompressionMiddleware, min_size=1024)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_per_minute
)

# Error handling (should be early to catch all errors)
app.add_middleware(ErrorHandlerMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (lazy loaded for better startup performance)
def _include_routers() -> None:
    """
    Include all routers - lazy loaded to improve startup time.
    
    This function imports and registers all API routers. Routers are imported
    lazily (inside this function) to reduce initial application startup time.
    Only the routers that are actually used will be loaded.
    
    Note:
        All routers are prefixed with `/api/v1` and tagged appropriately
        for better API documentation organization.
    """
    from .routes import router
    from .analysis_routes import router as analysis_router
    from .advanced_routes import router as advanced_router
    from .premium_routes import router as premium_router
    from .enterprise_routes import router as enterprise_router
    from .integration_routes import router as integration_router
    from .business_routes import router as business_router
    from .engagement_routes import router as engagement_router
    from .advanced_features_routes import router as advanced_features_router
    from .iot_erp_routes import router as iot_erp_router
    from .future_tech_routes import router as future_tech_router
    from .cutting_edge_routes import router as cutting_edge_router
    from .next_gen_routes import router as next_gen_router
    from .advanced_tech_routes import router as advanced_tech_router
    from .final_tech_routes import router as final_tech_router
    from .deep_learning_routes import router as deep_learning_router
    from .advanced_dl_routes import router as advanced_dl_router
    from .ml_ops_routes import router as ml_ops_router
    from .advanced_ml_routes import router as advanced_ml_router
    from .expert_ml_routes import router as expert_ml_router
    from .training_tools_routes import router as training_tools_router
    from .utility_routes import router as utility_router
    from .production_routes import router as production_router
    
    app.include_router(router, prefix="/api/v1", tags=["store-designer"])
    app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
    app.include_router(advanced_router, prefix="/api/v1", tags=["advanced"])
    app.include_router(premium_router, prefix="/api/v1", tags=["premium"])
    app.include_router(enterprise_router, prefix="/api/v1", tags=["enterprise"])
    app.include_router(integration_router, prefix="/api/v1", tags=["integration"])
    app.include_router(business_router, prefix="/api/v1", tags=["business"])
    app.include_router(engagement_router, prefix="/api/v1", tags=["engagement"])
    app.include_router(advanced_features_router, prefix="/api/v1", tags=["advanced-features"])
    app.include_router(iot_erp_router, prefix="/api/v1", tags=["iot-erp"])
    app.include_router(future_tech_router, prefix="/api/v1", tags=["future-tech"])
    app.include_router(cutting_edge_router, prefix="/api/v1", tags=["cutting-edge"])
    app.include_router(next_gen_router, prefix="/api/v1", tags=["next-gen"])
    app.include_router(advanced_tech_router, prefix="/api/v1", tags=["advanced-tech"])
    app.include_router(final_tech_router, prefix="/api/v1", tags=["final-tech"])
    app.include_router(deep_learning_router, prefix="/api/v1", tags=["deep-learning"])
    app.include_router(advanced_dl_router, prefix="/api/v1", tags=["advanced-dl"])
    app.include_router(ml_ops_router, prefix="/api/v1", tags=["ml-ops"])
    app.include_router(advanced_ml_router, prefix="/api/v1", tags=["advanced-ml"])
    app.include_router(expert_ml_router, prefix="/api/v1", tags=["expert-ml"])
    app.include_router(training_tools_router, prefix="/api/v1", tags=["training-tools"])
    app.include_router(utility_router, prefix="/api/v1", tags=["utilities"])
    app.include_router(production_router, prefix="/api/v1", tags=["production"])

# Include all routers
_include_routers()


@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns detailed health status including:
    - Overall system status
    - Component health (storage, services, etc.)
    - Performance metrics
    - Resource usage
    
    Returns:
        Dict containing comprehensive health information
        
    Example:
        ```json
        {
            "status": "healthy",
            "timestamp": "2024-01-01T12:00:00",
            "components": {...},
            "metrics": {...}
        }
        ```
    """
    health_checker = get_health_checker()
    return health_checker.run_all_checks()


@app.get("/health/live", tags=["health"])
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check - simple endpoint to verify service is running.
    
    This endpoint is used by orchestration systems (Kubernetes, Docker, etc.)
    to determine if the service should be restarted. Returns a simple
    status indicating the service is alive.
    
    Returns:
        Dict with status "alive"
        
    Example:
        ```json
        {"status": "alive"}
        ```
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - detailed readiness status.
    
    This endpoint checks if the service is ready to accept traffic.
    Returns detailed status including whether the service is ready
    to handle requests.
    
    Returns:
        Dict with status and ready flag
        
    Example:
        ```json
        {
            "status": "healthy",
            "ready": true
        }
        ```
    """
    health_checker = get_health_checker()
    status = health_checker.get_status()
    return {"status": status, "ready": status in ["healthy", "degraded"]}


@app.get("/", tags=["info"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint - service information.
    
    Returns basic information about the API service including:
    - Service name and version
    - Current status
    - Links to documentation and health endpoints
    
    Returns:
        Dict with service information
        
    Example:
        ```json
        {
            "service": "Physical Store Designer AI",
            "version": "1.45.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }
        ```
    """
    return {
        "service": "Physical Store Designer AI",
        "version": "1.46.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "environment": settings.environment,
        "features": {
            "ml": settings.enable_ml_features,
            "deep_learning": settings.enable_deep_learning,
            "health_checks": settings.enable_health_checks
        }
    }


@app.get("/metrics", tags=["monitoring"])
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics - performance and operational metrics.
    
    Returns comprehensive metrics including:
    - HTTP request metrics (counters, timers, gauges)
    - Service factory statistics
    - Performance indicators
    
    Returns:
        Dict with all collected metrics
        
    Example:
        ```json
        {
            "metrics": {
                "counters": {...},
                "gauges": {...},
                "timers": {...}
            },
            "factory": {
                "instances": {...},
                "cache_hits": 100
            },
            "timestamp": "2024-01-01T12:00:00"
        }
        ```
    """
    from ..core.metrics import get_metrics_collector
    from ..core.factories import ServiceFactory
    metrics = get_metrics_collector()
    factory_stats = ServiceFactory.get_stats()
    return {
        "metrics": metrics.get_all_metrics(),
        "factory": factory_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics - hit rates, sizes, etc."""
    from ..core.cache import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.get_all_stats()


@app.post("/cache/cleanup")
async def cleanup_cache() -> Dict[str, Any]:
    """Cleanup expired cache entries"""
    from ..core.cache import get_cache_manager
    cache_manager = get_cache_manager()
    results = cache_manager.cleanup_all()
    return {"cleaned": results}


@app.post("/cache/clear")
async def clear_cache() -> Dict[str, Any]:
    """Clear all caches"""
    from ..core.cache import get_cache_manager
    cache_manager = get_cache_manager()
    results = cache_manager.clear_all()
    return {"cleared": results}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get background task status"""
    from ..core.background_tasks import get_task_status
    from ..core.exceptions import NotFoundError
    status = get_task_status(task_id)
    if status is None:
        raise NotFoundError("Task", task_id)
    return status


@app.post("/tasks")
async def create_background_task(
    task_id: str,
    func_name: str,
    *args: Any,
    max_retries: int = 0,
    **kwargs: Any
) -> Dict[str, str]:
    """Create a background task (example - in production, use proper task definitions)"""
    # Note: In production, you'd have a registry of allowed functions
    # This is just an example
    return {"message": "Task creation endpoint - implement with proper function registry"}


@app.get("/circuit-breakers")
async def get_circuit_breakers() -> Dict[str, Any]:
    """Get all circuit breaker states"""
    from ..core.circuit_breaker import get_all_circuit_breakers
    return get_all_circuit_breakers()

