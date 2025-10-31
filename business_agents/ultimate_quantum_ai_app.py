"""
Ultimate Quantum AI Application
Complete FastAPI application with all quantum AI systems
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, ORJSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uvicorn
import os
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
import asyncio

# Simple API key dependency (optional)
from fastapi import Depends

def get_api_key(x_api_key: Optional[str] = None):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Import all systems
from ml_nlp_benchmark import get_ml_nlp_benchmark
from advanced_ml_nlp_benchmark import get_advanced_ml_nlp_benchmark
from ml_nlp_benchmark_quantum_computing import get_quantum_computing
from ml_nlp_benchmark_neuromorphic_computing import get_neuromorphic_computing
from ml_nlp_benchmark_biological_computing import get_biological_computing
from ml_nlp_benchmark_cognitive_computing import get_cognitive_computing
from ml_nlp_benchmark_quantum_ai import get_quantum_ai
from ml_nlp_benchmark_advanced_quantum_computing import get_advanced_quantum_computing

# Import all routes
from .domains.registry import load_routers
from .plugins import discover_plugin_routers
from .tasks.routes import router as tasks_router
from .batch_routes import router as batch_router
from .websockets.routes import router as websocket_router
from .webhooks.routes import router as webhook_router
from .settings import settings
from .cache import AsyncCache, cached_json
from .http_client import ResilientHTTPClient
from .security import auth_dependency
from .providers import provide_cache, provide_http_client
from .config_validator import validate_config
from .logging_config import setup_logging
from .exceptions import APIError
from .middleware.request_logging import AdvancedRequestLoggingMiddleware
from .rate_limiter_redis import DistributedRateLimitMiddleware
from .openapi_custom import custom_openapi
from .event_system import event_bus, EventType
from .db_helpers import init_db_manager, get_db_manager
from .middleware.profiling import ProfilingMiddleware
from .middleware.cache_headers import CacheHeadersMiddleware

# Configure logging
import os
use_json_logging = os.getenv("JSON_LOGGING", "false").lower() == "true"
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), use_json=use_json_logging)
logger = logging.getLogger(__name__)


# Optional OpenTelemetry instrumentation
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware  # type: ignore
    FastAPIInstrumentor.instrument_app  # type: ignore[attr-defined]
    OTEL_AVAILABLE = True
except Exception:
    OTEL_AVAILABLE = False

# Create FastAPI app
global_deps = []
if settings.enforce_auth:
    from fastapi import Depends
    global_deps = [Depends(auth_dependency)]

app = FastAPI(
    title="Ultimate Quantum AI ML NLP Benchmark",
    description="Complete ML NLP Benchmark system with quantum AI capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=ORJSONResponse,
    dependencies=global_deps
)

# Custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

# Expose event bus in app state
app.state.event_bus = event_bus

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID + Security Headers + Rate Limiting + Metrics + Limits/Timeout
# ------------------------------------------------------
REQUESTS_TOTAL = None
REQUEST_LATENCY = None

try:
    # Optional: only if prometheus_client is available
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    REQUESTS_TOTAL = Counter(
        "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
    )
except Exception:
    pass


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            process_time = time.time() - start_time
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(process_time)
                # Basic security headers
                response.headers.setdefault("X-Content-Type-Options", "nosniff")
                response.headers.setdefault("X-Frame-Options", "DENY")
                response.headers.setdefault("Referrer-Policy", "no-referrer")
                response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
                # Remove Server header if present (minor security/perf)
                if "server" in response.headers:
                    del response.headers["server"]
            # Prometheus metrics (best-effort)
            if REQUESTS_TOTAL and REQUEST_LATENCY:
                path_label = request.url.path
                REQUEST_LATENCY.labels(request.method, path_label).observe(process_time)
                status_code = str(response.status_code if response is not None else 500)
                REQUESTS_TOTAL.labels(request.method, path_label, status_code).inc()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.capacity = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0
        self.state: Dict[str, Dict[str, float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self.state.get(client_ip)
        if not bucket:
            bucket = {"tokens": float(self.capacity), "updated": now}
        # Refill tokens
        elapsed = now - bucket["updated"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["updated"] = now
        self.state[client_ip] = bucket

        # Cost of a request
        if bucket["tokens"] < 1.0:
            return JSONResponse(status_code=429, content={"error": "rate_limited", "detail": "Too Many Requests"})
        bucket["tokens"] -= 1.0
        return await call_next(request)


app.add_middleware(RequestContextMiddleware)

# Rate limiting: distributed (Redis) or in-memory
if settings.use_distributed_rate_limit and settings.redis_url:
    app.add_middleware(
        DistributedRateLimitMiddleware,
        redis_url=settings.redis_url,
        requests_per_minute=settings.rps_limit
    )
    # Store reference for initialization in startup
    app.state._use_distributed_rate_limit = True
else:
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rps_limit)
    app.state._use_distributed_rate_limit = False

# Advanced request logging (optional)
if settings.enable_request_logging:
    app.add_middleware(
        AdvancedRequestLoggingMiddleware,
        log_body=settings.log_request_body,
        log_headers=settings.log_request_headers
    )


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int = 5 * 1024 * 1024):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        # Short-circuit by Content-Length if present
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > self.max_body_bytes:
                return JSONResponse(status_code=413, content={"error": "payload_too_large"})

        async def receive_with_limit() -> Message:
            message = await request._receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body and len(body) > self.max_body_bytes:
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        request._receive = receive_with_limit  # type: ignore[attr-defined]
        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: float = 30.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504, content={"error": "timeout", "detail": "Request timed out"})


app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=settings.request_timeout_seconds)

# Enable OTEL if installed and enabled
if OTEL_AVAILABLE and settings.enable_otel:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
        FastAPIInstrumentor().instrument_app(app)  # type: ignore
        logger.info("OpenTelemetry instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to enable OpenTelemetry: {e}")


def create_app(custom_settings=None):
    # Factory returns the configured global app for now.
    # In future, build a new app instance using provided settings.
    return app

# Include all routers via registry
for router, prefix in load_routers():
    app.include_router(router, prefix=prefix)

# Include discovered plugin routers
for router, prefix in discover_plugin_routers():
    app.include_router(router, prefix=prefix)

# Include task management router
app.include_router(tasks_router, prefix="/api/v1")

# Include batch processing router
app.include_router(batch_router, prefix="/api/v1")

# Include WebSocket router
app.include_router(websocket_router)

# Include webhooks router
app.include_router(webhook_router, prefix="/api/v1")

# Global exception handler
@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle custom API errors."""
    content = {
        "error": exc.detail,
        "error_code": exc.error_code,
        "metadata": exc.metadata
    }
    logger.warning(f"API error: {exc.error_code} - {exc.detail}", extra={"extra_fields": exc.metadata})
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all other exceptions."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An internal error occurred"
        }
    )


# Startup/Shutdown hooks
@app.on_event("startup")
async def on_startup():
    logger.info("API starting up")
    
    # Validate configuration
    is_valid, warnings = validate_config()
    if warnings:
        logger.warning(f"Configuration validation found {len(warnings)} warning(s)")
    
    # Initialize services
    app.state.cache = provide_cache()
    await app.state.cache.init()
    app.state.http = provide_http_client()
    
    # Note: Distributed rate limiter initializes Redis on first use
    if getattr(app.state, "_use_distributed_rate_limit", False):
        logger.info("Distributed rate limiting enabled (will initialize Redis on first request)")
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("API shutting down")
    try:
        await app.state.http.close()
    except Exception:
        pass

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system overview"""
    return {
        "message": "Ultimate Quantum AI ML NLP Benchmark System",
        "version": "1.0.0",
        "description": "Complete ML NLP Benchmark system with quantum AI capabilities",
        "systems": [
            "Basic ML NLP Benchmark",
            "Advanced ML NLP Benchmark", 
            "Quantum Computing",
            "Neuromorphic Computing",
            "Biological Computing",
            "Cognitive Computing",
            "Quantum AI",
            "Advanced Quantum Computing"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "status": "/status",
            "capabilities": "/capabilities",
            "metrics": "/metrics",
            "prometheus": "/metrics/prometheus"
        },
        "timestamp": datetime.now().isoformat()
    }


# Liveness and Readiness
@app.get("/live")
async def live():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def ready():
    """Detailed readiness check with component validation."""
    try:
        checks = {
            "cache": False,
            "http_client": False,
        }
        
        # Check cache
        if hasattr(app.state, "cache"):
            try:
                # Try a simple cache operation
                await app.state.cache.set("_health_check", "ok", ttl=1)
                await app.state.cache.get("_health_check")
                checks["cache"] = True
            except Exception as e:
                logger.warning(f"Cache health check failed: {e}")
        
        # Check HTTP client
        if hasattr(app.state, "http"):
            checks["http_client"] = True
        
        all_ready = all(checks.values())
        
        if all_ready:
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "checks": checks
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "checks": checks,
                    "message": "Some components are not ready"
                }
            )
    except Exception as e:
        logger.error(f"Readiness failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all systems"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Check basic ML NLP Benchmark
        try:
            basic_system = get_ml_nlp_benchmark()
            basic_summary = basic_system.get_ml_nlp_benchmark_summary()
            health_status["systems"]["basic_ml_nlp_benchmark"] = {
                "status": "healthy",
                "total_analyses": basic_summary["total_analyses"],
                "active_analyses": basic_summary["active_analyses"]
            }
        except Exception as e:
            health_status["systems"]["basic_ml_nlp_benchmark"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check advanced ML NLP Benchmark
        try:
            advanced_system = get_advanced_ml_nlp_benchmark()
            advanced_summary = advanced_system.get_advanced_ml_nlp_benchmark_summary()
            health_status["systems"]["advanced_ml_nlp_benchmark"] = {
                "status": "healthy",
                "total_analyses": advanced_summary["total_analyses"],
                "active_analyses": advanced_summary["active_analyses"]
            }
        except Exception as e:
            health_status["systems"]["advanced_ml_nlp_benchmark"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Quantum Computing
        try:
            quantum_system = get_quantum_computing()
            quantum_summary = quantum_system.get_quantum_computing_summary()
            health_status["systems"]["quantum_computing"] = {
                "status": "healthy",
                "total_circuits": quantum_summary["total_circuits"],
                "active_circuits": quantum_summary["active_circuits"]
            }
        except Exception as e:
            health_status["systems"]["quantum_computing"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Neuromorphic Computing
        try:
            neuromorphic_system = get_neuromorphic_computing()
            neuromorphic_summary = neuromorphic_system.get_neuromorphic_computing_summary()
            health_status["systems"]["neuromorphic_computing"] = {
                "status": "healthy",
                "total_networks": neuromorphic_summary["total_networks"],
                "active_networks": neuromorphic_summary["active_networks"]
            }
        except Exception as e:
            health_status["systems"]["neuromorphic_computing"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Biological Computing
        try:
            biological_system = get_biological_computing()
            biological_summary = biological_system.get_biological_computing_summary()
            health_status["systems"]["biological_computing"] = {
                "status": "healthy",
                "total_systems": biological_summary["total_systems"],
                "active_systems": biological_summary["active_systems"]
            }
        except Exception as e:
            health_status["systems"]["biological_computing"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Cognitive Computing
        try:
            cognitive_system = get_cognitive_computing()
            cognitive_summary = cognitive_system.get_cognitive_computing_summary()
            health_status["systems"]["cognitive_computing"] = {
                "status": "healthy",
                "total_models": cognitive_summary["total_models"],
                "active_models": cognitive_summary["active_models"]
            }
        except Exception as e:
            health_status["systems"]["cognitive_computing"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Quantum AI
        try:
            quantum_ai_system = get_quantum_ai()
            quantum_ai_summary = quantum_ai_system.get_quantum_ai_summary()
            health_status["systems"]["quantum_ai"] = {
                "status": "healthy",
                "total_ais": quantum_ai_summary["total_ais"],
                "active_ais": quantum_ai_summary["active_ais"]
            }
        except Exception as e:
            health_status["systems"]["quantum_ai"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Advanced Quantum Computing
        try:
            advanced_quantum_system = get_advanced_quantum_computing()
            advanced_quantum_summary = advanced_quantum_system.get_advanced_quantum_summary()
            health_status["systems"]["advanced_quantum_computing"] = {
                "status": "healthy",
                "total_systems": advanced_quantum_summary["total_systems"],
                "active_systems": advanced_quantum_summary["active_systems"]
            }
        except Exception as e:
            health_status["systems"]["advanced_quantum_computing"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Status endpoint
@app.get("/status")
async def system_status():
    """Detailed system status"""
    try:
        async def _produce():
            status = {
                "timestamp": datetime.now().isoformat(),
                "systems": {}
            }
        
        # Get status from all systems
        systems = [
            ("basic_ml_nlp_benchmark", get_ml_nlp_benchmark),
            ("advanced_ml_nlp_benchmark", get_advanced_ml_nlp_benchmark),
            ("quantum_computing", get_quantum_computing),
            ("neuromorphic_computing", get_neuromorphic_computing),
            ("biological_computing", get_biological_computing),
            ("cognitive_computing", get_cognitive_computing),
            ("quantum_ai", get_quantum_ai),
            ("advanced_quantum_computing", get_advanced_quantum_computing)
        ]
        
            for system_name, system_getter in systems:
                try:
                    system = system_getter()
                    if hasattr(system, 'get_ml_nlp_benchmark_summary'):
                        summary = system.get_ml_nlp_benchmark_summary()
                    elif hasattr(system, 'get_advanced_ml_nlp_benchmark_summary'):
                        summary = system.get_advanced_ml_nlp_benchmark_summary()
                    elif hasattr(system, 'get_quantum_computing_summary'):
                        summary = system.get_quantum_computing_summary()
                    elif hasattr(system, 'get_neuromorphic_computing_summary'):
                        summary = system.get_neuromorphic_computing_summary()
                    elif hasattr(system, 'get_biological_computing_summary'):
                        summary = system.get_biological_computing_summary()
                    elif hasattr(system, 'get_cognitive_computing_summary'):
                        summary = system.get_cognitive_computing_summary()
                    elif hasattr(system, 'get_quantum_ai_summary'):
                        summary = system.get_quantum_ai_summary()
                    elif hasattr(system, 'get_advanced_quantum_summary'):
                        summary = system.get_advanced_quantum_summary()
                    else:
                        summary = {"status": "unknown"}
                    status["systems"][system_name] = summary
                except Exception as e:
                    status["systems"][system_name] = {"status": "error", "error": str(e)}
            return status

        cache: AsyncCache = app.state.cache
        return await cached_json(cache, "status", settings.cache_ttl_seconds, _produce)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Capabilities endpoint
@app.get("/capabilities")
async def system_capabilities():
    """Get all system capabilities"""
    try:
        async def _produce():
            capabilities = {
                "timestamp": datetime.now().isoformat(),
                "systems": {}
            }
        
        # Get capabilities from all systems
        systems = [
            ("basic_ml_nlp_benchmark", get_ml_nlp_benchmark),
            ("advanced_ml_nlp_benchmark", get_advanced_ml_nlp_benchmark),
            ("quantum_computing", get_quantum_computing),
            ("neuromorphic_computing", get_neuromorphic_computing),
            ("biological_computing", get_biological_computing),
            ("cognitive_computing", get_cognitive_computing),
            ("quantum_ai", get_quantum_ai),
            ("advanced_quantum_computing", get_advanced_quantum_computing)
        ]
        
            for system_name, system_getter in systems:
                try:
                    system = system_getter()
                    if hasattr(system, 'ml_nlp_benchmark_capabilities'):
                        capabilities["systems"][system_name] = system.ml_nlp_benchmark_capabilities
                    elif hasattr(system, 'advanced_ml_nlp_benchmark_capabilities'):
                        capabilities["systems"][system_name] = system.advanced_ml_nlp_benchmark_capabilities
                    elif hasattr(system, 'quantum_computing_capabilities'):
                        capabilities["systems"][system_name] = system.quantum_computing_capabilities
                    elif hasattr(system, 'neuromorphic_capabilities'):
                        capabilities["systems"][system_name] = system.neuromorphic_capabilities
                    elif hasattr(system, 'biological_capabilities'):
                        capabilities["systems"][system_name] = system.biological_capabilities
                    elif hasattr(system, 'cognitive_capabilities'):
                        capabilities["systems"][system_name] = system.cognitive_capabilities
                    elif hasattr(system, 'quantum_ai_capabilities'):
                        capabilities["systems"][system_name] = system.quantum_ai_capabilities
                    elif hasattr(system, 'advanced_quantum_capabilities'):
                        capabilities["systems"][system_name] = system.advanced_quantum_capabilities
                    else:
                        capabilities["systems"][system_name] = {"status": "unknown"}
                except Exception as e:
                    capabilities["systems"][system_name] = {"status": "error", "error": str(e)}
            return capabilities

        cache: AsyncCache = app.state.cache
        return await cached_json(cache, "capabilities", settings.cache_ttl_seconds, _produce)
        
    except Exception as e:
        logger.error(f"Capabilities check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics")
async def system_metrics():
    """Get comprehensive system metrics"""
    try:
        async def _produce():
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "overall_metrics": {
                    "total_systems": 8,
                    "active_systems": 0,
                    "total_analyses": 0,
                    "total_results": 0
                },
                "system_metrics": {}
            }
        
        # Get metrics from all systems
        systems = [
            ("basic_ml_nlp_benchmark", get_ml_nlp_benchmark),
            ("advanced_ml_nlp_benchmark", get_advanced_ml_nlp_benchmark),
            ("quantum_computing", get_quantum_computing),
            ("neuromorphic_computing", get_neuromorphic_computing),
            ("biological_computing", get_biological_computing),
            ("cognitive_computing", get_cognitive_computing),
            ("quantum_ai", get_quantum_ai),
            ("advanced_quantum_computing", get_advanced_quantum_computing)
        ]
        
            for system_name, system_getter in systems:
                try:
                    system = system_getter()
                    if hasattr(system, 'get_ml_nlp_benchmark_summary'):
                        summary = system.get_ml_nlp_benchmark_summary()
                    elif hasattr(system, 'get_advanced_ml_nlp_benchmark_summary'):
                        summary = system.get_advanced_ml_nlp_benchmark_summary()
                    elif hasattr(system, 'get_quantum_computing_summary'):
                        summary = system.get_quantum_computing_summary()
                    elif hasattr(system, 'get_neuromorphic_computing_summary'):
                        summary = system.get_neuromorphic_computing_summary()
                    elif hasattr(system, 'get_biological_computing_summary'):
                        summary = system.get_biological_computing_summary()
                    elif hasattr(system, 'get_cognitive_computing_summary'):
                        summary = system.get_cognitive_computing_summary()
                    elif hasattr(system, 'get_quantum_ai_summary'):
                        summary = system.get_quantum_ai_summary()
                    elif hasattr(system, 'get_advanced_quantum_summary'):
                        summary = system.get_advanced_quantum_summary()
                    else:
                        summary = {}
                    metrics["system_metrics"][system_name] = summary
                    if "total_analyses" in summary:
                        metrics["overall_metrics"]["total_analyses"] += summary.get("total_analyses", 0)
                    if "total_results" in summary:
                        metrics["overall_metrics"]["total_results"] += summary.get("total_results", 0)
                    if "active_analyses" in summary:
                        metrics["overall_metrics"]["active_systems"] += 1
                except Exception as e:
                    metrics["system_metrics"][system_name] = {"status": "error", "error": str(e)}
            return metrics

        cache: AsyncCache = app.state.cache
        return await cached_json(cache, "metrics", settings.cache_ttl_seconds, _produce)
        
    except Exception as e:
        logger.error(f"Metrics check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prometheus scrape endpoint (text format)
@app.get("/metrics/prometheus")
async def metrics_prometheus():
    try:
        if REQUESTS_TOTAL is None:
            return PlainTextResponse("prometheus_client not installed", status_code=503)
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Prometheus metrics failed: {e}")
        return PlainTextResponse("metrics error", status_code=500)

if __name__ == "__main__":
    uvicorn.run(
        "ultimate_quantum_ai_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )






