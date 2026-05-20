"""
FastAPI application for Faceless Video AI with AWS and observability support
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .auth_routes import router as auth_router
from .websocket_routes import router as ws_router
from ..services.middleware import RequestLoggingMiddleware, ErrorHandlerMiddleware, MetricsMiddleware
from ..services.health import get_health_checker
from ..config.settings import get_settings
import logging
import os

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Faceless Video AI API",
    description="API para generar videos sin rostro completamente con IA a partir de scripts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Setup OpenTelemetry for distributed tracing
try:
    from ..services.observability import setup_opentelemetry
    setup_opentelemetry(app)
    logger.info("OpenTelemetry instrumentation enabled")
except Exception as e:
    logger.warning(f"Failed to setup OpenTelemetry: {e}")

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(ws_router, prefix="/api/v1", tags=["websocket"])
app.include_router(router, prefix="/api/v1", tags=["faceless-video"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Faceless Video AI",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_checker = get_health_checker()
    return health_checker.run_all_checks()


@app.get("/health/live")
async def liveness_check():
    """Liveness check (simple) - used by Kubernetes/ECS"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check (detailed) - used by load balancers"""
    health_checker = get_health_checker()
    status = health_checker.get_status()
    return {"status": status, "ready": status in ["healthy", "degraded"]}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

