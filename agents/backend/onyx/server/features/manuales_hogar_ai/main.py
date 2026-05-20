"""
Main entry point para Manuales Hogar AI
========================================

Servidor FastAPI independiente para el feature.
Refactorizado para microservicios, serverless y cloud-native.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api.routes import (
    manuales_router,
    history_router,
    ratings_router,
    export_router,
    search_router,
    share_router,
    notifications_router,
    templates_router,
    analytics_router,
    ml_router,
    streaming_router,
    health_router,
    metrics_router,
)
from .config.settings import get_settings
from .middleware import (
    LoggingMiddleware,
    SecurityMiddleware,
    RateLimitMiddleware,
    TracingMiddleware,
    MetricsMiddleware,
)
from .core.error_handlers import (
    global_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    custom_exception_handler,
    database_exception_handler,
    httpx_exception_handler,
)
from .core.exceptions import ManualesHogarAIException
from sqlalchemy.exc import SQLAlchemyError
import httpx

# Get settings first
settings = get_settings()

# Configure structured logging
log_level = logging.DEBUG if settings.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Validate critical settings on startup
if not settings.openrouter_api_key:
    logger.warning("OpenRouter API key not configured. API calls will fail.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown with graceful handling."""
    # Startup
    logger.info("Starting Manuales Hogar AI service...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize connection manager
    from .core.connection.connection_manager import get_connection_manager
    connection_manager = None
    try:
        connection_manager = await get_connection_manager()
        # Start health checks
        await connection_manager.start_health_checks(interval=60)
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}", exc_info=True)
        # Continue anyway - service can work with degraded functionality
    
    yield
    
    # Shutdown - graceful cleanup
    logger.info("Shutting down Manuales Hogar AI service...")
    
    try:
        if connection_manager:
            await connection_manager.cleanup()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    logger.info("Service shutdown complete")


app = FastAPI(
    title="Manuales Hogar AI",
    description="Sistema de IA para generar manuales paso a paso tipo LEGO para oficios populares",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Register exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(ManualesHogarAIException, custom_exception_handler)
app.add_exception_handler(SQLAlchemyError, database_exception_handler)
app.add_exception_handler(httpx.HTTPError, httpx_exception_handler)

# Add advanced middleware (order matters!)
# 0. Stability (first for request validation)
from .middleware.stability_middleware import StabilityMiddleware
app.add_middleware(StabilityMiddleware)

# 1. Logging (second to capture all requests)
app.add_middleware(LoggingMiddleware)

# 2. Security headers
app.add_middleware(
    SecurityMiddleware,
    allow_origins=settings.allowed_origins,
)

# 3. Rate limiting
redis_url = None if settings.redis_url == "redis://localhost:6379/0" else settings.redis_url
app.add_middleware(
    RateLimitMiddleware,
    redis_url=redis_url,
    requests_per_minute=settings.rate_limit_per_minute,
    requests_per_hour=settings.rate_limit_per_hour,
)

# 4. Distributed tracing
if settings.enable_tracing:
    app.add_middleware(
        TracingMiddleware,
        service_name="manuales-hogar-ai",
        otlp_endpoint=settings.otlp_endpoint,
    )

# 5. Prometheus metrics
if settings.enable_prometheus:
    app.add_middleware(MetricsMiddleware)

# CORS (fallback, SecurityMiddleware also handles CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(health_router)
app.include_router(metrics_router)  # Prometheus metrics
app.include_router(manuales_router)
app.include_router(history_router)
app.include_router(ratings_router)
app.include_router(export_router)
app.include_router(search_router)
app.include_router(share_router)
app.include_router(notifications_router)
app.include_router(templates_router)
app.include_router(analytics_router)
app.include_router(ml_router)
app.include_router(streaming_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Manuales Hogar AI",
        "version": "1.0.0",
        "description": "Sistema de IA para generar manuales paso a paso tipo LEGO",
        "endpoints": {
            "health": "/api/v1/health",
            "models": "/api/v1/models",
            "generate_from_text": "/api/v1/generate-from-text",
            "generate_from_image": "/api/v1/generate-from-image",
            "generate_combined": "/api/v1/generate-combined",
            "categories": "/api/v1/categories"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting Manuales Hogar AI server on port {port}")
    logger.info(f"OpenRouter API Key configured: {settings.openrouter_api_key is not None}")
    
    # Disable reload in production
    reload = os.getenv("ENVIRONMENT", "dev") != "prod"
    
    uvicorn.run(
        "manuales_hogar_ai.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        workers=1 if reload else int(os.getenv("WORKERS", "4"))
    )

