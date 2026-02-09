from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from .routes import router as seo_router
from .middleware import (
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API module for Ultra-Optimized SEO Service.
Contains routes, middleware, and API-related components.
"""


    RequestLoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware,
    CacheMiddleware,
    CircuitBreakerMiddleware,
    RetryMiddleware
)

__all__ = [
    'seo_router',
    'RequestLoggingMiddleware',
    'MetricsMiddleware',
    'RateLimitMiddleware',
    'SecurityMiddleware',
    'CacheMiddleware',
    'CircuitBreakerMiddleware',
    'RetryMiddleware'
]

# Crear aplicación FastAPI
app = FastAPI(
    title="SEO Analysis API v2 - Ultra Optimized",
    description="API ultra-optimizada para análisis SEO con arquitectura modular",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar rate limiter
limiter = create_rate_limiter()

# Configurar middlewares
setup_middleware(app)

# Incluir rutas
app.include_router(seo_router)

# Configurar rate limiter en la app
app.state.limiter = limiter

# Endpoint de métricas Prometheus
@app.get("/metrics")
async def metrics():
    """Endpoint para métricas Prometheus."""
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Endpoint de salud
@app.get("/health")
async def health():
    """Health check básico."""
    return {"status": "healthy", "version": "2.0.0"}

# Endpoint raíz
@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "SEO Analysis API v2 - Ultra Optimized",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc) -> Any:
    """Manejo global de excepciones."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )

# Función para ejecutar la aplicación
def run_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    loop: str = "uvloop",
    http: str = "httptools",
    access_log: bool = False,
    log_level: str = "info"
):
    """Ejecuta la aplicación con configuración optimizada."""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        loop=loop,
        http=http,
        access_log=access_log,
        log_level=log_level
    )

match __name__:
    case "__main__":
    run_app() 