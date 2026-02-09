"""
API principal para Social Media Identity Clone AI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from .routes import router
from ..config import get_settings
from ..middleware.rate_limiter import RateLimitMiddleware, get_rate_limiter
from ..middleware.security import SecurityMiddleware
from ..middleware.logging_middleware import LoggingMiddleware
from ..middleware.performance_middleware import PerformanceMiddleware
from ..middleware.compression_middleware import CompressionMiddleware
from ..analytics.metrics import get_metrics_collector
from ..queue.worker import start_workers

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Identity Clone AI",
    description="API para clonar identidades de redes sociales y generar contenido",
    version="1.0.0"
)

# Middleware de logging (primero para capturar todo)
app.add_middleware(LoggingMiddleware)

# Compression (antes de otros middlewares para comprimir respuestas)
app.add_middleware(CompressionMiddleware)

# Performance monitoring
app.add_middleware(PerformanceMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
rate_limiter = get_rate_limiter()
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Security middleware
app.add_middleware(SecurityMiddleware, require_api_key=False)  # Cambiar a True en producción

# Incluir rutas
app.include_router(router, prefix="/api/v1", tags=["identity-clone"])


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Social Media Identity Clone AI",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - usa el endpoint en routes para health completo"""
    from ..utils.health_check import get_health_check
    health = get_health_check()
    return health.get_health_status()


@app.on_event("startup")
async def startup_event():
    """Inicia workers y scheduler al arrancar la aplicación"""
    try:
        # Iniciar workers
        await start_workers(num_workers=2)
        logger.info("Workers iniciados")
        
        # Iniciar scheduler
        from ..scheduler.scheduler_service import get_scheduler_service
        scheduler = get_scheduler_service()
        asyncio.create_task(scheduler.start_scheduler())
        logger.info("Scheduler iniciado")
    except Exception as e:
        logger.error(f"Error en startup: {e}", exc_info=True)

