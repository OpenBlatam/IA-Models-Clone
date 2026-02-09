"""
App Factory - Factory para crear aplicación FastAPI modular
============================================================

Factory que crea la aplicación FastAPI con estructura modular.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from ..core.microservices_integration import setup_microservices_app
from ..optimizations.performance import (
    optimize_app,
    enable_fast_json_serialization
)
from ..optimizations.async_optimizations import optimize_async_operations
from ..optimizations.memory_optimizations import optimize_memory_usage
from ..debug.debug_config import is_debug_enabled
from ..debug.debug_middleware import DebugMiddleware, PerformanceDebugMiddleware
from ..debug.debug_endpoints import setup_debug_endpoints
from ..core.error_handler import setup_error_handlers
from .routes import (
    projects_router,
    generation_router,
    validation_router,
    export_router,
    deployment_router,
    analytics_router,
    health_router
)

logger = logging.getLogger(__name__)


def create_app(
    base_output_dir: str = "generated_projects",
    enable_continuous: bool = True,
    title: str = "AI Project Generator API",
    version: str = "1.0.0"
) -> FastAPI:
    """
    Crea aplicación FastAPI modular.
    
    Args:
        base_output_dir: Directorio base para proyectos
        enable_continuous: Habilitar generación continua
        title: Título de la API
        version: Versión de la API
    
    Returns:
        Aplicación FastAPI configurada
    """
    # Crear aplicación
    app = FastAPI(
        title=title,
        description="API modular para generar automáticamente proyectos de IA",
        version=version,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Microservices integration (middleware avanzado, Prometheus, etc.)
    app = setup_microservices_app(app)
    
    # Optimizaciones de performance
    app = optimize_app(app)
    enable_fast_json_serialization(app)
    optimize_async_operations()
    optimize_memory_usage()
    
    # Error handlers centralizados
    setup_error_handlers(app)
    
    # Debugging (solo si está habilitado)
    if is_debug_enabled():
        app.add_middleware(DebugMiddleware, enable_debug=True)
        app.add_middleware(PerformanceDebugMiddleware)
        setup_debug_endpoints(app)
        logger.info("Debug mode enabled")
    
    # Registrar routers por dominio
    app.include_router(projects_router)
    app.include_router(generation_router)
    app.include_router(validation_router)
    app.include_router(export_router)
    app.include_router(deployment_router)
    app.include_router(analytics_router)
    app.include_router(health_router)
    
    logger.info("Modular FastAPI application created successfully")
    return app

