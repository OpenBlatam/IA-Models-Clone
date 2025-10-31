"""
Export IA API - Aplicación principal FastAPI
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import export, tasks, health
from ..core.engine import get_export_engine

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Crear aplicación FastAPI."""
    app = FastAPI(
        title="Export IA API",
        description="API para exportación de documentos con IA",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Incluir rutas
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(export.router, prefix="/api/v1", tags=["export"])
    app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])
    
    # Eventos de aplicación
    @app.on_event("startup")
    async def startup_event():
        """Inicializar motor al arrancar."""
        engine = get_export_engine()
        await engine.initialize()
        logger.info("Export IA API iniciada")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cerrar motor al apagar."""
        engine = get_export_engine()
        await engine.shutdown()
        logger.info("Export IA API cerrada")
    
    return app


# Crear instancia de la aplicación
app = create_app()




