"""
Aplicación Principal - Analizador de Documentos Inteligente
============================================================

Servidor FastAPI para el sistema de análisis de documentos.
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from api.router_registry import (
    get_registry,
    register_core_routers,
    register_optional_routers
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_routers(app: FastAPI) -> None:
    """
    Configurar y cargar todos los routers de la aplicación
    
    Args:
        app: Instancia de FastAPI
    """
    registry = get_registry()
    
    # Registrar routers principales y opcionales
    register_core_routers(registry)
    register_optional_routers(registry)
    
    # Cargar todos los routers
    routers = registry.load_all_routers()
    
    # Incluir routers en la aplicación
    for router in routers:
        try:
            app.include_router(router)
        except Exception as e:
            logger.error(f"Error al incluir router: {e}")
    
    # Registrar GraphQL si está disponible
    try:
        from api.graphql_schema import graphql_router
        app.include_router(graphql_router)
        logger.info("GraphQL router incluido")
    except ImportError:
        logger.debug("GraphQL no disponible (strawberry-graphql no instalado)")


# Crear aplicación FastAPI
app = FastAPI(
    title="Analizador de Documentos Inteligente",
    description="Sistema avanzado de análisis de documentos con fine-tuning",
    version="3.8.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar routers
setup_routers(app)


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("=" * 60)
    logger.info("Analizador de Documentos Inteligente v3.8.0")
    logger.info("=" * 60)
    registry = get_registry()
    logger.info(f"Routers cargados: {registry.get_loaded_count()}/{registry.get_total_registered()}")
    
    failed = registry.get_failed_imports()
    if failed:
        logger.warning(f"Routers opcionales no disponibles: {len(failed)}")
    
    logger.info("Servidor iniciado correctamente")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación"""
    logger.info("Cerrando servidor...")
    logger.info("Servidor detenido")


@app.get("/")
async def root():
    """Endpoint raíz"""
    registry = get_registry()
    failed_imports = registry.get_failed_imports()
    
    return {
        "service": "Analizador de Documentos Inteligente",
        "version": "3.8.0",
        "docs": "/docs",
        "dashboard": "/api/analizador-documentos/dashboard/",
        "health": "/api/analizador-documentos/health",
        "status": "operational",
        "routers_loaded": registry.get_loaded_count(),
        "routers_registered": registry.get_total_registered(),
        "optional_routers_failed": len(failed_imports) if failed_imports else 0
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect():
    """Redirección a dashboard"""
    from api.dashboard_routes import get_dashboard
    return await get_dashboard()


if __name__ == "__main__":
    # Configuración del servidor
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Iniciando servidor en http://{host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
