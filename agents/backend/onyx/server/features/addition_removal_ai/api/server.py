"""
FastAPI Server - Servidor principal de la API
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Dict, Any

from .routes import router
from ..core.rate_limiter import RateLimiter, RateLimitMiddleware

logger = logging.getLogger(__name__)


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Crear y configurar la aplicación FastAPI.

    Args:
        config: Configuración opcional

    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title="Addition Removal AI API",
        description="API para el sistema de IA de adiciones y eliminaciones",
        version="1.0.0"
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
    
    # Rate limiting
    rate_limiter = RateLimiter(max_requests=100, time_window=60)
    app.middleware("http")(RateLimitMiddleware(rate_limiter))

    # Incluir rutas
    app.include_router(router, prefix="/api/v1")
    
    # Agregar WebSocket si está disponible
    try:
        from .websocket import websocket_endpoint
        from ..core.editor import ContentEditor
        
        @app.websocket("/ws")
        async def websocket_route(websocket):
            editor_instance = ContentEditor(config)
            await websocket_endpoint(websocket, editor_instance)
    except Exception as e:
        logger.warning(f"WebSocket no disponible: {e}")

    @app.get("/")
    async def root():
        return {
            "service": "Addition Removal AI",
            "version": "1.0.0",
            "status": "running"
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app

