"""
Routes - Definición de rutas
"""

from typing import Optional
from fastapi import FastAPI
from .service import ServerService


def setup_routes(app: FastAPI, server_service: Optional[ServerService] = None) -> None:
    """Configura las rutas de la API"""
    
    @app.get("/")
    async def root():
        """Endpoint raíz"""
        return {"message": "Suno Clone AI API"}

    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy"}

    # Aquí se añadirán más rutas según se necesiten

