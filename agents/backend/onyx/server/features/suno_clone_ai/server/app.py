"""
FastAPI App - Aplicación FastAPI
"""

from typing import Optional
from fastapi import FastAPI
from configs.settings import Settings


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Crea la aplicación FastAPI"""
    settings = settings or Settings()
    app = FastAPI(
        title="Suno Clone AI",
        description="Sistema de generación de música con IA",
        version="1.0.0"
    )
    return app

