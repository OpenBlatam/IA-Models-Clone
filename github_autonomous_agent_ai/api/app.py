"""
Aplicación FastAPI principal
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.agent_routes import router as agent_router
from .routes.task_routes import router as task_router
from .routes.github_routes import router as github_router
from ..config.settings import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Crear aplicación FastAPI
    
    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description="Agente autónomo que se conecta a GitHub y ejecuta instrucciones continuamente"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Incluir routers con prefijos correctos
    app.include_router(agent_router, prefix="/api/agent", tags=["agent"])
    app.include_router(task_router, prefix="/api/tasks", tags=["tasks"])
    app.include_router(github_router, tags=["github"])
    app.include_router(github_router, prefix="/api/github", tags=["github"])
    
    @app.get("/")
    async def root():
        """Endpoint raíz"""
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "status": "running"
        }
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy"}
    
    return app
