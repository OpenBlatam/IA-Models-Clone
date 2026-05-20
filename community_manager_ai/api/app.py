"""
FastAPI Application - Aplicación Principal
===========================================

Aplicación FastAPI para Community Manager AI.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .routes import posts, memes, calendar, platforms, analytics, templates, export, webhooks, dashboard, batch, backup, monitoring, version, ml
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def create_app() -> FastAPI:
    """
    Crear y configurar la aplicación FastAPI
    
    Returns:
        Instancia de FastAPI
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Sistema completo de gestión automatizada de redes sociales con IA",
        debug=settings.debug
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar orígenes
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de logging y monitoreo
    from ..middleware import LoggingMiddleware, MonitoringMiddleware
    from ..services.monitoring_service import MonitoringService
    
    monitoring_service = MonitoringService()
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MonitoringMiddleware, monitoring_service=monitoring_service)
    
    # Incluir routers
    app.include_router(posts.router)
    app.include_router(memes.router)
    app.include_router(calendar.router)
    app.include_router(platforms.router)
    app.include_router(analytics.router)
    app.include_router(templates.router)
    app.include_router(export.router)
    app.include_router(webhooks.router)
    app.include_router(dashboard.router)
    app.include_router(batch.router)
    app.include_router(backup.router)
    app.include_router(monitoring.router)
    app.include_router(version.router)
    app.include_router(ml.router)
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "app_name": settings.app_name
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs"
        }
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Manejo global de excepciones"""
        from ..utils.error_handler import ErrorHandler, AppException
        
        if isinstance(exc, AppException):
            error_response = ErrorHandler.handle_error(exc)
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response
            )
        
        logger.error(f"Error no manejado: {exc}", exc_info=True)
        error_response = ErrorHandler.handle_error(exc)
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    logger.info(f"Aplicación {settings.app_name} v{settings.app_version} creada")
    
    return app

