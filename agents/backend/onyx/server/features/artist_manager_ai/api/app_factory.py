"""
App Factory
===========

Factory para crear aplicación FastAPI con todas las configuraciones.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import os

from .routes import router
from ..middleware import AuthMiddleware, LoggingMiddleware, RateLimitMiddleware
from ..health import HealthService
from ..utils import RateLimiter
from ..auth import AuthService


def create_app(
    title: str = "Artist Manager AI",
    version: str = "1.0.0",
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
    enable_cors: bool = True
) -> FastAPI:
    """
    Crear aplicación FastAPI.
    
    Args:
        title: Título de la API
        version: Versión
        enable_auth: Habilitar autenticación
        enable_rate_limit: Habilitar rate limiting
        enable_cors: Habilitar CORS
    
    Returns:
        Aplicación FastAPI configurada
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Sistema completo de gestión para artistas con IA",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configurar según necesidad
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Rate Limiter
    rate_limiter = None
    if enable_rate_limit:
        rate_limiter = RateLimiter()
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter,
            max_requests=100,
            window_seconds=60
        )
    
    # Auth
    auth_service = None
    if enable_auth:
        auth_service = AuthService()
        app.add_middleware(
            AuthMiddleware,
            auth_service=auth_service,
            public_paths=["/health", "/docs", "/redoc", "/openapi.json"]
        )
    
    # Logging
    app.add_middleware(LoggingMiddleware)
    
    # Routes
    app.include_router(router)
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        health_service = HealthService()
        return await health_service.check_all()
    
    # Custom OpenAPI
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=title,
            version=version,
            description="Sistema completo de gestión para artistas con IA integrada",
            routes=app.routes,
        )
        
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app

