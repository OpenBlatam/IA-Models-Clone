"""
CORS Middleware
===============

Middleware para manejo de CORS (Cross-Origin Resource Sharing).
"""

import logging
from typing import List, Optional
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware

logger = logging.getLogger(__name__)


def create_cors_middleware(
    app,
    allow_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    expose_headers: Optional[List[str]] = None,
    max_age: int = 3600
):
    """
    Crear middleware CORS configurado.
    
    Args:
        app: FastAPI application
        allow_origins: Lista de orígenes permitidos (default: ["*"])
        allow_credentials: Permitir credenciales (default: True)
        allow_methods: Métodos HTTP permitidos (default: ["*"])
        allow_headers: Headers permitidos (default: ["*"])
        expose_headers: Headers expuestos (default: ["X-Request-ID", "X-Process-Time"])
        max_age: Tiempo máximo de cache en segundos (default: 3600)
        
    Returns:
        FastAPI app con middleware CORS configurado
    """
    if allow_origins is None:
        allow_origins = ["*"]
    
    if allow_methods is None:
        allow_methods = ["*"]
    
    if allow_headers is None:
        allow_headers = ["*"]
    
    if expose_headers is None:
        expose_headers = ["X-Request-ID", "X-Process-Time", "X-Error-Type"]
    
    app.add_middleware(
        FastAPICORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
        max_age=max_age,
    )
    
    logger.info("CORS middleware configured")
    return app

