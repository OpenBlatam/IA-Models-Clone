"""
Compression - Compresión de requests/responses
==============================================

Middleware para comprimir respuestas y descomprimir requests.
"""

import gzip
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware para compresión de respuestas."""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 500):
        """
        Inicializar middleware.
        
        Args:
            app: Aplicación ASGI.
            minimum_size: Tamaño mínimo para comprimir (bytes).
        """
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con compresión."""
        # Verificar si el cliente acepta compresión
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding
        
        # Procesar request
        response = await call_next(request)
        
        # Comprimir respuesta si es necesario
        if supports_gzip and response.status_code == 200:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Solo comprimir si el tamaño es suficiente
            if len(body) >= self.minimum_size:
                compressed = gzip.compress(body)
                response = Response(
                    content=compressed,
                    status_code=response.status_code,
                    headers={
                        **response.headers,
                        "Content-Encoding": "gzip",
                        "Content-Length": str(len(compressed))
                    },
                    media_type=response.media_type
                )
            else:
                response = Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
        
        return response




