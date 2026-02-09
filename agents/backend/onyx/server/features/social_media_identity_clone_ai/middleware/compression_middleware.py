"""
Middleware para compresión de respuestas
Optimiza el tamaño de las respuestas HTTP
"""

import gzip
import logging
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware para comprimir respuestas HTTP con gzip"""
    
    # Tipos de contenido que se deben comprimir
    COMPRESSIBLE_TYPES = {
        "application/json",
        "application/javascript",
        "text/html",
        "text/css",
        "text/xml",
        "text/plain",
        "application/xml",
        "application/xhtml+xml",
        "text/javascript",
        "application/json",
    }
    
    # Tamaño mínimo para comprimir (bytes)
    MIN_SIZE = 500
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesa request y comprime respuesta si es necesario"""
        response = await call_next(request)
        
        # Verificar si el cliente acepta compresión
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # Verificar tipo de contenido
        content_type = response.headers.get("Content-Type", "")
        if not any(ct in content_type for ct in self.COMPRESSIBLE_TYPES):
            return response
        
        # Verificar tamaño
        if hasattr(response, "body"):
            body = response.body
            if len(body) < self.MIN_SIZE:
                return response
            
            # Comprimir
            compressed = gzip.compress(body, compresslevel=6)  # Balance velocidad/tamaño
            
            # Crear nueva respuesta comprimida
            headers = dict(response.headers)
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(compressed))
            headers["Vary"] = "Accept-Encoding"
            
            return Response(
                content=compressed,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type
            )
        
        return response

