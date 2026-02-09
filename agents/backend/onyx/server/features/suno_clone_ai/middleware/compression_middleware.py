"""
Compression Middleware
Compresión automática de respuestas para mejor rendimiento
"""

import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import Message

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware para compresión automática de respuestas"""
    
    def __init__(self, app, minimum_size: int = 500, compress_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compress_level = compress_level
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con compresión"""
        response = await call_next(request)
        
        # Verificar si el cliente acepta compresión
        accept_encoding = request.headers.get("accept-encoding", "").lower()
        
        if not accept_encoding:
            return response
        
        # Verificar tamaño mínimo
        if hasattr(response, "body"):
            body = response.body
            if len(body) < self.minimum_size:
                return response
            
            # Comprimir según preferencia del cliente
            if "br" in accept_encoding:
                # Brotli (mejor compresión)
                try:
                    import brotli
                    compressed = brotli.compress(body, quality=self.compress_level)
                    if len(compressed) < len(body):
                        response.body = compressed
                        response.headers["content-encoding"] = "br"
                        response.headers["content-length"] = str(len(compressed))
                        response.headers["vary"] = "Accept-Encoding"
                except ImportError:
                    # Fallback a gzip
                    pass
            
            if "gzip" in accept_encoding and "content-encoding" not in response.headers:
                try:
                    import gzip
                    compressed = gzip.compress(body, compresslevel=self.compress_level)
                    if len(compressed) < len(body):
                        response.body = compressed
                        response.headers["content-encoding"] = "gzip"
                        response.headers["content-length"] = str(len(compressed))
                        response.headers["vary"] = "Accept-Encoding"
                except ImportError:
                    pass
        
        return response















