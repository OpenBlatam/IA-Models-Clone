"""
Compression Middleware

Middleware for compressing responses with gzip.
"""

from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging_config import get_logger

logger = get_logger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware para comprimir respuestas con gzip"""
    
    def __init__(self, app, min_size: int = 1024, compresslevel: int = 6):
        super().__init__(app)
        self.min_size = min_size
        self.compresslevel = compresslevel
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Compress response if applicable"""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding
        
        if not supports_gzip:
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Check if response body should be compressed
        if hasattr(response, "body") and response.body:
            body = response.body
            if isinstance(body, str):
                body = body.encode('utf-8')
            
            if len(body) >= self.min_size:
                try:
                    from ..compression import compress_gzip
                    compressed = compress_gzip(body, compresslevel=self.compresslevel)
                    
                    # Only compress if it actually reduces size
                    if len(compressed) < len(body):
                        response.body = compressed
                        response.headers["Content-Encoding"] = "gzip"
                        response.headers["Content-Length"] = str(len(compressed))
                        response.headers["Vary"] = "Accept-Encoding"
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")
        
        return response

