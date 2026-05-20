"""
Compression Middleware for FastAPI
Automatically compresses responses
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

from ...core.infrastructure.compression import CompressionMiddleware, CompressionType

logger = logging.getLogger(__name__)


class ResponseCompressionMiddleware(BaseHTTPMiddleware):
    """Middleware to compress HTTP responses"""
    
    def __init__(self, app, min_size: int = 1024, compression_type: CompressionType = CompressionType.GZIP):
        super().__init__(app)
        self.compressor = CompressionMiddleware(min_size=min_size, compression_type=compression_type)
    
    async def dispatch(self, request: Request, call_next):
        # Check if client accepts compression
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding
        supports_deflate = "deflate" in accept_encoding
        
        if not (supports_gzip or supports_deflate):
            # Client doesn't support compression
            return await call_next(request)
        
        # Get response
        response = await call_next(request)
        
        # Check if response should be compressed
        if not isinstance(response, Response):
            return response
        
        content_type = response.headers.get("Content-Type", "")
        content_length = int(response.headers.get("Content-Length", 0))
        
        if not self.compressor.should_compress(content_type, content_length):
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Compress
        compression_type = CompressionType.GZIP if supports_gzip else CompressionType.DEFLATE
        compressed_body, encoding = self.compressor.compress(body, compression_type)
        
        if encoding:
            # Create new response with compressed body
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "Content-Encoding": encoding,
                    "Content-Length": str(len(compressed_body)),
                    "Vary": "Accept-Encoding"
                },
                media_type=content_type
            )
        
        return response















