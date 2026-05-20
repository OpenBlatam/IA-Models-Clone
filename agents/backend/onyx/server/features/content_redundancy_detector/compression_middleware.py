"""
Advanced Compression Middleware
Supports gzip, deflate, brotli compression with intelligent content detection
"""

import gzip
import zlib
from typing import Optional, Dict, Any
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import logging

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionMiddleware:
    """Advanced compression middleware with multiple algorithms"""
    
    def __init__(self, 
                 min_size: int = 1024,
                 compressible_types: Optional[list] = None,
                 compression_level: int = 6,
                 enable_brotli: bool = True):
        self.min_size = min_size
        self.compression_level = compression_level
        self.enable_brotli = enable_brotli and BROTLI_AVAILABLE
        
        # Default compressible content types
        self.compressible_types = compressible_types or [
            "application/json",
            "application/javascript",
            "application/xml",
            "text/css",
            "text/html",
            "text/javascript",
            "text/plain",
            "text/xml",
            "application/xml",
            "image/svg+xml",
            "application/rss+xml",
            "application/atom+xml"
        ]
        
        # Compression algorithms in order of preference
        self.algorithms = []
        if self.enable_brotli:
            self.algorithms.append("br")
        self.algorithms.extend(["gzip", "deflate"])
    
    def _is_compressible(self, content_type: str, content_length: int) -> bool:
        """Check if content should be compressed"""
        if content_length < self.min_size:
            return False
        
        # Check content type
        for compressible_type in self.compressible_types:
            if content_type.startswith(compressible_type):
                return True
        
        return False
    
    def _get_compression_algorithm(self, request: Request) -> Optional[str]:
        """Determine best compression algorithm based on client support"""
        accept_encoding = request.headers.get("accept-encoding", "").lower()
        
        for algorithm in self.algorithms:
            if algorithm in accept_encoding:
                return algorithm
        
        return None
    
    def _compress_content(self, content: bytes, algorithm: str) -> bytes:
        """Compress content using specified algorithm"""
        try:
            if algorithm == "br" and self.enable_brotli:
                return brotli.compress(content, quality=self.compression_level)
            elif algorithm == "gzip":
                return gzip.compress(content, compresslevel=self.compression_level)
            elif algorithm == "deflate":
                return zlib.compress(content, level=self.compression_level)
            else:
                return content
        except Exception as e:
            logger.warning(f"Compression failed for {algorithm}: {e}")
            return content
    
    def _get_content_encoding_header(self, algorithm: str) -> str:
        """Get appropriate Content-Encoding header"""
        if algorithm == "br":
            return "br"
        elif algorithm == "gzip":
            return "gzip"
        elif algorithm == "deflate":
            return "deflate"
        else:
            return "identity"
    
    async def __call__(self, request: Request, call_next):
        """Process request and response with compression"""
        # Process request
        response = await call_next(request)
        
        # Only compress if response is successful and not already compressed
        if (response.status_code < 200 or 
            response.status_code >= 300 or
            response.headers.get("content-encoding")):
            return response
        
        # Get content type and length
        content_type = response.headers.get("content-type", "").split(";")[0]
        content_length = int(response.headers.get("content-length", 0))
        
        # Check if content should be compressed
        if not self._is_compressible(content_type, content_length):
            return response
        
        # Determine compression algorithm
        algorithm = self._get_compression_algorithm(request)
        if not algorithm:
            return response
        
        # Get response body
        if hasattr(response, 'body'):
            body = response.body
        elif hasattr(response, 'content'):
            body = response.content
        else:
            # For streaming responses, we need to handle differently
            return response
        
        if not body:
            return response
        
        # Compress content
        compressed_body = self._compress_content(body, algorithm)
        
        # Only use compression if it actually reduces size
        if len(compressed_body) >= len(body):
            return response
        
        # Create new response with compressed content
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        # Add compression headers
        compressed_response.headers["content-encoding"] = self._get_content_encoding_header(algorithm)
        compressed_response.headers["content-length"] = str(len(compressed_body))
        compressed_response.headers["vary"] = "Accept-Encoding"
        
        # Add compression ratio header for monitoring
        ratio = len(compressed_body) / len(body)
        compressed_response.headers["x-compression-ratio"] = f"{ratio:.2f}"
        compressed_response.headers["x-compression-algorithm"] = algorithm
        
        return compressed_response


class StreamingCompressionMiddleware:
    """Middleware for compressing streaming responses"""
    
    def __init__(self, chunk_size: int = 8192, compression_level: int = 6):
        self.chunk_size = chunk_size
        self.compression_level = compression_level
    
    def _compress_stream(self, content_stream, algorithm: str):
        """Compress streaming content"""
        if algorithm == "gzip":
            compressor = gzip.compressobj(compresslevel=self.compression_level)
        elif algorithm == "deflate":
            compressor = zlib.compressobj(level=self.compression_level)
        elif algorithm == "br" and BROTLI_AVAILABLE:
            compressor = brotli.Compressor(quality=self.compression_level)
        else:
            # No compression
            async for chunk in content_stream:
                yield chunk
            return
        
        async for chunk in content_stream:
            if algorithm == "br" and BROTLI_AVAILABLE:
                compressed_chunk = compressor.process(chunk)
            else:
                compressed_chunk = compressor.compress(chunk)
            
            if compressed_chunk:
                yield compressed_chunk
        
        # Flush remaining data
        if algorithm == "br" and BROTLI_AVAILABLE:
            final_chunk = compressor.finish()
        else:
            final_chunk = compressor.flush()
        
        if final_chunk:
            yield final_chunk
    
    async def __call__(self, request: Request, call_next):
        """Process streaming response with compression"""
        response = await call_next(request)
        
        # Only handle streaming responses
        if not isinstance(response, StreamingResponse):
            return response
        
        # Check if compression is supported
        accept_encoding = request.headers.get("accept-encoding", "").lower()
        if not any(alg in accept_encoding for alg in ["gzip", "deflate", "br"]):
            return response
        
        # Determine compression algorithm
        algorithm = None
        if "br" in accept_encoding and BROTLI_AVAILABLE:
            algorithm = "br"
        elif "gzip" in accept_encoding:
            algorithm = "gzip"
        elif "deflate" in accept_encoding:
            algorithm = "deflate"
        
        if not algorithm:
            return response
        
        # Create compressed streaming response
        compressed_stream = self._compress_stream(response.body_iterator, algorithm)
        
        compressed_response = StreamingResponse(
            compressed_stream,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        # Add compression headers
        compressed_response.headers["content-encoding"] = algorithm
        compressed_response.headers["vary"] = "Accept-Encoding"
        compressed_response.headers["x-compression-algorithm"] = algorithm
        
        return compressed_response


# Global compression middleware instances
compression_middleware = CompressionMiddleware()
streaming_compression_middleware = StreamingCompressionMiddleware()


def add_compression_headers(response: Response, 
                          original_size: int, 
                          compressed_size: int,
                          algorithm: str) -> Response:
    """Add compression metadata headers to response"""
    response.headers["x-original-size"] = str(original_size)
    response.headers["x-compressed-size"] = str(compressed_size)
    response.headers["x-compression-ratio"] = f"{compressed_size/original_size:.2f}"
    response.headers["x-compression-algorithm"] = algorithm
    response.headers["x-compression-savings"] = f"{((original_size - compressed_size) / original_size * 100):.1f}%"
    
    return response





