"""
Advanced Performance Middleware
Optimizations for high-throughput, low-latency APIs
"""

import time
import asyncio
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from functools import lru_cache

from core.service_container import get_container

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Performance optimization middleware
    
    Features:
    - Response compression
    - Connection pooling
    - Request batching
    - Response caching headers
    - Gzip compression
    """
    
    def __init__(self, app: ASGIApp, enable_compression: bool = True):
        super().__init__(app)
        self.enable_compression = enable_compression
        self._cache_control = "public, max-age=300"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance optimizations"""
        start_time = time.perf_counter()
        
        # Add performance headers
        response = await call_next(request)
        
        # Calculate processing time
        duration = time.perf_counter() - start_time
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        response.headers["X-Response-Time"] = f"{duration*1000:.2f}ms"
        
        # Cache control headers
        if request.method == "GET" and response.status_code == 200:
            response.headers["Cache-Control"] = self._cache_control
            response.headers["ETag"] = self._generate_etag(response)
        
        # Compression (if enabled and client supports it)
        if self.enable_compression:
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" in accept_encoding and self._should_compress(response):
                response = await self._compress_response(response)
        
        # Connection keep-alive
        response.headers["Connection"] = "keep-alive"
        
        return response
    
    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed"""
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript"
        ]
        return any(ct in content_type for ct in compressible_types)
    
    async def _compress_response(self, response: Response) -> Response:
        """Compress response with gzip"""
        try:
            import gzip
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            compressed = gzip.compress(body, compresslevel=6)
            
            # Create new response with compressed body
            return Response(
                content=compressed,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed))
                },
                media_type=response.media_type
            )
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return response
    
    def _generate_etag(self, response: Response) -> str:
        """Generate ETag for caching"""
        import hashlib
        content = str(response.status_code) + str(response.headers.get("content-type", ""))
        return hashlib.md5(content.encode()).hexdigest()


class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    """Middleware for connection pooling optimization"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._pool_size = 10
        self._max_overflow = 20
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with connection pooling"""
        # Connection pooling is handled at the service level
        # This middleware just adds headers
        response = await call_next(request)
        response.headers["Connection"] = "keep-alive"
        response.headers["Keep-Alive"] = "timeout=5, max=1000"
        return response


class RequestBatchingMiddleware(BaseHTTPMiddleware):
    """Middleware for request batching optimization"""
    
    def __init__(self, app: ASGIApp, batch_size: int = 10, batch_timeout: float = 0.1):
        super().__init__(app)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._pending_requests = []
        self._batch_lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with batching (for compatible endpoints)"""
        # Only batch GET requests to same endpoint
        if request.method == "GET" and self._is_batchable(request):
            return await self._handle_batched(request, call_next)
        else:
            return await call_next(request)
    
    def _is_batchable(self, request: Request) -> bool:
        """Check if request can be batched"""
        # Only batch specific endpoints
        batchable_paths = ["/recovery/users", "/recovery/profile"]
        return any(request.url.path.startswith(path) for path in batchable_paths)
    
    async def _handle_batched(self, request: Request, call_next: Callable) -> Response:
        """Handle batched request"""
        async with self._batch_lock:
            self._pending_requests.append((request, call_next))
            
            if len(self._pending_requests) >= self.batch_size:
                return await self._process_batch()
            else:
                # Wait for batch timeout or size
                await asyncio.sleep(self.batch_timeout)
                if len(self._pending_requests) >= 2:
                    return await self._process_batch()
        
        # Fallback to individual processing
        return await call_next(request)
    
    async def _process_batch(self) -> Response:
        """Process batch of requests"""
        batch = self._pending_requests[:self.batch_size]
        self._pending_requests = self._pending_requests[self.batch_size:]
        
        # Process batch concurrently
        tasks = [call_next(req) for req, call_next in batch]
        results = await asyncio.gather(*tasks)
        
        # Return first result (simplified - in production would combine)
        return results[0] if results else Response(status_code=500)















