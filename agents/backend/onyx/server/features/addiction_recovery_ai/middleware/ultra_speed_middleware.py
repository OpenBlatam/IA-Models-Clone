"""
Ultra-Speed Middleware
Maximum performance optimizations for ultra-fast request processing
"""

import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
# Import asyncio for prefetching
import asyncio

from performance.ultra_speed_optimizer import (
    get_ultra_optimizer,
    get_precomputer,
    get_early_optimizer
)
from performance.serialization_optimizer import get_serializer

# Try to import Brotli
try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

logger = logging.getLogger(__name__)


class UltraSpeedMiddleware(BaseHTTPMiddleware):
    """
    Ultra-speed optimization middleware
    
    Features:
    - Ultra-fast Redis caching
    - Brotli compression
    - Request coalescing
    - Response pre-computation
    - Early response optimization
    - Smart prefetching
    """
    
    def __init__(
        self,
        app: ASGIApp,
        redis_url: Optional[str] = None,
        enable_brotli: bool = True,
        enable_coalescing: bool = True,
        enable_prefetch: bool = True
    ):
        super().__init__(app)
        self.optimizer = get_ultra_optimizer(redis_url)
        self.precomputer = get_precomputer()
        self.early_optimizer = get_early_optimizer()
        self.serializer = get_serializer()
        self.enable_brotli = enable_brotli and BROTLI_AVAILABLE
        self.enable_coalescing = enable_coalescing
        self.enable_prefetch = enable_prefetch
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with ultra-speed optimizations"""
        start_time = time.perf_counter()
        
        # Check for early response
        early_response = self.early_optimizer.get_early_response(request.url.path)
        if early_response:
            response = Response(
                content=early_response,
                status_code=200,
                media_type="application/json",
                headers={
                    "X-Early-Response": "true",
                    "X-Process-Time": f"{time.perf_counter() - start_time:.4f}"
                }
            )
            return response
        
        # Check for precomputed response
        precomputed = self.precomputer.get_precomputed(request.url.path)
        if precomputed and request.method == "GET":
            response = Response(
                content=precomputed,
                status_code=200,
                media_type="application/json",
                headers={
                    "X-Precomputed": "true",
                    "X-Process-Time": f"{time.perf_counter() - start_time:.4f}"
                }
            )
            return response
        
        # Generate cache key
        cache_key = self.optimizer.generate_cache_key(
            request.method,
            request.url.path,
            str(request.url.query)
        )
        
        # Check cache for GET requests
        if request.method == "GET":
            cached_response = await self.optimizer.get_cached(cache_key)
            if cached_response:
                # Determine compression
                accept_encoding = request.headers.get("accept-encoding", "")
                compression = self.optimizer.get_compression_algorithm(accept_encoding)
                
                response = self._create_cached_response(
                    cached_response,
                    compression,
                    time.perf_counter() - start_time
                )
                return response
        
        # Process request (with coalescing if enabled)
        if self.enable_coalescing and request.method == "GET":
            async def process_request():
                return await call_next(request)
            
            response = await self.optimizer.coalesce_requests(
                cache_key,
                process_request
            )
        else:
            response = await call_next(request)
        
        # Optimize response
        response = await self._optimize_response(
            request,
            response,
            cache_key,
            time.perf_counter() - start_time
        )
        
        # Prefetch likely data
        if self.enable_prefetch and request.method == "GET" and response.status_code == 200:
            user_id = request.path_params.get("user_id")
            if user_id:
                asyncio.create_task(
                    self.optimizer.prefetch_likely_data(user_id, request.url.path)
                )
        
        return response
    
    def _create_cached_response(
        self,
        content: bytes,
        compression: Optional[str],
        process_time: float
    ) -> Response:
        """Create response from cached content"""
        headers = {
            "X-Cached": "true",
            "X-Process-Time": f"{process_time:.4f}",
            "Content-Type": "application/json"
        }
        
        # Apply compression if requested
        if compression == "br" and self.enable_brotli:
            compressed = self.optimizer.compress_brotli(content)
            headers["Content-Encoding"] = "br"
            headers["Content-Length"] = str(len(compressed))
            return Response(content=compressed, status_code=200, headers=headers)
        elif compression == "gzip":
            import gzip
            compressed = gzip.compress(content, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(compressed))
            return Response(content=compressed, status_code=200, headers=headers)
        
        headers["Content-Length"] = str(len(content))
        return Response(content=content, status_code=200, headers=headers)
    
    async def _optimize_response(
        self,
        request: Request,
        response: Response,
        cache_key: str,
        process_time: float
    ) -> Response:
        """Optimize response with compression and caching"""
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Ultra-Speed"] = "true"
        
        # Only optimize successful GET responses
        if request.method != "GET" or response.status_code != 200:
            return response
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Cache response
        if len(body) > 0:
            await self.optimizer.set_cached(cache_key, body, ttl=300)
        
        # Determine compression
        accept_encoding = request.headers.get("accept-encoding", "")
        compression = self.optimizer.get_compression_algorithm(accept_encoding)
        
        # Apply compression
        if compression == "br" and self.enable_brotli and len(body) > 1024:
            compressed = self.optimizer.compress_brotli(body)
            response.headers["Content-Encoding"] = "br"
            response.headers["Content-Length"] = str(len(compressed))
            return Response(
                content=compressed,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        elif compression == "gzip" and len(body) > 1024:
            import gzip
            compressed = gzip.compress(body, compresslevel=6)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))
            return Response(
                content=compressed,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Return original response with updated headers
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

