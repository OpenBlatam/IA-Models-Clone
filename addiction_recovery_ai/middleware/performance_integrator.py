"""
Performance Integrator Middleware
Integrates all performance optimizations for maximum efficiency
"""

import time
import logging
import asyncio
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import all performance optimizers
try:
    from performance.query_optimizer_advanced import get_query_optimizer
    from performance.response_streaming import get_streaming_optimizer
    from performance.adaptive_rate_limiter import get_adaptive_limiter, SystemLoad
    from performance.http2_push import get_http2_push_optimizer
    from performance.request_prioritizer import get_request_prioritizer, RequestPriority
    from performance.cdn_integration import get_cdn_optimizer
    from performance.connection_pool_advanced import get_connection_pool
    from performance.ultra_speed_optimizer import get_ultra_optimizer
    from performance.memory_optimizer import get_memory_optimizer
    PERFORMANCE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Performance modules not fully available: {e}")
    PERFORMANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceIntegratorMiddleware(BaseHTTPMiddleware):
    """
    Performance integrator middleware
    
    Integrates all performance optimizations:
    - Query optimization
    - Response streaming
    - Adaptive rate limiting
    - HTTP/2 push
    - Request prioritization
    - CDN optimization
    - Connection pooling
    - Memory optimization
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enable_all: bool = True,
        redis_url: Optional[str] = None
    ):
        super().__init__(app)
        self.enable_all = enable_all
        
        if not PERFORMANCE_AVAILABLE:
            logger.warning("⚠️ Performance modules not available")
            return
        
        # Initialize all optimizers
        self.query_optimizer = get_query_optimizer()
        self.streaming_optimizer = get_streaming_optimizer()
        self.rate_limiter = get_adaptive_limiter()
        self.http2_push = get_http2_push_optimizer()
        self.request_prioritizer = get_request_prioritizer()
        self.cdn_optimizer = get_cdn_optimizer()
        self.ultra_optimizer = get_ultra_optimizer(redis_url)
        self.memory_optimizer = get_memory_optimizer()
        
        # System metrics
        self._system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "request_count": 0,
            "error_count": 0
        }
        
        # Start metrics collection
        if self.enable_all:
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._adjust_optimizations())
        
        logger.info("✅ Performance integrator middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with integrated optimizations"""
        start_time = time.perf_counter()
        
        # Determine request priority
        priority = self._determine_priority(request)
        
        # Memory optimization before request
        if self.memory_optimizer:
            self.memory_optimizer.optimize_gc()
        
        # Adaptive rate limiting
        if self.rate_limiter:
            allowed = await self.rate_limiter.acquire(priority=priority.value)
            if not allowed:
                return Response(
                    status_code=429,
                    content='{"error": "Rate limit exceeded"}',
                    headers={"Content-Type": "application/json", "Retry-After": "1"}
                )
        
        # Process request with prioritization
        try:
            if self.request_prioritizer and priority >= RequestPriority.HIGH:
                # Use prioritizer for high-priority requests
                response = await self.request_prioritizer.submit(
                    priority,
                    call_next,
                    request
                )
            else:
                response = await call_next(request)
            
            # Record response time
            response_time = time.perf_counter() - start_time
            if self.rate_limiter:
                self.rate_limiter.record_response_time(response_time)
                self.rate_limiter.record_success()
            
            # Optimize response
            response = await self._optimize_response(request, response, response_time)
            
            # Update metrics
            self._system_metrics["request_count"] += 1
            
            return response
            
        except Exception as e:
            # Record error
            if self.rate_limiter:
                self.rate_limiter.record_error()
            
            self._system_metrics["error_count"] += 1
            logger.error(f"Request error: {e}")
            raise
    
    def _determine_priority(self, request: Request) -> RequestPriority:
        """Determine request priority"""
        path = request.url.path
        
        # Critical endpoints
        if path in ["/health", "/metrics", "/recovery/health"]:
            return RequestPriority.CRITICAL
        
        # High priority endpoints
        if any(path.startswith(p) for p in [
            "/recovery/emergency",
            "/recovery/alerts",
            "/recovery/auth/login"
        ]):
            return RequestPriority.HIGH
        
        # Low priority endpoints
        if any(path.startswith(p) for p in [
            "/recovery/analytics",
            "/recovery/reports",
            "/recovery/export"
        ]):
            return RequestPriority.LOW
        
        return RequestPriority.NORMAL
    
    async def _optimize_response(
        self,
        request: Request,
        response: Response,
        response_time: float
    ) -> Response:
        """Apply all response optimizations"""
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check if should stream
        if len(body) > 100000:  # > 100KB
            # Use streaming for large responses
            items = self._parse_json_array(body)
            if items:
                stream = self.streaming_optimizer.stream_json_array(items)
                return self.streaming_optimizer.create_streaming_response(
                    stream,
                    headers=self._get_optimized_headers(request, response)
                )
        
        # CDN optimization
        cdn_headers = self.cdn_optimizer.get_cache_headers(request.url.path)
        
        # HTTP/2 push hints
        push_resources = self.http2_push.get_push_resources(request.url.path)
        if push_resources:
            link_header = self.http2_push.generate_link_header(push_resources)
            cdn_headers["Link"] = link_header
        
        # Combine all headers
        all_headers = {
            **dict(response.headers),
            **cdn_headers,
            "X-Response-Time": f"{response_time:.4f}",
            "X-Performance-Optimized": "true"
        }
        
        return Response(
            content=body,
            status_code=response.status_code,
            headers=all_headers,
            media_type=response.media_type
        )
    
    def _parse_json_array(self, data: bytes) -> Optional[list]:
        """Parse JSON array from bytes"""
        try:
            from performance.serialization_optimizer import get_serializer
            serializer = get_serializer()
            parsed = serializer.loads(data)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return None
    
    def _get_optimized_headers(
        self,
        request: Request,
        response: Response
    ) -> Dict[str, str]:
        """Get optimized headers"""
        headers = dict(response.headers)
        
        # Add performance headers
        headers["X-Performance-Optimized"] = "true"
        headers["X-Streaming"] = "true"
        
        return headers
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        try:
            import psutil
            import os
            
            while True:
                process = psutil.Process(os.getpid())
                self._system_metrics["cpu_usage"] = process.cpu_percent()
                self._system_metrics["memory_usage"] = process.memory_percent()
                
                await asyncio.sleep(5)  # Every 5 seconds
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _adjust_optimizations(self):
        """Adjust optimizations based on system metrics"""
        try:
            while True:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Determine system load
                cpu = self._system_metrics["cpu_usage"]
                memory = self._system_metrics["memory_usage"]
                
                if cpu < 30 and memory < 50:
                    load = SystemLoad.LOW
                elif cpu < 60 and memory < 70:
                    load = SystemLoad.NORMAL
                elif cpu < 80 and memory < 85:
                    load = SystemLoad.HIGH
                else:
                    load = SystemLoad.CRITICAL
                
                # Adjust rate limiter
                if self.rate_limiter:
                    await self.rate_limiter.adjust_rate(load, cpu, memory)
                
                logger.debug(f"System load: {load.name}, CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
        except Exception as e:
            logger.error(f"Error adjusting optimizations: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "system_metrics": self._system_metrics.copy(),
        }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        if self.query_optimizer:
            stats["query_optimizer"] = self.query_optimizer.get_cache_stats()
        
        if self.request_prioritizer:
            stats["request_prioritizer"] = self.request_prioritizer.get_stats()
        
        return stats















