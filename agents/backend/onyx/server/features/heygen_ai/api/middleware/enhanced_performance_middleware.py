from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import gzip
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis
import psutil
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Enhanced Performance Middleware for HeyGen AI FastAPI
Advanced middleware with request batching, intelligent caching, and monitoring.
"""


logger = structlog.get_logger()

# =============================================================================
# Enhanced Performance Metrics
# =============================================================================

class EnhancedPerformanceMetrics:
    """Enhanced performance metrics with detailed tracking."""
    
    def __init__(self) -> Any:
        # Request metrics
        self.request_times: deque = deque(maxlen=2000)
        self.response_sizes: deque = deque(maxlen=2000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Cache metrics
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.cache_size_bytes: int = 0
        
        # Compression metrics
        self.compression_savings: float = 0.0
        self.compressed_responses: int = 0
        
        # Rate limiting metrics
        self.rate_limited_requests: int = 0
        self.rate_limit_violations: Dict[str, int] = defaultdict(int)
        
        # Batch processing metrics
        self.batched_requests: int = 0
        self.batch_efficiency: float = 0.0
        
        # Resource usage
        self.cpu_usage_history: deque = deque(maxlen=100)
        self.memory_usage_history: deque = deque(maxlen=100)
        
        # Timing
        self.start_time = datetime.now(timezone.utc)
        
        # Slow requests
        self.slow_requests: List[Dict[str, Any]] = []
        self.slow_request_threshold_ms = 1000
    
    def record_request(
        self,
        duration: float,
        endpoint: str,
        status_code: int,
        response_size: Optional[int] = None,
        was_cached: bool = False,
        was_compressed: bool = False,
        was_batched: bool = False
    ):
        """Record comprehensive request metrics."""
        # Basic metrics
        self.request_times.append(duration)
        
        if response_size:
            self.response_sizes.append(response_size)
        
        # Endpoint-specific metrics
        if endpoint:
            self.endpoint_performance[endpoint].append(duration)
            if len(self.endpoint_performance[endpoint]) > 200:
                self.endpoint_performance[endpoint] = self.endpoint_performance[endpoint][-200:]
        
        # Cache metrics
        if was_cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Compression metrics
        if was_compressed:
            self.compressed_responses += 1
        
        # Batch metrics
        if was_batched:
            self.batched_requests += 1
        
        # Error tracking
        if status_code >= 400:
            error_type = f"{status_code // 100}xx"
            self.error_counts[error_type] += 1
        
        # Slow request tracking
        if duration > self.slow_request_threshold_ms:
            self.slow_requests.append({
                "endpoint": endpoint,
                "duration_ms": duration,
                "status_code": status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "was_cached": was_cached,
                "was_compressed": was_compressed
            })
            
            # Keep only last 50 slow requests
            if len(self.slow_requests) > 50:
                self.slow_requests = self.slow_requests[-50:]
    
    def record_resource_usage(self) -> Any:
        """Record current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_info.percent)
        except Exception:
            pass
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self.request_times:
            return {"message": "No data available"}
        
        request_times = list(self.request_times)
        response_sizes = list(self.response_sizes)
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = (len(sorted_data) - 1) * p / 100
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower = sorted_data[int(index)]
                upper = sorted_data[int(index) + 1]
                return lower + (upper - lower) * (index % 1)
        
        # Resource averages
        avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else 0
        avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
        
        return {
            "request_metrics": {
                "total_requests": len(request_times),
                "avg_response_time_ms": sum(request_times) / len(request_times),
                "min_response_time_ms": min(request_times),
                "max_response_time_ms": max(request_times),
                "p50_response_time_ms": percentile(request_times, 50),
                "p90_response_time_ms": percentile(request_times, 90),
                "p95_response_time_ms": percentile(request_times, 95),
                "p99_response_time_ms": percentile(request_times, 99),
                "rps": len(request_times) / max(1, (datetime.now(timezone.utc) - self.start_time).total_seconds())
            },
            "cache_metrics": {
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "cache_size_bytes": self.cache_size_bytes
            },
            "compression_metrics": {
                "compressed_responses": self.compressed_responses,
                "compression_savings_bytes": self.compression_savings,
                "compression_rate": self.compressed_responses / len(request_times) if request_times else 0
            },
            "rate_limiting": {
                "rate_limited_requests": self.rate_limited_requests,
                "violations_by_ip": dict(self.rate_limit_violations)
            },
            "resource_usage": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "current_cpu_percent": self.cpu_usage_history[-1] if self.cpu_usage_history else 0,
                "current_memory_percent": self.memory_usage_history[-1] if self.memory_usage_history else 0
            },
            "error_metrics": dict(self.error_counts),
            "slow_requests": {
                "count": len(self.slow_requests),
                "threshold_ms": self.slow_request_threshold_ms,
                "recent_slow_requests": self.slow_requests[-10:]  # Last 10
            },
            "batch_metrics": {
                "batched_requests": self.batched_requests,
                "batch_efficiency": self.batch_efficiency
            }
        }

# =============================================================================
# Request Batcher
# =============================================================================

class RequestBatcher:
    """Intelligent request batching for similar operations."""
    
    def __init__(self) -> Any:
        self.batch_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_timeout = 0.05  # 50ms batch window
        self.max_batch_size = 10
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def add_to_batch(
        self,
        batch_key: str,
        request: Request,
        response_future: asyncio.Future
    ) -> bool:
        """Add request to batch if batchable."""
        if not self._is_batchable(request):
            return False
        
        # Add to batch
        batch_item = {
            "request": request,
            "response_future": response_future,
            "timestamp": time.time()
        }
        
        self.batch_queue[batch_key].append(batch_item)
        
        # Start batch processing if not already running
        if batch_key not in self.processing_tasks:
            self.processing_tasks[batch_key] = asyncio.create_task(
                self._process_batch(batch_key)
            )
        
        return True
    
    def _is_batchable(self, request: Request) -> bool:
        """Determine if request can be batched."""
        # Only batch GET requests for now
        if request.method != "GET":
            return False
        
        # Check if endpoint supports batching
        batchable_endpoints = [
            "/api/users/batch",
            "/api/videos/batch",
            "/api/analytics/batch"
        ]
        
        return any(endpoint in str(request.url.path) for endpoint in batchable_endpoints)
    
    async def _process_batch(self, batch_key: str):
        """Process a batch of requests."""
        await asyncio.sleep(self.batch_timeout)
        
        if batch_key not in self.batch_queue or not self.batch_queue[batch_key]:
            self.processing_tasks.pop(batch_key, None)
            return
        
        batch = self.batch_queue[batch_key].copy()
        self.batch_queue[batch_key].clear()
        
        try:
            # Process batch requests together
            responses = await self._execute_batch(batch)
            
            # Send responses to waiting futures
            for item, response in zip(batch, responses):
                if not item["response_future"].done():
                    item["response_future"].set_result(response)
        
        except Exception as e:
            # Send error to all waiting futures
            for item in batch:
                if not item["response_future"].done():
                    item["response_future"].set_exception(e)
        
        finally:
            self.processing_tasks.pop(batch_key, None)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]) -> List[Response]:
        """Execute batch of requests efficiently."""
        # Placeholder for actual batch processing logic
        responses = []
        
        for item in batch:
            # Simulate batch processing
            await asyncio.sleep(0.001)  # Minimal processing time
            responses.append(JSONResponse({"batched": True, "result": "processed"}))
        
        return responses

# =============================================================================
# Intelligent Cache Manager
# =============================================================================

class IntelligentCacheManager:
    """Intelligent caching with adaptive TTL and invalidation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.adaptive_ttl: Dict[str, int] = {}
        self.max_memory_cache_size = 1000
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate intelligent cache key."""
        # Include method, path, and query parameters
        key_parts = [
            request.method,
            str(request.url.path),
            str(sorted(request.query_params.items()))
        ]
        
        # Include relevant headers
        cache_relevant_headers = ["authorization", "accept-language", "user-agent"]
        for header in cache_relevant_headers:
            if header in request.headers:
                key_parts.append(f"{header}:{request.headers[header]}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cacheable(self, request: Request, response: Response) -> bool:
        """Determine if request/response should be cached."""
        # Only cache GET requests
        if request.method != "GET":
            return False
        
        # Only cache successful responses
        if response.status_code != 200:
            return False
        
        # Don't cache if explicit cache control headers
        if "no-cache" in request.headers.get("cache-control", "").lower():
            return False
        
        # Check for cacheable endpoints
        cacheable_patterns = [
            "/api/users/",
            "/api/videos/",
            "/api/analytics/",
            "/health",
            "/metrics"
        ]
        
        return any(pattern in str(request.url.path) for pattern in cacheable_patterns)
    
    async def get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response if available."""
        cache_key = self._generate_cache_key(request)
        endpoint = str(request.url.path)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            cache_entry = self.memory_cache[cache_key]
            if cache_entry["expires_at"] > time.time():
                self.cache_stats[endpoint]["hits"] += 1
                return JSONResponse(
                    content=cache_entry["content"],
                    headers={"X-Cache": "HIT-MEMORY"}
                )
            else:
                del self.memory_cache[cache_key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"response_cache:{cache_key}")
                if cached_data:
                    cache_entry = json.loads(cached_data)
                    self.cache_stats[endpoint]["hits"] += 1
                    
                    # Store in memory cache for faster access
                    self._store_in_memory_cache(cache_key, cache_entry["content"], 300)
                    
                    return JSONResponse(
                        content=cache_entry["content"],
                        headers={"X-Cache": "HIT-REDIS"}
                    )
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
        
        self.cache_stats[endpoint]["misses"] += 1
        return None
    
    async def store_response(
        self,
        request: Request,
        response: Response,
        content: Any
    ):
        """Store response in cache with adaptive TTL."""
        if not self._is_cacheable(request, response):
            return
        
        cache_key = self._generate_cache_key(request)
        endpoint = str(request.url.path)
        
        # Determine adaptive TTL
        ttl = self._calculate_adaptive_ttl(endpoint)
        
        # Store in memory cache
        self._store_in_memory_cache(cache_key, content, ttl)
        
        # Store in Redis cache
        if self.redis_client:
            try:
                cache_entry = {
                    "content": content,
                    "cached_at": time.time(),
                    "endpoint": endpoint
                }
                
                await self.redis_client.setex(
                    f"response_cache:{cache_key}",
                    ttl,
                    json.dumps(cache_entry, default=str)
                )
            except Exception as e:
                logger.error(f"Redis cache store error: {e}")
    
    def _store_in_memory_cache(self, cache_key: str, content: Any, ttl: int):
        """Store response in memory cache with LRU eviction."""
        # LRU eviction if cache is full
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]["cached_at"]
            )
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = {
            "content": content,
            "expires_at": time.time() + ttl,
            "cached_at": time.time()
        }
    
    def _calculate_adaptive_ttl(self, endpoint: str) -> int:
        """Calculate adaptive TTL based on endpoint performance."""
        base_ttl = 300  # 5 minutes default
        
        # Get cache hit rate for this endpoint
        stats = self.cache_stats[endpoint]
        total_requests = stats["hits"] + stats["misses"]
        
        if total_requests > 10:  # Enough data for adaptation
            hit_rate = stats["hits"] / total_requests
            
            # Increase TTL for high hit rate endpoints
            if hit_rate > 0.8:
                return base_ttl * 2  # 10 minutes
            elif hit_rate > 0.6:
                return int(base_ttl * 1.5)  # 7.5 minutes
            elif hit_rate < 0.3:
                return base_ttl // 2  # 2.5 minutes
        
        return base_ttl
    
    async def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern:
            # Selective invalidation
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(f"response_cache:*{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis cache invalidation error: {e}")
        else:
            # Clear all cache
            self.memory_cache.clear()
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys("response_cache:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis cache clear error: {e}")

# =============================================================================
# Enhanced Performance Middleware
# =============================================================================

class EnhancedPerformanceMiddleware(BaseHTTPMiddleware):
    """Enhanced performance middleware with advanced features."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_client: Optional[redis.Redis] = None,
        enable_caching: bool = True,
        enable_compression: bool = True,
        enable_batching: bool = True,
        enable_rate_limiting: bool = True,
        enable_monitoring: bool = True,
        compression_threshold: int = 1024,
        compression_level: int = 6,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        exclude_paths: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None
    ):
        
    """__init__ function."""
super().__init__(app)
        self.redis_client = redis_client
        self.enable_caching = enable_caching
        self.enable_compression = enable_compression
        self.enable_batching = enable_batching
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_monitoring = enable_monitoring
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.exclude_paths = exclude_paths or []
        self.exclude_methods = exclude_methods or ["OPTIONS"]
        
        # Initialize components
        self.metrics = EnhancedPerformanceMetrics()
        self.cache_manager = IntelligentCacheManager(redis_client)
        self.request_batcher = RequestBatcher()
        
        # Rate limiting
        self.rate_limit_data: Dict[str, List[float]] = defaultdict(list)
        
        # Start monitoring task
        if self.enable_monitoring:
            asyncio.create_task(self._monitoring_loop())
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Enhanced request processing with all optimizations."""
        start_time = time.time()
        
        # Skip processing for excluded paths/methods
        if self._should_exclude(request):
            response = await call_next(request)
            return response
        
        # Rate limiting check
        if self.enable_rate_limiting and not await self._check_rate_limit(request):
            self.metrics.rate_limited_requests += 1
            client_ip = self._get_client_ip(request)
            self.metrics.rate_limit_violations[client_ip] += 1
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )
        
        # Try cache first
        if self.enable_caching:
            cached_response = await self.cache_manager.get_cached_response(request)
            if cached_response:
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_request(
                    duration=duration_ms,
                    endpoint=str(request.url.path),
                    status_code=cached_response.status_code,
                    response_size=None,
                    was_cached=True
                )
                return cached_response
        
        # Try batching for eligible requests
        batched = False
        if self.enable_batching:
            batch_key = self._generate_batch_key(request)
            response_future = asyncio.Future()
            
            if await self.request_batcher.add_to_batch(batch_key, request, response_future):
                try:
                    response = await asyncio.wait_for(response_future, timeout=1.0)
                    batched = True
                    
                    duration_ms = (time.time() - start_time) * 1000
                    self.metrics.record_request(
                        duration=duration_ms,
                        endpoint=str(request.url.path),
                        status_code=response.status_code,
                        was_batched=True
                    )
                    return response
                except asyncio.TimeoutError:
                    pass  # Fall through to normal processing
        
        # Normal request processing
        response = await call_next(request)
        
        # Record response body for caching and compression
        response_body = None
        if hasattr(response, 'body'):
            response_body = response.body
        elif isinstance(response, JSONResponse):
            response_body = json.dumps(response.body).encode() if hasattr(response, 'body') else b""
        
        # Apply compression if beneficial
        compressed = False
        if (self.enable_compression and 
            response_body and 
            len(response_body) > self.compression_threshold):
            
            compressed_body = gzip.compress(response_body, compresslevel=self.compression_level)
            compression_ratio = len(compressed_body) / len(response_body)
            
            # Only use compression if it saves at least 10%
            if compression_ratio < 0.9:
                self.metrics.compression_savings += len(response_body) - len(compressed_body)
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(compressed_body))
                
                # Create new response with compressed body
                if isinstance(response, JSONResponse):
                    response = Response(
                        content=compressed_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                compressed = True
        
        # Cache response if enabled
        if self.enable_caching and response_body:
            try:
                if isinstance(response, JSONResponse):
                    content = json.loads(response_body.decode()) if isinstance(response_body, bytes) else response_body
                else:
                    content = response_body.decode() if isinstance(response_body, bytes) else response_body
                
                await self.cache_manager.store_response(request, response, content)
            except Exception as e:
                logger.error(f"Cache storage error: {e}")
        
        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        response_size = len(response_body) if response_body else None
        
        self.metrics.record_request(
            duration=duration_ms,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            response_size=response_size,
            was_cached=False,
            was_compressed=compressed,
            was_batched=batched
        )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        if compressed:
            response.headers["X-Compression"] = "gzip"
        if batched:
            response.headers["X-Batched"] = "true"
        
        return response
    
    def _should_exclude(self, request: Request) -> bool:
        """Check if request should be excluded from processing."""
        # Check method exclusions
        if request.method in self.exclude_methods:
            return True
        
        # Check path exclusions
        path = str(request.url.path)
        return any(excluded in path for excluded in self.exclude_paths)
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limits."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - self.rate_limit_window
        self.rate_limit_data[client_ip] = [
            timestamp for timestamp in self.rate_limit_data[client_ip]
            if timestamp > cutoff_time
        ]
        
        # Check limit
        if len(self.rate_limit_data[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.rate_limit_data[client_ip].append(current_time)
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return getattr(request.client, "host", "unknown")
    
    def _generate_batch_key(self, request: Request) -> str:
        """Generate batching key for similar requests."""
        return f"{request.method}:{request.url.path}"
    
    async def _monitoring_loop(self) -> Any:
        """Background monitoring loop."""
        while True:
            try:
                self.metrics.record_resource_usage()
                await asyncio.sleep(10)  # Record every 10 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return self.metrics.get_detailed_stats()

# =============================================================================
# Factory Function
# =============================================================================

def create_enhanced_performance_middleware(
    redis_client: Optional[redis.Redis] = None,
    **kwargs
) -> EnhancedPerformanceMiddleware:
    """Create enhanced performance middleware with configuration."""
    return EnhancedPerformanceMiddleware(
        app=None,  # Will be set by FastAPI
        redis_client=redis_client,
        **kwargs
    ) 