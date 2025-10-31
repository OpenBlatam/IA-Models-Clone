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
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import json
from collections import defaultdict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import queue
import signal
import os
from fastapi import Request, Response, HTTPException, BackgroundTasks
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Async Middleware for HeyGen AI FastAPI
Middleware and utilities to prevent blocking operations in routes.
"""



logger = structlog.get_logger()

# =============================================================================
# Middleware Types
# =============================================================================

class MiddlewareType(Enum):
    """Middleware type enumeration."""
    ASYNC_ROUTE = "async_route"
    NON_BLOCKING = "non_blocking"
    BACKGROUND_PROCESSING = "background_processing"
    RATE_LIMITING = "rate_limiting"
    CIRCUIT_BREAKER = "circuit_breaker"
    CACHING = "caching"
    MONITORING = "monitoring"

class BlockingDetectionLevel(Enum):
    """Blocking detection level."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    STRICT = "strict"

@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    middleware_type: MiddlewareType
    blocking_detection: BlockingDetectionLevel = BlockingDetectionLevel.BASIC
    timeout: float = 30.0
    max_concurrent: int = 100
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    rate_limit: int = 100  # requests per second
    cache_ttl: int = 300
    background_workers: int = 4

@dataclass
class RouteMetrics:
    """Route performance metrics."""
    route_path: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    blocking_operations: int = 0
    background_tasks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_request_time: Optional[datetime] = None
    first_request_time: Optional[datetime] = None

# =============================================================================
# Async Route Middleware
# =============================================================================

class AsyncRouteMiddleware(BaseHTTPMiddleware):
    """Middleware to ensure routes are non-blocking."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.route_metrics: Dict[str, RouteMetrics] = {}
        self.blocking_operations: List[str] = []
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()
        self._rate_limit_semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with async middleware."""
        start_time = time.time()
        route_key = f"{request.method}:{request.url.path}"
        
        # Initialize route metrics
        if route_key not in self.route_metrics:
            self.route_metrics[route_key] = RouteMetrics(
                route_path=request.url.path,
                method=request.method,
                first_request_time=datetime.now(timezone.utc)
            )
        
        metrics = self.route_metrics[route_key]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.now(timezone.utc)
        
        try:
            # Apply rate limiting
            if self.config.enable_rate_limiting:
                await self._apply_rate_limiting(request)
            
            # Check for blocking operations
            if self.config.blocking_detection != BlockingDetectionLevel.NONE:
                blocking_ops = self._detect_blocking_operations(request)
                if blocking_ops:
                    metrics.blocking_operations += 1
                    logger.warning(f"Route {route_key} contains blocking operations: {blocking_ops}")
            
            # Process request with timeout
            async with asyncio.timeout(self.config.timeout):
                response = await call_next(request)
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics.successful_requests += 1
            metrics.total_duration_ms += duration_ms
            metrics.avg_duration_ms = metrics.total_duration_ms / metrics.successful_requests
            metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
            metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
            
            return response
            
        except asyncio.TimeoutError:
            # Handle timeout
            duration_ms = (time.time() - start_time) * 1000
            metrics.failed_requests += 1
            
            logger.error(f"Request timeout for {route_key} after {duration_ms:.2f}ms")
            return JSONResponse(
                status_code=408,
                content={"error": "Request timeout", "duration_ms": duration_ms}
            )
            
        except Exception as e:
            # Handle other errors
            duration_ms = (time.time() - start_time) * 1000
            metrics.failed_requests += 1
            
            logger.error(f"Request error for {route_key}: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "duration_ms": duration_ms}
            )
    
    async def _apply_rate_limiting(self, request: Request):
        """Apply rate limiting to request."""
        await self._rate_limit_semaphore.acquire()
        try:
            # Simple rate limiting - in practice, use Redis or similar
            await asyncio.sleep(1.0 / self.config.rate_limit)
        finally:
            self._rate_limit_semaphore.release()
    
    def _detect_blocking_operations(self, request: Request) -> List[str]:
        """Detect potentially blocking operations in request."""
        blocking_operations = []
        
        # Check request headers for blocking indicators
        user_agent = request.headers.get("user-agent", "")
        if "sync" in user_agent.lower() or "blocking" in user_agent.lower():
            blocking_operations.append("blocking_user_agent")
        
        # Check request body size (large bodies might indicate blocking operations)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            blocking_operations.append("large_request_body")
        
        return blocking_operations
    
    def get_route_metrics(self) -> Dict[str, RouteMetrics]:
        """Get route performance metrics."""
        return self.route_metrics.copy()
    
    def get_blocking_operations(self) -> List[str]:
        """Get list of detected blocking operations."""
        return self.blocking_operations.copy()

# =============================================================================
# Non-Blocking Middleware
# =============================================================================

class NonBlockingMiddleware(BaseHTTPMiddleware):
    """Middleware to ensure non-blocking operations."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.background_workers)
        self.blocking_operations: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with non-blocking middleware."""
        # Check if request contains blocking operations
        blocking_ops = self._identify_blocking_operations(request)
        
        if blocking_ops:
            # Move blocking operations to background
            return await self._handle_blocking_operations(request, blocking_ops)
        else:
            # Process normally
            return await call_next(request)
    
    def _identify_blocking_operations(self, request: Request) -> List[str]:
        """Identify blocking operations in request."""
        blocking_ops = []
        
        # Check request path for blocking indicators
        path = request.url.path.lower()
        if any(keyword in path for keyword in ["sync", "blocking", "heavy", "process"]):
            blocking_ops.append("blocking_path")
        
        # Check request method
        if request.method in ["POST", "PUT", "PATCH"]:
            # Check content type for potential blocking operations
            content_type = request.headers.get("content-type", "")
            if "multipart" in content_type or "file" in content_type:
                blocking_ops.append("file_upload")
        
        return blocking_ops
    
    async def _handle_blocking_operations(
        self,
        request: Request,
        blocking_ops: List[str]
    ) -> Response:
        """Handle blocking operations by moving them to background."""
        # Create background task
        task_id = f"task_{int(time.time() * 1000)}"
        
        # Submit to background
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.thread_pool,
            self._execute_blocking_operation,
            request,
            blocking_ops
        )
        
        # Store task
        with self._lock:
            self.blocking_operations[task_id] = {
                "future": future,
                "created_at": datetime.now(timezone.utc),
                "blocking_ops": blocking_ops
            }
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Request accepted for background processing",
                "task_id": task_id,
                "status": "processing"
            }
        )
    
    def _execute_blocking_operation(self, request: Request, blocking_ops: List[str]):
        """Execute blocking operation in thread pool."""
        # Simulate blocking operation
        time.sleep(5)
        
        # Process the request
        # In practice, this would handle the actual blocking operation
        
        return {"status": "completed", "result": "success"}
    
    def get_background_tasks(self) -> Dict[str, Any]:
        """Get background task status."""
        with self._lock:
            return {
                task_id: {
                    "created_at": task_data["created_at"].isoformat(),
                    "blocking_ops": task_data["blocking_ops"],
                    "status": "running" if not task_data["future"].done() else "completed"
                }
                for task_id, task_data in self.blocking_operations.items()
            }

# =============================================================================
# Background Processing Middleware
# =============================================================================

class BackgroundProcessingMiddleware(BaseHTTPMiddleware):
    """Middleware for background processing of heavy operations."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.background_queue = asyncio.Queue(maxsize=1000)
        self.workers: List[asyncio.Task] = []
        self.task_handlers: Dict[str, Callable] = {}
        self._is_running = False
        self._setup_workers()
    
    def _setup_workers(self) -> Any:
        """Setup background workers."""
        for i in range(self.config.background_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        self._is_running = True
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a task handler."""
        self.task_handlers[task_type] = handler
    
    async def dispatch(self, request: Request, call_next):
        """Process request with background processing middleware."""
        # Check if request should be processed in background
        if self._should_process_in_background(request):
            return await self._process_in_background(request)
        else:
            return await call_next(request)
    
    def _should_process_in_background(self, request: Request) -> bool:
        """Determine if request should be processed in background."""
        # Check request headers
        background_header = request.headers.get("x-background-processing")
        if background_header and background_header.lower() == "true":
            return True
        
        # Check request path
        path = request.url.path.lower()
        if any(keyword in path for keyword in ["background", "async", "queue"]):
            return True
        
        # Check request method and content
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if "multipart" in content_type:
                return True
        
        return False
    
    async def _process_in_background(self, request: Request) -> Response:
        """Process request in background."""
        task_id = f"bg_task_{int(time.time() * 1000)}"
        
        # Create background task
        task_data = {
            "task_id": task_id,
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "created_at": datetime.now(timezone.utc)
        }
        
        # Add to background queue
        await self.background_queue.put(task_data)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Request queued for background processing",
                "task_id": task_id,
                "status": "queued"
            }
        )
    
    async def _worker(self, worker_name: str):
        """Background worker task."""
        while self._is_running:
            try:
                # Get task from queue
                task_data = await asyncio.wait_for(self.background_queue.get(), timeout=1.0)
                
                # Process task
                task_type = task_data.get("path", "").split("/")[-1]
                if task_type in self.task_handlers:
                    await self.task_handlers[task_type](task_data)
                
                # Mark task as done
                self.background_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background worker {worker_name} error: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get background queue statistics."""
        return {
            "queue_size": self.background_queue.qsize(),
            "worker_count": len(self.workers),
            "is_running": self._is_running
        }

# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.request_counts: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Process request
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use IP address as client ID
        client_ip = request.client.host if request.client else "unknown"
        
        # Add user ID if available
        user_id = request.headers.get("x-user-id")
        if user_id:
            return f"{client_ip}:{user_id}"
        
        return client_ip
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        with self._lock:
            # Clean old requests
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if timestamp > window_start
            ]
            
            # Check if limit exceeded
            if len(self.request_counts[client_id]) >= self.config.rate_limit:
                return False
            
            # Add current request
            self.request_counts[client_id].append(now)
            return True
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self._lock:
            return {
                "total_clients": len(self.request_counts),
                "rate_limit": self.config.rate_limit,
                "client_counts": {
                    client_id: len(timestamps)
                    for client_id, timestamps in self.request_counts.items()
                }
            }

# =============================================================================
# Circuit Breaker Middleware
# =============================================================================

class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware for circuit breaker pattern."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "state": "closed",  # closed, open, half-open
            "failure_count": 0,
            "last_failure_time": None,
            "success_count": 0,
            "threshold": 5,
            "timeout": 60
        })
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with circuit breaker."""
        # Get circuit breaker key
        circuit_key = f"{request.method}:{request.url.path}"
        
        # Check circuit breaker state
        if not self._should_allow_request(circuit_key):
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "circuit_state": self.circuit_breakers[circuit_key]["state"]
                }
            )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record success
            self._record_success(circuit_key)
            
            return response
            
        except Exception as e:
            # Record failure
            self._record_failure(circuit_key)
            raise
    
    async def _should_allow_request(self, circuit_key: str) -> bool:
        """Check if request should be allowed based on circuit breaker state."""
        with self._lock:
            circuit = self.circuit_breakers[circuit_key]
            
            if circuit["state"] == "closed":
                return True
            
            elif circuit["state"] == "open":
                # Check if timeout has passed
                if (circuit["last_failure_time"] and 
                    time.time() - circuit["last_failure_time"] > circuit["timeout"]):
                    circuit["state"] = "half-open"
                    return True
                return False
            
            elif circuit["state"] == "half-open":
                return True
            
            return True
    
    def _record_success(self, circuit_key: str):
        """Record successful request."""
        with self._lock:
            circuit = self.circuit_breakers[circuit_key]
            circuit["success_count"] += 1
            
            if circuit["state"] == "half-open":
                circuit["state"] = "closed"
                circuit["failure_count"] = 0
    
    def _record_failure(self, circuit_key: str):
        """Record failed request."""
        with self._lock:
            circuit = self.circuit_breakers[circuit_key]
            circuit["failure_count"] += 1
            circuit["last_failure_time"] = time.time()
            
            if circuit["failure_count"] >= circuit["threshold"]:
                circuit["state"] = "open"
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "circuit_breakers": {
                    key: {
                        "state": circuit["state"],
                        "failure_count": circuit["failure_count"],
                        "success_count": circuit["success_count"],
                        "threshold": circuit["threshold"]
                    }
                    for key, circuit in self.circuit_breakers.items()
                }
            }

# =============================================================================
# Caching Middleware
# =============================================================================

class CachingMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with caching."""
        # Check if request is cacheable
        if not self._is_cacheable(request):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache response
        self._cache_response(cache_key, response)
        
        return response
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable."""
        # Only cache GET requests
        if request.method != "GET":
            return False
        
        # Check cache control headers
        cache_control = request.headers.get("cache-control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        return True
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items())),
            request.headers.get("authorization", "")
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response."""
        with self._lock:
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.config.cache_ttl:
                    return cached_data["response"]
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Response):
        """Cache response."""
        with self._lock:
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        with self._lock:
            return {
                "cache_size": len(self.cache),
                "cache_ttl": self.config.cache_ttl
            }

# =============================================================================
# Monitoring Middleware
# =============================================================================

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring."""
    
    def __init__(self, app, config: MiddlewareConfig):
        
    """__init__ function."""
super().__init__(app)
        self.config = config
        self.request_metrics: Dict[str, RouteMetrics] = {}
        self._lock = threading.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with monitoring."""
        start_time = time.time()
        route_key = f"{request.method}:{request.url.path}"
        
        # Initialize metrics
        if route_key not in self.request_metrics:
            self.request_metrics[route_key] = RouteMetrics(
                route_path=request.url.path,
                method=request.method,
                first_request_time=datetime.now(timezone.utc)
            )
        
        metrics = self.request_metrics[route_key]
        metrics.total_requests += 1
        metrics.last_request_time = datetime.now(timezone.utc)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics.successful_requests += 1
            metrics.total_duration_ms += duration_ms
            metrics.avg_duration_ms = metrics.total_duration_ms / metrics.successful_requests
            metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
            metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
            
            return response
            
        except Exception as e:
            # Update error metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics.failed_requests += 1
            
            logger.error(f"Request error for {route_key}: {e}")
            raise
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            return {
                "total_routes": len(self.request_metrics),
                "route_metrics": {
                    key: asdict(metrics)
                    for key, metrics in self.request_metrics.items()
                }
            }

# =============================================================================
# Middleware Manager
# =============================================================================

class MiddlewareManager:
    """Manager for all async middlewares."""
    
    def __init__(self, config: MiddlewareConfig):
        
    """__init__ function."""
self.config = config
        self.middlewares: List[BaseHTTPMiddleware] = []
        self._is_initialized = False
    
    def add_middleware(self, middleware: BaseHTTPMiddleware):
        """Add middleware to manager."""
        self.middlewares.append(middleware)
    
    def setup_default_middlewares(self, app) -> Any:
        """Setup default middlewares."""
        # Add middlewares in order
        app.add_middleware(MonitoringMiddleware, config=self.config)
        app.add_middleware(CachingMiddleware, config=self.config)
        app.add_middleware(RateLimitingMiddleware, config=self.config)
        app.add_middleware(CircuitBreakerMiddleware, config=self.config)
        app.add_middleware(BackgroundProcessingMiddleware, config=self.config)
        app.add_middleware(NonBlockingMiddleware, config=self.config)
        app.add_middleware(AsyncRouteMiddleware, config=self.config)
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get statistics from all middlewares."""
        stats = {}
        
        for middleware in self.middlewares:
            if hasattr(middleware, 'get_route_metrics'):
                stats['async_route'] = middleware.get_route_metrics()
            if hasattr(middleware, 'get_background_tasks'):
                stats['background_tasks'] = middleware.get_background_tasks()
            if hasattr(middleware, 'get_rate_limit_stats'):
                stats['rate_limiting'] = middleware.get_rate_limit_stats()
            if hasattr(middleware, 'get_circuit_breaker_stats'):
                stats['circuit_breaker'] = middleware.get_circuit_breaker_stats()
            if hasattr(middleware, 'get_cache_stats'):
                stats['caching'] = middleware.get_cache_stats()
            if hasattr(middleware, 'get_monitoring_stats'):
                stats['monitoring'] = middleware.get_monitoring_stats()
        
        return stats

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "MiddlewareType",
    "BlockingDetectionLevel",
    "MiddlewareConfig",
    "RouteMetrics",
    "AsyncRouteMiddleware",
    "NonBlockingMiddleware",
    "BackgroundProcessingMiddleware",
    "RateLimitingMiddleware",
    "CircuitBreakerMiddleware",
    "CachingMiddleware",
    "MonitoringMiddleware",
    "MiddlewareManager",
] 