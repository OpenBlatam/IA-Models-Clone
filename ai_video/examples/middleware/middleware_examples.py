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

import time
import logging
import json
import uuid
import asyncio
import psutil
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import redis.asyncio as redis
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🔧 MIDDLEWARE EXAMPLES - PRACTICAL IMPLEMENTATIONS
==================================================

Real-world examples of middleware for logging, error monitoring, and performance optimization.
"""



logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE 1: AI VIDEO API MIDDLEWARE
# ============================================================================

class AIVideoLoggingMiddleware(BaseHTTPMiddleware):
    """Specialized logging middleware for AI Video API."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Increment request counter
        self.request_count += 1
        
        # Log request
        await self.log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - request.state.start_time
            
            # Log response
            await self.log_response(response, request_id, duration)
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-Count"] = str(self.request_count)
            
            return response
            
        except Exception as e:
            # Increment error counter
            self.error_count += 1
            
            # Calculate duration
            duration = time.time() - request.state.start_time
            
            # Log error
            await self.log_error(request, request_id, e, duration)
            
            # Create error response
            return await self.create_error_response(e, request_id, duration)
    
    async def log_request(self, request: Request, request_id: str):
        """Log AI Video API request."""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": self.get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "timestamp": datetime.now().isoformat(),
            "service": "ai_video_api"
        }
        
        logger.info(f"🎬 AI Video Request: {json.dumps(log_data, indent=2)}")
    
    async def log_response(self, response: Response, request_id: str, duration: float):
        """Log AI Video API response."""
        log_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "duration": duration,
            "content_type": response.headers.get("content-type"),
            "timestamp": datetime.now().isoformat(),
            "service": "ai_video_api"
        }
        
        logger.info(f"✅ AI Video Response: {json.dumps(log_data, indent=2)}")
    
    async def log_error(self, request: Request, request_id: str, error: Exception, duration: float):
        """Log AI Video API error."""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "service": "ai_video_api"
        }
        
        logger.error(f"❌ AI Video Error: {json.dumps(log_data, indent=2)}")
    
    async def create_error_response(self, error: Exception, request_id: str, duration: float) -> Response:
        """Create standardized error response for AI Video API."""
        error_data = {
            "error": {
                "id": request_id,
                "type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "service": "ai_video_api"
            }
        }
        
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(error, HTTPException):
            status_code = error.status_code
        
        return JSONResponse(
            status_code=status_code,
            content=error_data,
            headers={
                "X-Request-ID": request_id,
                "X-Processing-Time": f"{duration:.3f}s",
                "X-Error-Count": str(self.error_count)
            }
        )
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# ============================================================================
# EXAMPLE 2: PERFORMANCE MONITORING MIDDLEWARE
# ============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware for AI Video processing."""
    
    def __init__(self, app, 
                 slow_threshold: float = 2.0,
                 memory_threshold: int = 100 * 1024 * 1024):  # 100MB
        super().__init__(app)
        self.slow_threshold = slow_threshold
        self.memory_threshold = memory_threshold
        self.performance_metrics = []
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record initial state
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.cpu_percent()
            
            memory_delta = end_memory - start_memory
            cpu_delta = end_cpu - start_cpu
            
            # Track performance
            await self.track_performance(
                request, response, duration, memory_delta, cpu_delta
            )
            
            # Add performance headers
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            response.headers["X-Memory-Delta"] = f"{memory_delta / 1024 / 1024:.2f}MB"
            response.headers["X-CPU-Delta"] = f"{cpu_delta:.1f}%"
            
            return response
            
        except Exception as e:
            # Calculate metrics for errors
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            # Track error performance
            await self.track_performance(
                request, None, duration, memory_delta, 0, error=True
            )
            
            raise
    
    async def track_performance(self, request: Request, response: Optional[Response],
                              duration: float, memory_delta: int, cpu_delta: float,
                              error: bool = False):
        """Track performance metrics."""
        metrics = {
            "endpoint": f"{request.method}:{request.url.path}",
            "duration": duration,
            "memory_delta_mb": memory_delta / 1024 / 1024,
            "cpu_delta": cpu_delta,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "slow_request": duration > self.slow_threshold,
            "high_memory": memory_delta > self.memory_threshold
        }
        
        if response:
            metrics["status_code"] = response.status_code
            metrics["content_length"] = response.headers.get("content-length")
        
        # Log slow requests
        if duration > self.slow_threshold:
            logger.warning(f"🐌 Slow AI Video request: {duration:.3f}s for {request.url.path}")
        
        # Log high memory usage
        if memory_delta > self.memory_threshold:
            logger.warning(f"💾 High memory usage: {memory_delta / 1024 / 1024:.2f}MB for {request.url.path}")
        
        # Store metrics
        self.performance_metrics.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
        
        logger.info(f"📊 Performance: {json.dumps(metrics, indent=2)}")

# ============================================================================
# EXAMPLE 3: ERROR MONITORING MIDDLEWARE
# ============================================================================

class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Error monitoring middleware for AI Video API."""
    
    def __init__(self, app, 
                 alert_threshold: int = 5,
                 alert_window: int = 300,  # 5 minutes
                 error_types_to_track: List[str] = None):
        
    """__init__ function."""
super().__init__(app)
        self.alert_threshold = alert_threshold
        self.alert_window = alert_window
        self.error_counts = {}
        self.error_types_to_track = error_types_to_track or [
            "ValidationError", "ModelLoadError", "VideoGenerationError", "TimeoutError"
        ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            
            # Track 4xx and 5xx responses
            if 400 <= response.status_code < 600:
                await self.track_error(response.status_code, request)
            
            return response
            
        except Exception as e:
            # Track exceptions
            await self.track_exception(e, request)
            
            # Check if we should alert
            await self.check_alert_threshold()
            
            # Return error response
            return await self.create_error_response(e, request)
    
    async def track_error(self, status_code: int, request: Request):
        """Track HTTP error responses."""
        error_key = f"http_error:{status_code}:{request.url.path}"
        current_time = time.time()
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = []
        
        self.error_counts[error_key].append(current_time)
        
        # Clean old entries
        cutoff_time = current_time - self.alert_window
        self.error_counts[error_key] = [
            t for t in self.error_counts[error_key] 
            if t > cutoff_time
        ]
        
        logger.warning(f"HTTP Error {status_code} on {request.url.path}")
    
    async def track_exception(self, exception: Exception, request: Request):
        """Track exceptions."""
        error_type = type(exception).__name__
        error_key = f"exception:{error_type}:{request.url.path}"
        current_time = time.time()
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = []
        
        self.error_counts[error_key].append(current_time)
        
        # Clean old entries
        cutoff_time = current_time - self.alert_window
        self.error_counts[error_key] = [
            t for t in self.error_counts[error_key] 
            if t > cutoff_time
        ]
        
        # Log detailed error for tracked types
        if error_type in self.error_types_to_track:
            logger.error(f"🚨 Critical Error {error_type} on {request.url.path}: {str(exception)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"Exception {error_type} on {request.url.path}: {str(exception)}")
    
    async def check_alert_threshold(self) -> Any:
        """Check if error threshold has been exceeded."""
        current_time = time.time()
        cutoff_time = current_time - self.alert_window
        
        for error_key, timestamps in self.error_counts.items():
            recent_errors = [t for t in timestamps if t > cutoff_time]
            
            if len(recent_errors) >= self.alert_threshold:
                logger.critical(f"🚨 ALERT: {len(recent_errors)} errors for {error_key} in {self.alert_window}s")
                # Here you would send alerts (email, Slack, etc.)
                await self.send_alert(error_key, len(recent_errors))
    
    async def send_alert(self, error_key: str, error_count: int):
        """Send alert for error threshold exceeded."""
        alert_data = {
            "error_key": error_key,
            "error_count": error_count,
            "threshold": self.alert_threshold,
            "window": self.alert_window,
            "timestamp": datetime.now().isoformat(),
            "service": "ai_video_api"
        }
        
        logger.critical(f"ALERT DATA: {json.dumps(alert_data, indent=2)}")
        # In production, send to alerting service (Slack, email, etc.)
    
    async def create_error_response(self, exception: Exception, request: Request) -> Response:
        """Create standardized error response."""
        error_id = str(uuid.uuid4())
        error_type = type(exception).__name__
        
        # Log detailed error
        logger.error(f"Error ID {error_id}: {str(exception)}")
        
        # Create error response
        error_data = {
            "error": {
                "id": error_id,
                "type": error_type,
                "message": str(exception),
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path,
                "method": request.method,
                "service": "ai_video_api"
            }
        }
        
        # Determine status code
        if isinstance(exception, HTTPException):
            status_code = exception.status_code
        elif error_type == "ValidationError":
            status_code = status.HTTP_400_BAD_REQUEST
        elif error_type == "ModelLoadError":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif error_type == "VideoGenerationError":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif error_type == "TimeoutError":
            status_code = status.HTTP_408_REQUEST_TIMEOUT
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        return JSONResponse(
            status_code=status_code,
            content=error_data,
            headers={
                "X-Error-ID": error_id,
                "X-Error-Type": error_type
            }
        )

# ============================================================================
# EXAMPLE 4: CACHE MIDDLEWARE FOR AI VIDEO
# ============================================================================

class AIVideoCacheMiddleware(BaseHTTPMiddleware):
    """Cache middleware specifically for AI Video API responses."""
    
    def __init__(self, app, 
                 redis_client: Optional[redis.Redis] = None,
                 default_ttl: int = 3600):
        
    """__init__ function."""
super().__init__(app)
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.cacheable_endpoints = {
            "/health": 60,  # 1 minute
            "/metrics": 300,  # 5 minutes
            "/api/v1/videos": 1800,  # 30 minutes
            "/api/v1/models": 3600,  # 1 hour
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if endpoint is cacheable
        cache_ttl = self.cacheable_endpoints.get(request.url.path)
        if not cache_ttl:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self.generate_cache_key(request)
        
        # Try to get cached response
        cached_response = await self.get_cached_response(cache_key)
        if cached_response:
            cached_response.headers["X-Cache"] = "HIT"
            cached_response.headers["X-Cache-TTL"] = str(cache_ttl)
            return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if 200 <= response.status_code < 300:
            await self.cache_response(cache_key, response, cache_ttl)
            response.headers["X-Cache"] = "MISS"
            response.headers["X-Cache-TTL"] = str(cache_ttl)
        
        return response
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key for AI Video API request."""
        # Include path and query parameters
        key_parts = [request.url.path]
        
        # Add query parameters
        if request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.append("&".join(f"{k}={v}" for k, v in sorted_params))
        
        # Add user agent for different client types
        user_agent = request.headers.get("user-agent", "")
        if "mobile" in user_agent.lower():
            key_parts.append("mobile")
        elif "desktop" in user_agent.lower():
            key_parts.append("desktop")
        
        return f"ai_video_cache:{':'.join(key_parts)}"
    
    async def get_cached_response(self, cache_key: str) -> Optional[Response]:
        """Get cached response from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return JSONResponse(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers=data["headers"]
                )
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_response(self, cache_key: str, response: Response, ttl: int):
        """Cache response in Redis."""
        if not self.redis_client:
            return
        
        try:
            # Get response content
            if hasattr(response, 'body'):
                content = response.body.decode()
            else:
                content = ""
            
            cache_data = {
                "content": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "cached_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data)
            )
            
            logger.info(f"Cached response for {cache_key} with TTL {ttl}s")
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")

# ============================================================================
# EXAMPLE 5: RATE LIMITING FOR AI VIDEO
# ============================================================================

class AIVideoRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware specifically for AI Video API."""
    
    def __init__(self, app, 
                 redis_client: Optional[redis.Redis] = None,
                 rate_limits: Dict[str, Dict[str, int]] = None):
        
    """__init__ function."""
super().__init__(app)
        self.redis_client = redis_client
        
        # Default rate limits
        self.rate_limits = rate_limits or {
            "/api/v1/videos/generate": {"requests": 10, "window": 3600},  # 10 per hour
            "/api/v1/videos/batch": {"requests": 5, "window": 3600},      # 5 per hour
            "/api/v1/models": {"requests": 100, "window": 3600},          # 100 per hour
            "default": {"requests": 1000, "window": 3600}                 # 1000 per hour
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get rate limit for endpoint
        endpoint = request.url.path
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Get client identifier
        client_id = self.get_client_id(request)
        
        # Check rate limit
        if not await self.check_rate_limit(client_id, endpoint, rate_limit):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "endpoint": endpoint,
                    "limit": rate_limit["requests"],
                    "window": rate_limit["window"],
                    "retry_after": rate_limit["window"]
                },
                headers={
                    "Retry-After": str(rate_limit["window"]),
                    "X-RateLimit-Limit": str(rate_limit["requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + rate_limit["window"]))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self.get_remaining_requests(client_id, endpoint, rate_limit)
        response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + rate_limit["window"]))
        
        return response
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, decode JWT to get user ID
            return f"user_{hash(auth_header)}"
        
        # Fall back to IP address
        return request.client.host if request.client else "unknown"
    
    async def check_rate_limit(self, client_id: str, endpoint: str, rate_limit: Dict[str, int]) -> bool:
        """Check if client has exceeded rate limit."""
        if not self.redis_client:
            return True  # Skip rate limiting if no Redis
        
        key = f"rate_limit:{client_id}:{endpoint}"
        
        try:
            # Get current count
            current = await self.redis_client.get(key)
            
            if current is None:
                # First request
                await self.redis_client.setex(key, rate_limit["window"], 1)
                return True
            
            current_count = int(current)
            if current_count >= rate_limit["requests"]:
                return False
            
            # Increment counter
            await self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request if rate limiting fails
    
    async async def get_remaining_requests(self, client_id: str, endpoint: str, rate_limit: Dict[str, int]) -> int:
        """Get remaining requests for client."""
        if not self.redis_client:
            return rate_limit["requests"]
        
        try:
            key = f"rate_limit:{client_id}:{endpoint}"
            current = await self.redis_client.get(key)
            current_count = int(current) if current else 0
            return max(0, rate_limit["requests"] - current_count)
        except Exception:
            return rate_limit["requests"]

# ============================================================================
# EXAMPLE 6: COMPREHENSIVE MIDDLEWARE STACK
# ============================================================================

def create_ai_video_middleware_stack(app: FastAPI, 
                                   redis_client: Optional[redis.Redis] = None) -> None:
    """Create comprehensive middleware stack for AI Video API."""
    
    # Add middleware in order (last added = first executed)
    
    # 1. Rate limiting (first)
    app.add_middleware(AIVideoRateLimitMiddleware, redis_client=redis_client)
    
    # 2. Caching
    app.add_middleware(AIVideoCacheMiddleware, redis_client=redis_client)
    
    # 3. Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware,
                      slow_threshold=2.0,
                      memory_threshold=100 * 1024 * 1024)
    
    # 4. Error monitoring
    app.add_middleware(ErrorMonitoringMiddleware,
                      alert_threshold=5,
                      alert_window=300)
    
    # 5. Request logging (last)
    app.add_middleware(AIVideoLoggingMiddleware)
    
    # 6. Standard FastAPI middleware
    app.add_middleware(CORSMiddleware,
                      allow_origins=["*"],
                      allow_credentials=True,
                      allow_methods=["*"],
                      allow_headers=["*"])
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================================================
# EXAMPLE 7: FASTAPI APPLICATION WITH MIDDLEWARE
# ============================================================================

def create_ai_video_app() -> FastAPI:
    """Create FastAPI application with comprehensive middleware."""
    
    app = FastAPI(
        title="AI Video Generation API",
        description="API for AI video generation with comprehensive middleware",
        version="1.0.0"
    )
    
    # Create Redis client
    try:
        redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None
    
    # Create middleware stack
    create_ai_video_middleware_stack(app, redis_client)
    
    # API endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ai_video_api",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics."""
        return {
            "service": "ai_video_api",
            "timestamp": datetime.now().isoformat(),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent()
        }
    
    @app.post("/api/v1/videos/generate")
    async def generate_video(request: Request):
        """Generate video endpoint."""
        # Simulate video generation
        await asyncio.sleep(1)
        
        return {
            "video_id": str(uuid.uuid4()),
            "status": "generated",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/api/v1/videos")
    async def list_videos():
        """List videos endpoint."""
        return {
            "videos": [
                {"id": "1", "title": "Video 1"},
                {"id": "2", "title": "Video 2"}
            ]
        }
    
    @app.get("/api/v1/models")
    async def list_models():
        """List models endpoint."""
        return {
            "models": [
                {"name": "stable-diffusion", "version": "1.5"},
                {"name": "text-to-video", "version": "2.0"}
            ]
        }
    
    return app

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_middleware():
    """Example of basic middleware usage."""
    
    app = FastAPI()
    
    # Add basic logging middleware
    app.add_middleware(AIVideoLoggingMiddleware)
    
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Hello AI Video API"}
    
    return app

def example_performance_middleware():
    """Example of performance monitoring middleware."""
    
    app = FastAPI()
    
    # Add performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware,
                      slow_threshold=1.0,
                      memory_threshold=50 * 1024 * 1024)
    
    @app.get("/slow")
    async def slow_endpoint():
        
    """slow_endpoint function."""
await asyncio.sleep(2)  # Simulate slow processing
        return {"message": "Slow response"}
    
    return app

def example_error_monitoring():
    """Example of error monitoring middleware."""
    
    app = FastAPI()
    
    # Add error monitoring
    app.add_middleware(ErrorMonitoringMiddleware,
                      alert_threshold=3,
                      alert_window=60)
    
    @app.get("/error")
    async def error_endpoint():
        
    """error_endpoint function."""
raise HTTPException(status_code=500, detail="Simulated error")
    
    return app

if __name__ == "__main__":
    # Example usage
    app = create_ai_video_app()
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 