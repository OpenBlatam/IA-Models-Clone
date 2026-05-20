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
import uuid
import logging
import asyncio
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from ..utils.error_handling import ErrorTracker, ErrorType, ErrorSeverity
from ..utils.performance_monitor import PerformanceMonitor
from typing import Any, List, Dict, Optional
"""
Performance Middleware for Instagram Captions API v14.0

Advanced middleware for:
- Request/response logging with structured data
- Performance monitoring and metrics collection
- Error tracking and monitoring
- Security validation and threat detection
- Response optimization and caching
"""




logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Advanced request/response logging middleware with structured data."""
    
    def __init__(self, app, enable_detailed_logging: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.error_tracker = ErrorTracker()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract request info
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        api_key = request.headers.get("authorization", "").replace("Bearer ", "")
        
        # Log request start with structured data
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "api_key_present": bool(api_key),
            "content_length": request.headers.get("content-length"),
            "timestamp": timestamp.isoformat(),
            "event": "request_started"
        }
        
        if self.enable_detailed_logging:
            logger.info("Request started", extra=log_data)
        else:
            logger.info(f"🚀 {request.method} {request.url.path} - {request_id}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            
            # Log successful response
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "duration_seconds": round(duration, 3),
                "response_size": len(response.body) if hasattr(response, 'body') else 0,
                "event": "request_completed"
            }
            
            if self.enable_detailed_logging:
                logger.info("Request completed", extra=response_data)
            else:
                logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {duration*1000:.1f}ms")
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration*1000:.3f}ms"
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            
            # Record error
            self.error_tracker.record_error(
                error_type=ErrorType.SYSTEM,
                message=f"Request failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                details={
                    "method": request.method,
                    "url": str(request.url),
                    "duration": duration,
                    "client_ip": client_ip
                },
                request_id=request_id
            )
            
            # Log error with structured data
            error_data = {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": round(duration * 1000, 2),
                "method": request.method,
                "url": str(request.url),
                "event": "request_failed"
            }
            
            if self.enable_detailed_logging:
                logger.error("Request failed", extra=error_data, exc_info=True)
            else:
                logger.error(f"❌ {request.method} {request.url.path} - ERROR: {str(e)}")
            
            # Return error response
            error_response = {
                "error": True,
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": timestamp.isoformat()
            }
            
            return JSONResponse(
                status_code=500,
                content=error_response,
                headers={"X-Request-ID": request_id}
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Advanced performance monitoring middleware with metrics collection."""
    
    def __init__(self, app, slow_request_threshold: float = 1.0, enable_metrics: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.enable_metrics = enable_metrics
        self.performance_monitor = PerformanceMonitor()
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "slow_requests": 0,
            "average_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0,
            "endpoint_metrics": {},
            "response_times": []
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        endpoint = f"{request.method} {request.url.path}"
        
        # Initialize endpoint metrics if not exists
        if endpoint not in self.metrics["endpoint_metrics"]:
            self.metrics["endpoint_metrics"][endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "error_count": 0,
                "slow_count": 0,
                "response_times": []
            }
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.perf_counter() - start_time
            self._update_metrics(endpoint, duration, False)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Performance-Tier"] = self._get_performance_tier(duration)
            response.headers["X-Endpoint"] = endpoint
            
            # Log slow requests
            if duration > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected: {endpoint} took {duration*1000:.1f}ms "
                    f"(threshold: {self.slow_request_threshold*1000:.0f}ms)"
                )
            
            # Check performance thresholds
            performance_error = self.performance_monitor.check_performance("response_time", duration)
            if performance_error:
                logger.warning(f"Performance threshold exceeded: {performance_error}")
            
            return response
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self._update_metrics(endpoint, duration, True)
            raise
    
    def _update_metrics(self, endpoint: str, duration: float, is_error: bool):
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["response_times"].append(duration)
        
        endpoint_stats = self.metrics["endpoint_metrics"][endpoint]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += duration
        endpoint_stats["response_times"].append(duration)
        
        if is_error:
            self.metrics["total_errors"] += 1
            endpoint_stats["error_count"] += 1
        
        if duration > self.slow_request_threshold:
            self.metrics["slow_requests"] += 1
            endpoint_stats["slow_count"] += 1
        
        # Update average response time
        total_time = sum(
            stats["total_time"] 
            for stats in self.metrics["endpoint_metrics"].values()
        )
        self.metrics["average_response_time"] = total_time / self.metrics["total_requests"]
        
        # Calculate percentiles
        if len(self.metrics["response_times"]) > 0:
            sorted_times = sorted(self.metrics["response_times"])
            self.metrics["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
            self.metrics["p99_response_time"] = sorted_times[int(len(sorted_times) * 0.99)]
    
    def _get_performance_tier(self, duration: float) -> str:
        """Get performance tier based on response time."""
        if duration < 0.050:  # 50ms
            return "excellent"
        elif duration < 0.100:  # 100ms
            return "good"
        elif duration < 0.250:  # 250ms
            return "acceptable"
        elif duration < 0.500:  # 500ms
            return "slow"
        elif duration < 1.000:  # 1s
            return "very-slow"
        else:
            return "critical"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def get_endpoint_metrics(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific endpoint."""
        return self.metrics["endpoint_metrics"].get(endpoint)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with threat detection and validation."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
        self.error_tracker = ErrorTracker()
        self.threat_patterns = [
            r"<script", r"javascript:", r"data:", r"vbscript:",
            r"onload=", r"onerror=", r"onclick=",
            r"../", r"..\\", r"~", r"/etc/", r"/proc/",
            r"union.*select", r"drop.*table", r"insert.*into",
            r"exec\(", r"system\(", r"eval\("
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            self.error_tracker.record_error(
                error_type=ErrorType.SECURITY,
                message="Request size exceeds limit",
                severity=ErrorSeverity.MEDIUM,
                details={"content_length": content_length, "max_size": self.max_request_size},
                request_id=request_id
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request size exceeds {self.max_request_size // (1024*1024)}MB limit"
                }
            )
        
        # Security scan for malicious content
        try:
            body = await request.body()
            if body:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.threat_patterns:
                    if pattern.lower() in body_str.lower():
                        self.error_tracker.record_error(
                            error_type=ErrorType.SECURITY,
                            message=f"Malicious content detected: {pattern}",
                            severity=ErrorSeverity.HIGH,
                            details={"pattern": pattern, "client_ip": request.client.host},
                            request_id=request_id
                        )
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": True,
                                "error_code": "MALICIOUS_CONTENT",
                                "message": "Malicious content detected in request"
                            }
                        )
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling middleware with structured responses."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.error_tracker = ErrorTracker()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "unknown")
        
        try:
            return await call_next(request)
        except HTTPException as e:
            # HTTP exceptions are handled by FastAPI
            self.error_tracker.record_error(
                error_type=ErrorType.VALIDATION,
                message=str(e.detail),
                severity=ErrorSeverity.MEDIUM,
                details={"status_code": e.status_code},
                request_id=request_id
            )
            raise
        except ValueError as e:
            # Validation errors
            self.error_tracker.record_error(
                error_type=ErrorType.VALIDATION,
                message=str(e),
                severity=ErrorSeverity.MEDIUM,
                details={"exception_type": "ValueError"},
                request_id=request_id
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            # Unexpected errors
            self.error_tracker.record_error(
                error_type=ErrorType.SYSTEM,
                message=str(e),
                severity=ErrorSeverity.HIGH,
                details={
                    "exception_type": type(e).__name__,
                    "method": request.method,
                    "url": str(request.url)
                },
                request_id=request_id
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware for performance optimization."""
    
    def __init__(self, app, min_size: int = 1000, enable_gzip: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.min_size = min_size
        self.enable_gzip = enable_gzip
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if response should be compressed
        if self.enable_gzip and self._should_compress(response):
            # Add compression headers
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_compress(self, response: Response) -> bool:
        """Determine if response should be compressed."""
        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "application/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return False
        
        # Check response size
        if hasattr(response, 'body'):
            return len(response.body) >= self.min_size
        
        return False


class CacheMiddleware(BaseHTTPMiddleware):
    """Cache control middleware for response optimization."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.cache_settings = {
            "/health": {"max_age": 30},  # 30 seconds
            "/metrics": {"max_age": 60},  # 1 minute
            "/performance/status": {"max_age": 120},  # 2 minutes
            "/api/v14/info": {"max_age": 300},  # 5 minutes
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add cache headers based on endpoint
        path = request.url.path
        cache_config = self.cache_settings.get(path, {"max_age": 0})
        
        if cache_config["max_age"] > 0:
            response.headers["Cache-Control"] = f"public, max-age={cache_config['max_age']}"
        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


# Middleware factory functions
def create_middleware_stack(app, config: Dict[str, Any] = None):
    """Create and configure the complete middleware stack."""
    if config is None:
        config = {
            "enable_detailed_logging": True,
            "slow_request_threshold": 1.0,
            "enable_metrics": True,
            "max_request_size": 10 * 1024 * 1024,
            "enable_gzip": True,
            "min_compression_size": 1000
        }
    
    # Add middleware in reverse order (last added = first executed)
    
    # Error handling middleware (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security middleware
    app.add_middleware(SecurityMiddleware, max_request_size=config["max_request_size"])
    
    # Performance monitoring middleware
    if config["enable_metrics"]:
        app.add_middleware(
            PerformanceMonitoringMiddleware,
            slow_request_threshold=config["slow_request_threshold"],
            enable_metrics=config["enable_metrics"]
        )
    
    # Request logging middleware
    app.add_middleware(
        RequestLoggingMiddleware,
        enable_detailed_logging=config["enable_detailed_logging"]
    )
    
    # Compression middleware
    if config["enable_gzip"]:
        app.add_middleware(
            CompressionMiddleware,
            min_size=config["min_compression_size"],
            enable_gzip=config["enable_gzip"]
        )
    
    # Cache middleware
    app.add_middleware(CacheMiddleware)
    
    return app


# Context manager for middleware performance tracking
@asynccontextmanager
async def middleware_performance_context(operation: str, request_id: str):
    """Context manager for tracking middleware performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        if duration > 0.1:  # Log slow middleware operations
            logger.warning(f"Slow middleware operation: {operation} took {duration*1000:.1f}ms") 