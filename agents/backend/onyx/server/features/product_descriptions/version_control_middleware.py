from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import logging
import json
import traceback
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
from contextvars import ContextVar
import uuid
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp
import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Version Control Middleware
Product Descriptions Feature - FastAPI Middleware for Logging, Monitoring, and Performance
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
start_time_var: ContextVar[Optional[float]] = ContextVar('start_time', default=None)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging"""
    
    def __init__(self, app: ASGIApp, log_requests: bool = True, log_responses: bool = True):
        
    """__init__ function."""
super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        start_time_var.set(time.time())
        
        # Log request details
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response details
            if self.log_responses:
                await self._log_response(request, response, request_id)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error details
            await self._log_error(request, e, request_id)
            raise
    
    async async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request details"""
        try:
            # Get request body if available
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        body = body.decode('utf-8')[:1000]  # Limit body size
                except Exception:
                    body = "[Unable to read body]"
            
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "body": body
            }
            
            logger.info(f"Request: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _log_response(self, request: Request, response: Response, request_id: str) -> None:
        """Log response details"""
        try:
            start_time = start_time_var.get()
            duration = time.time() - start_time if start_time else 0
            
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_headers": dict(response.headers),
                "content_length": response.headers.get("content-length")
            }
            
            logger.info(f"Response: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, request_id: str) -> None:
        """Log error details"""
        try:
            start_time = start_time_var.get()
            duration = time.time() - start_time if start_time else 0
            
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "duration_ms": round(duration * 1000, 2)
            }
            
            logger.error(f"Error: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and optimization"""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        
    """__init__ function."""
super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.request_times: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Track performance metrics
            await self._track_performance(request, duration, response.status_code)
            
            # Check for slow requests
            if duration > self.slow_request_threshold:
                await self._log_slow_request(request, duration)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            await self._track_performance(request, duration, 500)
            raise
    
    async def _track_performance(self, request: Request, duration: float, status_code: int) -> None:
        """Track performance metrics"""
        try:
            path = request.url.path
            method = request.method
            
            # Initialize path tracking if not exists
            if path not in self.request_times:
                self.request_times[path] = []
            
            # Add duration to tracking
            self.request_times[path].append({
                "duration": duration,
                "method": method,
                "status_code": status_code,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 100 requests per path
            if len(self.request_times[path]) > 100:
                self.request_times[path] = self.request_times[path][-100:]
                
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    async async def _log_slow_request(self, request: Request, duration: float) -> None:
        """Log slow request details"""
        try:
            log_data = {
                "request_id": request_id_var.get(),
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "duration": duration,
                "threshold": self.slow_request_threshold,
                "client_ip": request.client.host if request.client else None
            }
            
            logger.warning(f"Slow Request: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging slow request: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = {}
            for path, times in self.request_times.items():
                if not times:
                    continue
                
                durations = [t["duration"] for t in times]
                status_codes = [t["status_code"] for t in times]
                
                stats[path] = {
                    "request_count": len(times),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "success_rate": len([s for s in status_codes if s < 400]) / len(status_codes),
                    "last_request": times[-1]["timestamp"] if times else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling and monitoring"""
    
    def __init__(self, app: ASGIApp, log_errors: bool = True, notify_errors: bool = False):
        
    """__init__ function."""
super().__init__(app)
        self.log_errors = log_errors
        self.notify_errors = notify_errors
        self.error_counts: Dict[str, int] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # Track error
            await self._track_error(request, e)
            
            # Log error if enabled
            if self.log_errors:
                await self._log_error(request, e)
            
            # Notify error if enabled
            if self.notify_errors:
                await self._notify_error(request, e)
            
            # Return error response
            return await self._create_error_response(request, e)
    
    async def _track_error(self, request: Request, error: Exception) -> None:
        """Track error statistics"""
        try:
            error_type = type(error).__name__
            path = request.url.path
            
            key = f"{path}:{error_type}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    async def _log_error(self, request: Request, error: Exception) -> None:
        """Log error details"""
        try:
            log_data = {
                "request_id": request_id_var.get(),
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
            
            logger.error(f"Application Error: {json.dumps(log_data, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    async def _notify_error(self, request: Request, error: Exception) -> None:
        """Notify about critical errors (placeholder for external notification)"""
        try:
            # This is a placeholder for external error notification
            # Could integrate with services like Sentry, LogRocket, etc.
            error_data = {
                "request_id": request_id_var.get(),
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "client_ip": request.client.host if request.client else None
            }
            
            # Example: Send to external monitoring service
            # await send_to_monitoring_service(error_data)
            
            logger.info(f"Error notification sent: {error_data}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _create_error_response(self, request: Request, error: Exception) -> JSONResponse:
        """Create standardized error response"""
        try:
            error_data = {
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id_var.get(),
                "timestamp": datetime.now().isoformat(),
                "path": request.url.path
            }
            
            # Include error details in development
            if request.headers.get("x-debug-mode") == "true":
                error_data.update({
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc()
                })
            
            return JSONResponse(
                status_code=500,
                content=error_data,
                headers={"X-Request-ID": request_id_var.get() or ""}
            )
            
        except Exception as e:
            logger.error(f"Error creating error response: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
                headers={"X-Request-ID": request_id_var.get() or ""}
            )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        try:
            stats = {
                "total_errors": sum(self.error_counts.values()),
                "error_breakdown": self.error_counts.copy(),
                "most_common_errors": sorted(
                    self.error_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting error stats: {e}")
            return {}

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and basic protection"""
    
    def __init__(self, app: ASGIApp):
        
    """__init__ function."""
super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        
    """__init__ function."""
super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Initialize client tracking
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove old requests (older than 1 minute)
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "retry_after": 60
                }
            )
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.request_counts[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response

# Middleware factory function
def create_middleware_stack(app: ASGIApp) -> ASGIApp:
    """Create and configure middleware stack"""
    
    # Add middleware in order (last added = first executed)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RateLimitingMiddleware, requests_per_minute=100)
    app.add_middleware(ErrorHandlingMiddleware, log_errors=True, notify_errors=False)
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=1.0)
    app.add_middleware(RequestLoggingMiddleware, log_requests=True, log_responses=True)
    
    return app

# Utility functions for accessing middleware data
async def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_var.get()

async def get_request_duration() -> Optional[float]:
    """Get current request duration"""
    start_time = start_time_var.get()
    if start_time:
        return time.time() - start_time
    return None 