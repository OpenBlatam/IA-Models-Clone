"""
Real-World Middleware for BUL API
================================

Production-ready middleware following FastAPI best practices:
- Functional programming approach
- Early returns and guard clauses
- Real-world error handling
- Performance monitoring
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import uuid

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send

# Real-world middleware
class RealMiddleware(BaseHTTPMiddleware):
    """Real-world middleware for request/response processing"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0
    
    async def dispatch(self, request: Request, call_next):
        """Process request with early returns and guard clauses"""
        # Early request processing
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            self.request_count += 1
            self.total_duration += duration
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(duration)
            response.headers["X-Request-Count"] = str(self.request_count)
            
            # Log slow requests
            if duration > 0.1:
                logging.warning(f"Slow request: {duration:.4f}s - {request.method} {request.url}")
            
            return response
            
        except Exception as e:
            # Handle errors with early returns
            duration = time.time() - start_time
            self.error_count += 1
            
            logging.error(f"Request error: {str(e)} - {request.method} {request.url}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

class RealCORSMiddleware:
    """Real-world CORS middleware"""
    
    def __init__(self, app: ASGIApp, allow_origins: List[str] = None):
        self.app = app
        self.allow_origins = allow_origins or ["*"]
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Handle CORS with early returns"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Handle preflight requests
        if scope["method"] == "OPTIONS":
            response = Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "86400"
                }
            )
            await response(scope, receive, send)
            return
        
        # Add CORS headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers.extend([
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                    (b"access-control-allow-headers", b"*")
                ])
                message["headers"] = list(headers.items())
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class RealSecurityMiddleware:
    """Real-world security middleware"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Add security headers with early returns"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Add security headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                security_headers = [
                    (b"x-frame-options", b"DENY"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin")
                ]
                headers.update(security_headers)
                message["headers"] = list(headers.items())
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class RealRateLimitMiddleware:
    """Real-world rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Handle rate limiting with early returns"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Get client IP
        client_ip = scope.get("client", ("", 0))[0]
        now = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if now - req_time < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            response = Response(
                status_code=429,
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }),
                headers={"Retry-After": "60"}
            )
            await response(scope, receive, send)
            return
        
        # Add request
        self.requests[client_ip].append(now)
        
        await self.app(scope, receive, send)

class RealLoggingMiddleware:
    """Real-world logging middleware"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Log requests with early returns"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Log request
        logging.info(json.dumps({
            "type": "request",
            "request_id": request_id,
            "method": scope["method"],
            "path": scope["path"],
            "query_string": scope.get("query_string", b"").decode(),
            "timestamp": datetime.now().isoformat()
        }))
        
        # Process request
        status_code = 200
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            status_code = 500
            logging.error(json.dumps({
                "type": "error",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }))
            raise
        
        # Log response
        duration = time.time() - start_time
        logging.info(json.dumps({
            "type": "response",
            "request_id": request_id,
            "status_code": status_code,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }))

class RealMetricsMiddleware:
    """Real-world metrics middleware"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
        self.metrics = {
            "request_count": 0,
            "error_count": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "max_duration": 0.0,
            "min_duration": float('inf')
        }
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Collect metrics with early returns"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        try:
            await self.app(scope, receive, send)
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics["request_count"] += 1
            self.metrics["total_duration"] += duration
            self.metrics["avg_duration"] = self.metrics["total_duration"] / self.metrics["request_count"]
            self.metrics["max_duration"] = max(self.metrics["max_duration"], duration)
            self.metrics["min_duration"] = min(self.metrics["min_duration"], duration)
            
        except Exception as e:
            self.metrics["error_count"] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

# Real-world middleware factory
def create_real_middleware_stack(app: ASGIApp) -> ASGIApp:
    """Create real-world middleware stack"""
    # Apply middleware in reverse order (last applied is first executed)
    app = RealMetricsMiddleware(app)
    app = RealLoggingMiddleware(app)
    app = RealRateLimitMiddleware(app)
    app = RealSecurityMiddleware(app)
    app = RealCORSMiddleware(app)
    app = RealMiddleware(app)
    
    return app

# Real-world error handlers
def create_real_error_handlers(app: ASGIApp) -> None:
    """Create real-world error handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
                "timestamp": datetime.now().isoformat()
            }
        )

# Real-world request processing
async def process_request(request: Request) -> Dict[str, Any]:
    """Process request with early returns"""
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host,
        "timestamp": datetime.now().isoformat()
    }

async def process_response(response: Response, duration: float) -> Dict[str, Any]:
    """Process response with early returns"""
    return {
        "status_code": response.status_code,
        "duration": duration,
        "timestamp": datetime.now().isoformat()
    }

# Export functions
__all__ = [
    # Middleware classes
    "RealMiddleware",
    "RealCORSMiddleware",
    "RealSecurityMiddleware",
    "RealRateLimitMiddleware",
    "RealLoggingMiddleware",
    "RealMetricsMiddleware",
    
    # Factory functions
    "create_real_middleware_stack",
    "create_real_error_handlers",
    
    # Processing functions
    "process_request",
    "process_response"
]












