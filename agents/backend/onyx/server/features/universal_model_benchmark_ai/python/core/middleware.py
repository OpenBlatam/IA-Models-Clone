"""
Middleware Module - Middleware system for request processing.

Provides:
- Request/response middleware
- Pipeline processing
- Middleware chaining
- Context passing
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Request object."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    body: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Response:
    """Response object."""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    context: Dict[str, Any] = field(default_factory=dict)


MiddlewareFunc = Callable[[Request, Callable], Response]


class MiddlewarePipeline:
    """Middleware pipeline."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.middlewares: List[MiddlewareFunc] = []
    
    def use(self, middleware: MiddlewareFunc) -> None:
        """
        Add middleware to pipeline.
        
        Args:
            middleware: Middleware function
        """
        self.middlewares.append(middleware)
        logger.info(f"Added middleware: {middleware.__name__}")
    
    def execute(self, request: Request, handler: Callable) -> Response:
        """
        Execute middleware pipeline.
        
        Args:
            request: Request object
            handler: Final handler function
            
        Returns:
            Response object
        """
        def next_middleware(req: Request, index: int = 0) -> Response:
            if index >= len(self.middlewares):
                return handler(req)
            
            middleware = self.middlewares[index]
            return middleware(req, lambda r: next_middleware(r, index + 1))
        
        return next_middleware(request)
    
    def clear(self) -> None:
        """Clear all middlewares."""
        self.middlewares.clear()


# Common middleware functions

def logging_middleware(request: Request, next_handler: Callable) -> Response:
    """Logging middleware."""
    start_time = time.time()
    logger.info(f"{request.method} {request.path}")
    
    response = next_handler(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.path} - "
        f"{response.status_code} - {duration:.3f}s"
    )
    
    return response


def timing_middleware(request: Request, next_handler: Callable) -> Response:
    """Timing middleware."""
    start_time = time.time()
    response = next_handler(request)
    duration = time.time() - start_time
    
    response.context["duration"] = duration
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response


def cors_middleware(request: Request, next_handler: Callable) -> Response:
    """CORS middleware."""
    response = next_handler(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    
    return response


def error_handling_middleware(request: Request, next_handler: Callable) -> Response:
    """Error handling middleware."""
    try:
        return next_handler(request)
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return Response(
            status_code=500,
            body={"error": str(e)},
        )


def rate_limit_middleware(rate_limiter) -> MiddlewareFunc:
    """Create rate limiting middleware."""
    def middleware(request: Request, next_handler: Callable) -> Response:
        # Extract user ID from request
        user_id = request.context.get("user_id", request.headers.get("X-User-ID", "anonymous"))
        
        allowed, remaining = rate_limiter.check_rate_limit(user_id)
        
        if not allowed:
            return Response(
                status_code=429,
                headers={"X-RateLimit-Remaining": str(remaining or 0)},
                body={"error": "Rate limit exceeded"},
            )
        
        response = next_handler(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining or 0)
        
        return response
    
    return middleware


def authentication_middleware(auth_manager) -> MiddlewareFunc:
    """Create authentication middleware."""
    def middleware(request: Request, next_handler: Callable) -> Response:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if not token:
            return Response(
                status_code=401,
                body={"error": "Missing authentication token"},
            )
        
        payload = auth_manager.verify_token(token)
        if not payload:
            return Response(
                status_code=401,
                body={"error": "Invalid authentication token"},
            )
        
        request.context["user_id"] = payload.get("sub")
        request.context["user_role"] = payload.get("role")
        
        return next_handler(request)
    
    return middleware












