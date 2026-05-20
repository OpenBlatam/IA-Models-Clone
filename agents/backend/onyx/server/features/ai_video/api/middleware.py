from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import json
import logging
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
🚀 FASTAPI MIDDLEWARE - AI VIDEO SYSTEM
=======================================

Middleware stack for the AI Video system.
"""


# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# MIDDLEWARE STACK
# ============================================================================

def create_middleware_stack(app) -> Any:
    """
    Create and configure middleware stack.
    
    Args:
        app: FastAPI application instance
    """
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "yourdomain.com", "*.yourdomain.com"]
    )
    
    # Performance monitoring middleware
    @app.middleware("http")
    async def performance_middleware(request: Request, call_next):
        
    """performance_middleware function."""
start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"took {process_time:.4f}s "
            f"for {request.method} {request.url.path}"
        )
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.4f}s"
            )
        
        return response
    
    # Error handling middleware
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        
    """error_handling_middleware function."""
try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error
            logger.error(
                f"Unhandled error in {request.method} {request.url.path}: {str(e)}",
                exc_info=True
            )
            
            # Return error response
            error_response = {
                "error_code": "INTERNAL_ERROR",
                "error_type": "server_error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat(),
                "request_id": request.headers.get("X-Request-ID", "unknown")
            }
            
            return Response(
                content=json.dumps(error_response),
                status_code=500,
                media_type="application/json"
            )
    
    # Request validation middleware
    @app.middleware("http")
    async def validation_middleware(request: Request, call_next):
        
    """validation_middleware function."""
# Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return Response(
                content=json.dumps({
                    "error_code": "PAYLOAD_TOO_LARGE",
                    "error_type": "validation_error",
                    "message": "Request payload too large",
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=413,
                media_type="application/json"
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return Response(
                    content=json.dumps({
                        "error_code": "INVALID_CONTENT_TYPE",
                        "error_type": "validation_error",
                        "message": "Content-Type must be application/json",
                        "timestamp": datetime.now().isoformat()
                    }),
                    status_code=415,
                    media_type="application/json"
                )
        
        return await call_next(request)

# ============================================================================
# CUSTOM MIDDLEWARE
# ============================================================================

class SecurityMiddleware:
    """Custom security middleware."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            # Add security headers
            async def send_with_headers(message) -> Any:
                if message["type"] == "http.response.start":
                    message["headers"].extend([
                        (b"X-Content-Type-Options", b"nosniff"),
                        (b"X-Frame-Options", b"DENY"),
                        (b"X-XSS-Protection", b"1; mode=block"),
                        (b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains")
                    ])
                await send(message)
            
            await self.app(scope, receive, send_with_headers)
        else:
            await self.app(scope, receive, send)

class LoggingMiddleware:
    """Custom logging middleware."""
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            # Log request details
            logger.info(f"Request: {scope['method']} {scope['path']}")
            
            # Track response
            async def send_with_logging(message) -> Any:
                if message["type"] == "http.response.start":
                    logger.info(f"Response: {message['status']}")
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_custom_headers(response: Response, headers: dict):
    """
    Add custom headers to response.
    
    Args:
        response: FastAPI response object
        headers: Dictionary of headers to add
    """
    for key, value in headers.items():
        response.headers[key] = str(value)

async def get_request_info(request: Request) -> dict:
    """
    Extract request information for logging.
    
    Args:
        request: FastAPI request object
        
    Returns:
        dict: Request information
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent"),
        "content_length": request.headers.get("content-length"),
        "content_type": request.headers.get("content-type")
    } 