from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, List, Any, Optional, Callable, Awaitable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
import structlog
import time
import json
import hashlib
from datetime import datetime, timezone
            import traceback
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Middleware Package
FastAPI best practices for middleware organization and registration.
"""


logger = structlog.get_logger()

# =============================================================================
# Middleware Types
# =============================================================================

class MiddlewareType(Enum):
    """Middleware type enumeration following FastAPI best practices."""
    REQUEST_LOGGING = "request_logging"
    ERROR_HANDLING = "error_handling"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    CACHING = "caching"
    COMPRESSION = "compression"
    CORS = "cors"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"

# =============================================================================
# Middleware Registry
# =============================================================================

class MiddlewareRegistry:
    """Middleware registry following FastAPI best practices."""
    
    def __init__(self) -> Any:
        self.middleware: Dict[str, Any] = {}
        self.middleware_order: List[str] = []
        self._is_initialized = False
    
    def register_middleware(
        self,
        name: str,
        middleware_class: type,
        middleware_type: MiddlewareType,
        priority: int = 0,
        **kwargs
    ):
        """Register middleware with the registry."""
        if name in self.middleware:
            logger.warning(f"Middleware {name} already registered, overwriting")
        
        self.middleware[name] = {
            "class": middleware_class,
            "type": middleware_type,
            "priority": priority,
            "kwargs": kwargs
        }
        
        # Insert in order based on priority
        self._insert_ordered(name, priority)
        
        logger.info(f"Registered middleware: {name} (type: {middleware_type.value}, priority: {priority})")
    
    def _insert_ordered(self, name: str, priority: int):
        """Insert middleware in priority order."""
        for i, existing_name in enumerate(self.middleware_order):
            existing_priority = self.middleware[existing_name]["priority"]
            if priority < existing_priority:
                self.middleware_order.insert(i, name)
                return
        
        self.middleware_order.append(name)
    
    def get_middleware(self, name: str) -> Optional[Dict[str, Any]]:
        """Get registered middleware."""
        return self.middleware.get(name)
    
    def get_all_middleware(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered middleware."""
        return self.middleware.copy()
    
    def get_middleware_order(self) -> List[str]:
        """Get middleware execution order."""
        return self.middleware_order.copy()
    
    def setup_app(self, app) -> Any:
        """Setup all middleware on the FastAPI app."""
        if self._is_initialized:
            return
        
        # Add middleware in priority order
        for name in self.middleware_order:
            middleware_info = self.middleware[name]
            middleware_class = middleware_info["class"]
            kwargs = middleware_info["kwargs"]
            
            if issubclass(middleware_class, BaseHTTPMiddleware):
                app.add_middleware(middleware_class, **kwargs)
            else:
                # For built-in FastAPI middleware
                app.add_middleware(middleware_class, **kwargs)
            
            logger.info(f"Added middleware: {name}")
        
        self._is_initialized = True
        logger.info("Middleware registry setup completed")

# Global middleware registry instance
middleware_registry = MiddlewareRegistry()

# =============================================================================
# Base Middleware Class
# =============================================================================

class BaseMiddleware(BaseHTTPMiddleware):
    """Base middleware class following FastAPI best practices."""
    
    def __init__(self, app, name: str = None):
        
    """__init__ function."""
super().__init__(app)
        self.name = name or self.__class__.__name__
        self.logger = structlog.get_logger(self.name)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Base dispatch method following FastAPI best practices."""
        start_time = time.time()
        
        try:
            # Pre-processing
            await self.pre_process(request)
            
            # Process request
            response = await call_next(request)
            
            # Post-processing
            await self.post_process(request, response, start_time)
            
            return response
            
        except Exception as e:
            # Error handling
            return await self.handle_error(request, e, start_time)
    
    async def pre_process(self, request: Request):
        """Pre-processing hook following FastAPI best practices."""
        pass
    
    async def post_process(self, request: Request, response: Response, start_time: float):
        """Post-processing hook following FastAPI best practices."""
        pass
    
    async def handle_error(self, request: Request, error: Exception, start_time: float) -> Response:
        """Error handling hook following FastAPI best practices."""
        duration = time.time() - start_time
        
        self.logger.error(
            f"Middleware error in {self.name}",
            error=str(error),
            method=request.method,
            url=str(request.url),
            duration_ms=duration * 1000
        )
        
        # Return error response
        return Response(
            content=json.dumps({
                "success": False,
                "message": "Internal server error",
                "error_code": "MIDDLEWARE_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }),
            status_code=500,
            media_type="application/json"
        )

# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseMiddleware):
    """Request logging middleware following FastAPI best practices."""
    
    def __init__(self, app, include_body: bool = False, include_headers: bool = True):
        
    """__init__ function."""
super().__init__(app, "RequestLogging")
        self.include_body = include_body
        self.include_headers = include_headers
    
    async def pre_process(self, request: Request):
        """Log incoming request following FastAPI best practices."""
        # Generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}_{hashlib.md5(str(request.url).encode()).hexdigest()[:8]}"
            request.state.request_id = request_id
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.include_headers:
            log_data["headers"] = dict(request.headers)
        
        self.logger.info("Request started", **log_data)
    
    async def post_process(self, request: Request, response: Response, start_time: float):
        """Log response following FastAPI best practices."""
        duration = time.time() - start_time
        request_id = getattr(request.state, "request_id", "unknown")
        
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": duration * 1000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add response headers
        if self.include_headers:
            log_data["response_headers"] = dict(response.headers)
        
        # Add response size
        if hasattr(response, "body"):
            log_data["response_size"] = len(response.body)
        
        # Set request ID in response headers
        response.headers["X-Request-ID"] = request_id
        
        # Log based on status code
        if response.status_code >= 400:
            self.logger.warning("Request completed with error", **log_data)
        else:
            self.logger.info("Request completed successfully", **log_data)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address following FastAPI best practices."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client address
        if request.client:
            return request.client.host
        
        return "unknown"

# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseMiddleware):
    """Error handling middleware following FastAPI best practices."""
    
    def __init__(self, app, include_traceback: bool = False):
        
    """__init__ function."""
super().__init__(app, "ErrorHandling")
        self.include_traceback = include_traceback
    
    async def handle_error(self, request: Request, error: Exception, start_time: float) -> Response:
        """Handle errors following FastAPI best practices."""
        duration = time.time() - start_time
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log error
        error_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": duration * 1000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.include_traceback:
            error_data["traceback"] = traceback.format_exc()
        
        self.logger.error("Unhandled error", **error_data)
        
        # Return structured error response
        error_response = {
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Include error details in development
        if self.include_traceback:
            error_response["error_details"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        
        return Response(
            content=json.dumps(error_response),
            status_code=500,
            media_type="application/json"
        )

# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseMiddleware):
    """Authentication middleware following FastAPI best practices."""
    
    def __init__(self, app, exclude_paths: List[str] = None):
        
    """__init__ function."""
super().__init__(app, "Authentication")
        self.exclude_paths = exclude_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v1/users/login",
            "/api/v1/users/register"
        ]
    
    async def pre_process(self, request: Request):
        """Authenticate request following FastAPI best practices."""
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return
        
        # Extract token from headers
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate token format
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate token (implement your token validation logic here)
        user = await self._validate_token(token)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Store user in request state
        request.state.user = user
        request.state.token = token
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token following FastAPI best practices."""
        # Implement your token validation logic here
        # This is a placeholder implementation
        try:
            # Decode and validate JWT token
            # user = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            # return user
            return {"user_id": "1", "username": "test_user"}
        except Exception as e:
            self.logger.warning(f"Token validation failed: {e}")
            return None

# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitingMiddleware(BaseMiddleware):
    """Rate limiting middleware following FastAPI best practices."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 10):
        
    """__init__ function."""
super().__init__(app, "RateLimiting")
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.rate_limits: Dict[str, List[float]] = {}
    
    async def pre_process(self, request: Request):
        """Apply rate limiting following FastAPI best practices."""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + 60))
                }
            )
        
        # Add current request to rate limit tracking
        self._add_request(client_id)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use user ID if authenticated
        if hasattr(request.state, "user"):
            return f"user_{request.state.user.get('user_id', 'unknown')}"
        
        # Use IP address as fallback
        return f"ip_{self._get_client_ip(request)}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Get client's request history
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Remove old requests outside the window
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if req_time > window_start
        ]
        
        # Check if within limits
        return len(self.rate_limits[client_id]) < self.requests_per_minute
    
    def _add_request(self, client_id: str):
        """Add current request to rate limit tracking."""
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        self.rate_limits[client_id].append(time.time())

# =============================================================================
# CORS Middleware Configuration
# =============================================================================

def create_cors_middleware(
    allow_origins: List[str] = None,
    allow_credentials: bool = True,
    allow_methods: List[str] = None,
    allow_headers: List[str] = None
) -> CORSMiddleware:
    """Create CORS middleware following FastAPI best practices."""
    
    if allow_origins is None:
        allow_origins = ["*"]  # Configure appropriately for production
    
    if allow_methods is None:
        allow_methods = ["*"]
    
    if allow_headers is None:
        allow_headers = ["*"]
    
    return CORSMiddleware(
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
    )

# =============================================================================
# Compression Middleware Configuration
# =============================================================================

def create_compression_middleware(minimum_size: int = 1000) -> GZipMiddleware:
    """Create compression middleware following FastAPI best practices."""
    return GZipMiddleware(minimum_size=minimum_size)

# =============================================================================
# Security Middleware Configuration
# =============================================================================

def create_trusted_host_middleware(allowed_hosts: List[str] = None) -> TrustedHostMiddleware:
    """Create trusted host middleware following FastAPI best practices."""
    if allowed_hosts is None:
        allowed_hosts = ["*"]  # Configure appropriately for production
    
    return TrustedHostMiddleware(allowed_hosts=allowed_hosts)

async def create_https_redirect_middleware() -> HTTPSRedirectMiddleware:
    """Create HTTPS redirect middleware following FastAPI best practices."""
    return HTTPSRedirectMiddleware()

# =============================================================================
# Middleware Setup
# =============================================================================

def setup_default_middleware(app, config: Dict[str, Any] = None):
    """Setup default middleware following FastAPI best practices."""
    if config is None:
        config = {}
    
    # Register middleware in order of execution
    middleware_registry.register_middleware(
        "trusted_host",
        create_trusted_host_middleware(config.get("allowed_hosts")),
        MiddlewareType.SECURITY,
        priority=100
    )
    
    middleware_registry.register_middleware(
        "cors",
        create_cors_middleware(
            allow_origins=config.get("cors_origins"),
            allow_credentials=config.get("cors_credentials", True),
            allow_methods=config.get("cors_methods"),
            allow_headers=config.get("cors_headers")
        ),
        MiddlewareType.CORS,
        priority=90
    )
    
    middleware_registry.register_middleware(
        "compression",
        create_compression_middleware(config.get("compression_min_size", 1000)),
        MiddlewareType.COMPRESSION,
        priority=80
    )
    
    middleware_registry.register_middleware(
        "request_logging",
        RequestLoggingMiddleware,
        MiddlewareType.REQUEST_LOGGING,
        priority=70,
        include_body=config.get("log_request_body", False),
        include_headers=config.get("log_request_headers", True)
    )
    
    middleware_registry.register_middleware(
        "error_handling",
        ErrorHandlingMiddleware,
        MiddlewareType.ERROR_HANDLING,
        priority=60,
        include_traceback=config.get("include_traceback", False)
    )
    
    middleware_registry.register_middleware(
        "authentication",
        AuthenticationMiddleware,
        MiddlewareType.AUTHENTICATION,
        priority=50,
        exclude_paths=config.get("auth_exclude_paths")
    )
    
    middleware_registry.register_middleware(
        "rate_limiting",
        RateLimitingMiddleware,
        MiddlewareType.RATE_LIMITING,
        priority=40,
        requests_per_minute=config.get("rate_limit_requests", 60),
        burst_size=config.get("rate_limit_burst", 10)
    )
    
    # Setup middleware on app
    middleware_registry.setup_app(app)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "MiddlewareType",
    "MiddlewareRegistry",
    "middleware_registry",
    "BaseMiddleware",
    "RequestLoggingMiddleware",
    "ErrorHandlingMiddleware",
    "AuthenticationMiddleware",
    "RateLimitingMiddleware",
    "create_cors_middleware",
    "create_compression_middleware",
    "create_trusted_host_middleware",
    "create_https_redirect_middleware",
    "setup_default_middleware"
] 