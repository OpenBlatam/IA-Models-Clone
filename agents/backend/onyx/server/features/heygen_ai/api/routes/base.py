from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone
import structlog
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import traceback
from functools import wraps
from pydantic import BaseModel, Field, validator
    from api.async_operations.async_database import AsyncDatabaseOperations
    from api.async_operations.async_external_api import AsyncExternalAPIOperations
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Base Route Class for HeyGen AI FastAPI
Provides common functionality and dependency injection for all routes.
"""



logger = structlog.get_logger()

# =============================================================================
# Route Types
# =============================================================================

class RouteType(Enum):
    """Route type enumeration."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class RouteCategory(Enum):
    """Route category enumeration."""
    AUTH = "Authentication and Authorization"
    USERS = "User Management"
    VIDEOS = "Video Processing"
    AI = "AI and Machine Learning"
    ANALYTICS = "Analytics and Reporting"
    SYSTEM = "System and Health"
    EXTERNAL = "External API Integration"
    UTILS = "Utility and Helper Functions"

@dataclass
class RouteMetadata:
    """Route metadata for documentation and organization."""
    name: str
    description: str
    category: RouteCategory
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    version: str = "1.0.0"
    requires_auth: bool = False
    rate_limited: bool = True
    cacheable: bool = False
    cache_ttl: int = 300

@dataclass
class RouteMetrics:
    """Route performance metrics."""
    route_name: str
    method: str
    path: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    request_size_bytes: int = 0
    response_size_bytes: int = 0

# =============================================================================
# Base Response Models
# =============================================================================

class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None

class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    data: List[Any] = Field(default_factory=list)
    pagination: Dict[str, Any] = Field(default_factory=dict)
    total_count: int = 0
    page: int = 1
    page_size: int = 10
    total_pages: int = 0

# =============================================================================
# Base Route Class
# =============================================================================

class BaseRoute:
    """Base route class providing common functionality for all routes."""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: RouteCategory,
        tags: Optional[List[str]] = None,
        prefix: str = "",
        dependencies: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
self.name = name
        self.description = description
        self.category = category
        self.tags = tags or []
        self.prefix = prefix
        self.dependencies = dependencies or {}
        
        # Create router
        self.router = APIRouter(
            prefix=prefix,
            tags=self.tags,
            responses={
                200: {"description": "Success"},
                400: {"description": "Bad Request"},
                401: {"description": "Unauthorized"},
                403: {"description": "Forbidden"},
                404: {"description": "Not Found"},
                500: {"description": "Internal Server Error"}
            }
        )
        
        # Route metadata
        self.metadata = RouteMetadata(
            name=name,
            description=description,
            category=category,
            tags=self.tags
        )
        
        # Performance tracking
        self.metrics: Dict[str, RouteMetrics] = {}
        
        # Register common middleware
        self._register_middleware()
        
        logger.info(f"Initialized base route: {name}")
    
    def _register_middleware(self) -> Any:
        """Register common middleware for the route."""
        @self.router.middleware("http")
        async def route_middleware(request: Request, call_next):
            
    """route_middleware function."""
# Start timing
            start_time = time.time()
            request_id = request.headers.get("X-Request-ID", f"req_{int(start_time * 1000)}")
            
            # Create metrics
            metrics = RouteMetrics(
                route_name=self.name,
                method=request.method,
                path=str(request.url.path),
                start_time=datetime.now(timezone.utc),
                request_size_bytes=len(await request.body()) if request.method in ["POST", "PUT", "PATCH"] else 0
            )
            
            try:
                # Process request
                response = await call_next(request)
                
                # Record success metrics
                end_time = time.time()
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_ms = (end_time - start_time) * 1000
                metrics.success = True
                metrics.status_code = response.status_code
                metrics.response_size_bytes = len(response.body) if hasattr(response, 'body') else 0
                
                # Add request ID to response
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                # Record error metrics
                end_time = time.time()
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_ms = (end_time - start_time) * 1000
                metrics.success = False
                metrics.error_message = str(e)
                metrics.status_code = 500
                
                logger.error(f"Route error in {self.name}: {e}")
                
                # Return error response
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        success=False,
                        message="Internal server error",
                        error_code="INTERNAL_ERROR",
                        request_id=request_id
                    ).dict()
                )
            
            finally:
                # Store metrics
                self.metrics[request_id] = metrics
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """Get a dependency by name."""
        return self.dependencies.get(name)
    
    def add_dependency(self, name: str, dependency: Any):
        """Add a dependency to the route."""
        self.dependencies[name] = dependency
        logger.info(f"Added dependency {name} to route {self.name}")
    
    def success_response(
        self,
        data: Any = None,
        message: str = "Success",
        request_id: Optional[str] = None
    ) -> BaseResponse:
        """Create a success response."""
        return BaseResponse(
            success=True,
            message=message,
            data=data,
            request_id=request_id
        )
    
    def error_response(
        self,
        message: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        status_code: int = 400
    ) -> ErrorResponse:
        """Create an error response."""
        return ErrorResponse(
            success=False,
            message=message,
            error_code=error_code,
            error_details=error_details,
            request_id=request_id
        )
    
    def paginated_response(
        self,
        data: List[Any],
        total_count: int,
        page: int = 1,
        page_size: int = 10,
        request_id: Optional[str] = None
    ) -> PaginatedResponse:
        """Create a paginated response."""
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedResponse(
            success=True,
            data=data,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            pagination={
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_count": total_count,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            request_id=request_id
        )
    
    def get_metrics(self) -> Dict[str, RouteMetrics]:
        """Get route performance metrics."""
        return self.metrics.copy()
    
    def get_route_info(self) -> Dict[str, Any]:
        """Get route information."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "tags": self.tags,
            "prefix": self.prefix,
            "metadata": self.metadata.__dict__,
            "metrics": {
                "total_requests": len(self.metrics),
                "successful_requests": sum(1 for m in self.metrics.values() if m.success),
                "failed_requests": sum(1 for m in self.metrics.values() if not m.success),
                "avg_duration_ms": sum(m.duration_ms or 0 for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0
            }
        }

# =============================================================================
# Route Decorators
# =============================================================================

def route_metrics(func: Callable) -> Callable:
    """Decorator to track route metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Route {func.__name__} completed in {duration_ms:.2f}ms")
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Route {func.__name__} failed after {duration_ms:.2f}ms: {e}")
            raise
    return wrapper

def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Authentication logic would go here
        # For now, just log the requirement
        logger.info(f"Route {func.__name__} requires authentication")
        return await func(*args, **kwargs)
    return wrapper

def rate_limit(requests_per_minute: int = 60) -> Callable:
    """Decorator to apply rate limiting."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Rate limiting logic would go here
            # For now, just log the requirement
            logger.info(f"Route {func.__name__} has rate limit: {requests_per_minute} requests/minute")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def cache_response(ttl: int = 300) -> Callable:
    """Decorator to cache responses."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Caching logic would go here
            # For now, just log the requirement
            logger.info(f"Route {func.__name__} has cache TTL: {ttl} seconds")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Dependency Injection Helpers
# =============================================================================

def get_database_operations():
    """Get database operations dependency."""
    # This would return the actual database operations instance
    return AsyncDatabaseOperations

def get_api_operations():
    """Get external API operations dependency."""
    # This would return the actual API operations instance
    return AsyncExternalAPIOperations

def get_current_user():
    """Get current user dependency."""
    # This would implement actual user authentication
    return {"user_id": "test_user", "username": "test"}

async def get_request_id(request: Request) -> str:
    """Get request ID from headers or generate one."""
    return request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "RouteType",
    "RouteCategory",
    "RouteMetadata",
    "RouteMetrics",
    "BaseResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "BaseRoute",
    "route_metrics",
    "require_auth",
    "rate_limit",
    "cache_response",
    "get_database_operations",
    "get_api_operations",
    "get_current_user",
    "get_request_id"
] 