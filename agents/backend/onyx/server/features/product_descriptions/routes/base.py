from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextvars import ContextVar
import logging
import time
from datetime import datetime
from ..dependencies import (
from ..schemas.base import BaseResponse, ErrorResponse
from ..utils.logging import get_logger
from typing import Any, List, Dict, Optional
import asyncio
"""
Base Router with Common Dependencies

This module provides the base router with shared dependencies,
middleware, and common functionality used across all route modules.
"""


# Import shared dependencies and utilities
    get_current_user,
    get_db_session,
    get_cache_manager,
    get_performance_monitor,
    get_error_monitor,
    get_async_io_manager
)

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["base"])

# Security
security = HTTPBearer(auto_error=False)

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
start_time_var: ContextVar[Optional[float]] = ContextVar("start_time", default=None)

# Logger
logger = get_logger(__name__)

# Common response models
class HealthResponse(BaseResponse):
    """Health check response model."""
    status: str = "healthy"
    timestamp: datetime
    version: str = "1.0.0"

class StatusResponse(BaseResponse):
    """Status response model."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Common dependencies
async def get_request_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    current_user = Depends(get_current_user),
    db_session = Depends(get_db_session),
    cache_manager = Depends(get_cache_manager),
    performance_monitor = Depends(get_performance_monitor),
    error_monitor = Depends(get_error_monitor),
    async_io_manager = Depends(get_async_io_manager)
) -> Dict[str, Any]:
    """
    Common dependency that provides all shared resources and context.
    
    This dependency is used across all routes to ensure consistent
    access to database, cache, monitoring, and user context.
    """
    return {
        "user": current_user,
        "db_session": db_session,
        "cache_manager": cache_manager,
        "performance_monitor": performance_monitor,
        "error_monitor": error_monitor,
        "async_io_manager": async_io_manager,
        "credentials": credentials
    }

async def get_authenticated_context(
    context: Dict[str, Any] = Depends(get_request_context)
) -> Dict[str, Any]:
    """
    Dependency that requires authentication.
    
    Ensures the user is authenticated before proceeding.
    """
    if not context["user"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return context

async def get_admin_context(
    context: Dict[str, Any] = Depends(get_authenticated_context)
) -> Dict[str, Any]:
    """
    Dependency that requires admin privileges.
    
    Ensures the user has admin role before proceeding.
    """
    if not context["user"].is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return context

# Base routes
@router.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint providing API information."""
    return StatusResponse(
        status="success",
        message="Product Descriptions API",
        data={
            "version": "1.0.0",
            "description": "AI-powered product description generation API",
            "endpoints": {
                "health": "/api/v1/health",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="success",
        message="Service is healthy",
        timestamp=datetime.utcnow(),
        data={
            "uptime": time.time(),
            "version": "1.0.0"
        }
    )

@router.get("/status", response_model=StatusResponse)
async def status_check(context: Dict[str, Any] = Depends(get_request_context)):
    """Detailed status check with dependency injection."""
    try:
        # Check database connectivity
        db_status = "healthy"
        try:
            await context["db_session"].execute("SELECT 1")
        except Exception as e:
            db_status = f"error: {str(e)}"
            logger.error(f"Database health check failed: {e}")
        
        # Check cache connectivity
        cache_status = "healthy"
        try:
            await context["cache_manager"].ping()
        except Exception as e:
            cache_status = f"error: {str(e)}"
            logger.error(f"Cache health check failed: {e}")
        
        return StatusResponse(
            status="success",
            message="Service status check completed",
            data={
                "database": db_status,
                "cache": cache_status,
                "performance_monitor": "active",
                "error_monitor": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Status check failed"
        )

# Error handlers
@router.exception_handler(HTTPException)
async async def http_exception_handler(request, exc) -> Any:
    """Global HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return ErrorResponse(
        status="error",
        message=exc.detail,
        error_code=exc.status_code
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc) -> Any:
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return ErrorResponse(
        status="error",
        message="Internal server error",
        error_code=500
    )

# Utility functions for other routers
def create_route_context(
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized route context."""
    return {
        "user_id": user_id,
        "request_id": request_id,
        "timestamp": datetime.utcnow(),
        "start_time": time.time()
    }

def log_route_access(
    route_name: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
):
    """Log route access for monitoring."""
    logger.info(
        f"Route accessed: {route_name}",
        extra={
            "route_name": route_name,
            "user_id": user_id,
            "request_id": request_id,
            **kwargs
        }
    ) 