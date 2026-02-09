"""
Helper functions and decorators for API routes

This module provides decorators for consistent error handling and metrics tracking
across all API routes.
"""

from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any
from fastapi import HTTPException
from ..core.logging_config import get_logger
from ..core.exceptions import PhysicalStoreDesignerError, ServiceError
from ..core.metrics import get_metrics_collector, time_operation

P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger(__name__)


def handle_route_errors(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to handle errors in route handlers consistently
    Converts exceptions to appropriate HTTP responses
    """
    @wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except PhysicalStoreDesignerError as e:
            # Custom exceptions are already handled by middleware
            raise
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}",
                extra={"function": func.__name__},
                exc_info=True
            )
            get_metrics_collector().increment("http.errors", tags={"type": "unexpected"})
            raise ServiceError(
                service=func.__name__,
                message=str(e),
                details={"error_type": type(e).__name__}
            )
    
    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except PhysicalStoreDesignerError as e:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}",
                extra={"function": func.__name__},
                exc_info=True
            )
            get_metrics_collector().increment("http.errors", tags={"type": "unexpected"})
            raise ServiceError(
                service=func.__name__,
                message=str(e),
                details={"error_type": type(e).__name__}
            )
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def track_route_metrics(metric_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to track route metrics
    
    Args:
        metric_name: Name of the metric to track
        
    Returns:
        Decorator function that wraps the route handler
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with time_operation(metric_name, tags={"route": func.__name__}):
                result = await func(*args, **kwargs)
                get_metrics_collector().increment("http.requests.success", tags={"route": func.__name__})
                return result
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with time_operation(metric_name, tags={"route": func.__name__}):
                result = func(*args, **kwargs)
                get_metrics_collector().increment("http.requests.success", tags={"route": func.__name__})
                return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

