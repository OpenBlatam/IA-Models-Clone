"""Error handling utilities."""

import traceback
import asyncio
from typing import Optional, Dict, Any, Callable, List
from functools import wraps
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from core.exceptions import PlasticSurgeryAIException
from config.settings import settings
from utils.logger import get_logger
from utils.metrics import metrics_collector

logger = get_logger(__name__)


def format_error_response(
    error: Exception,
    status_code: int = 500,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Format error response.
    
    Args:
        error: Exception instance
        status_code: HTTP status code
        include_traceback: Whether to include traceback
        
    Returns:
        Formatted error response dict
    """
    response = {
        "error": error.__class__.__name__,
        "message": str(error),
    }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


async def handle_exception(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle exception and return appropriate response.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSONResponse with error details
    """
    # Log error
    logger.error(
        "exception_occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=exc.__class__.__name__,
        exc_info=True
    )
    
    # Record metrics
    metrics_collector.increment("errors.total")
    metrics_collector.increment(f"errors.{exc.__class__.__name__}")
    
    # Determine status code
    status_code = 500
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    elif isinstance(exc, PlasticSurgeryAIException):
        status_code = _get_exception_status_code(exc)
    
    # Format response
    include_traceback = settings.log_level.upper() == "DEBUG"
    response_data = format_error_response(exc, status_code, include_traceback)
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


def _get_exception_status_code(exc: PlasticSurgeryAIException) -> int:
    """Get HTTP status code for exception."""
    from core.exceptions import (
        VisualizationNotFoundError,
        RateLimitExceededError,
        ImageValidationError
    )
    
    if isinstance(exc, VisualizationNotFoundError):
        return 404
    elif isinstance(exc, RateLimitExceededError):
        return 429
    elif isinstance(exc, ImageValidationError):
        return 400
    else:
        return 500


def safe_execute(func, *args, default=None, **kwargs):
    """
    Safely execute a function, returning default on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Error in safe_execute: {e}")
        return default


async def safe_execute_async(coro, default=None):
    """
    Safely execute a coroutine, returning default on error.
    
    Args:
        coro: Coroutine to execute
        default: Default value on error
        
    Returns:
        Coroutine result or default
    """
    try:
        return await coro
    except Exception as e:
        logger.warning(f"Error in safe_execute_async: {e}")
        return default


# Additional functions from error_recovery.py
class RetryStrategy:
    """Retry strategy configuration."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class FallbackHandler:
    """Handler for fallback operations."""
    
    def __init__(self, fallback_func: Callable, error_message: Optional[str] = None):
        self.fallback_func = fallback_func
        self.error_message = error_message or "Fallback executed"
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute fallback function."""
        logger.warning(self.error_message)
        if asyncio.iscoroutinefunction(self.fallback_func):
            return await self.fallback_func(*args, **kwargs)
        else:
            return self.fallback_func(*args, **kwargs)


def with_fallback(fallback_func: Callable, error_message: Optional[str] = None):
    """
    Decorator to add fallback behavior.
    
    Args:
        fallback_func: Fallback function to execute on error
        error_message: Optional error message
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                handler = FallbackHandler(fallback_func, error_message)
                return await handler.execute(*args, **kwargs)
        
        return wrapper
    return decorator


class ErrorRecovery:
    """Error recovery manager."""
    
    def __init__(self):
        self.recovery_strategies: dict = {}
    
    def register_strategy(
        self,
        error_type: type,
        strategy: Callable
    ) -> None:
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = strategy
    
    async def recover(self, error: Exception, context: dict = None) -> Any:
        """Attempt to recover from error."""
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            try:
                if asyncio.iscoroutinefunction(strategy):
                    return await strategy(error, context or {})
                else:
                    return strategy(error, context or {})
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    if asyncio.iscoroutinefunction(strategy):
                        return await strategy(error, context or {})
                    else:
                        return strategy(error, context or {})
                except Exception as e:
                    logger.error(f"Recovery strategy failed: {e}")
        
        return None

