"""
Helper functions for TruthGPT client operations.

Centralizes common patterns to eliminate duplication.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar
from functools import wraps

from .truthgpt_status import TruthGPTStatus

logger = logging.getLogger(__name__)

T = TypeVar('T')


def require_truthgpt_available(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to ensure TruthGPT is available before executing function.
    
    Returns fallback response if TruthGPT is not available.
    
    Examples:
        >>> @require_truthgpt_available
        ... async def process_query(query: str) -> Dict[str, Any]:
        ...     # This will only execute if TruthGPT is available
        ...     return await process_with_truthgpt(query)
    """
    @wraps(func)
    async def wrapper(self, query: str, *args, **kwargs) -> T:
        if not TruthGPTStatus.is_available():
            return TruthGPTStatus.get_fallback_response(query)  # type: ignore
        return await func(self, query, *args, **kwargs)
    return wrapper


def require_integration_manager(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to ensure integration manager is initialized.
    
    Returns fallback response if manager is not available.
    
    Examples:
        >>> @require_integration_manager
        ... async def process_with_truthgpt(self, query: str) -> Dict[str, Any]:
        ...     # This will only execute if manager is available
        ...     return await self._integration_manager.integrate(...)
    """
    @wraps(func)
    async def wrapper(self, query: str, *args, **kwargs) -> T:
        if not self._integration_manager:
            return TruthGPTStatus.get_fallback_response(query)  # type: ignore
        return await func(self, query, *args, **kwargs)
    return wrapper


def handle_truthgpt_operation(
    operation: Callable[[], Awaitable[T]],
    query: str,
    operation_name: str = "TruthGPT operation"
) -> T:
    """
    Handle TruthGPT operations with consistent error handling.
    
    Args:
        operation: Async operation to execute
        query: Original query for fallback
        operation_name: Name of operation for logging
        
    Returns:
        Operation result or fallback response
        
    Examples:
        >>> result = await handle_truthgpt_operation(
        ...     lambda: self._integration_manager.integrate(data),
        ...     query,
        ...     "process query"
        ... )
    """
    async def _execute():
        try:
            return await operation()
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            return TruthGPTStatus.get_error_response(query, e)
    
    import asyncio
    if asyncio.iscoroutinefunction(operation):
        return _execute()
    else:
        # For sync operations wrapped in async
        try:
            result = operation()
            if hasattr(result, '__await__'):
                return result
            return result
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            return TruthGPTStatus.get_error_response(query, e)


def check_truthgpt_ready(
    integration_manager: Optional[Any],
    analytics_manager: Optional[Any] = None
) -> bool:
    """
    Check if TruthGPT is ready for operations.
    
    Args:
        integration_manager: Integration manager instance
        analytics_manager: Optional analytics manager instance
        
    Returns:
        True if TruthGPT is ready, False otherwise
    """
    return (
        TruthGPTStatus.is_available() and
        integration_manager is not None and
        (analytics_manager is None or analytics_manager is not None)
    )


async def safe_truthgpt_call(
    query: str,
    operation: Callable[[], Awaitable[T]],
    fallback_value: T,
    operation_name: str = "TruthGPT operation"
) -> T:
    """
    Safely call TruthGPT operation with fallback.
    
    Args:
        query: Original query
        operation: Async operation to execute
        fallback_value: Value to return on error
        operation_name: Name of operation for logging
        
    Returns:
        Operation result or fallback value
        
    Examples:
        >>> result = await safe_truthgpt_call(
        ...     query,
        ...     lambda: self._integration_manager.integrate(data),
        ...     query,  # fallback to original query
        ...     "optimize query"
        ... )
    """
    try:
        return await operation()
    except Exception as e:
        logger.warning(f"{operation_name} failed: {e}")
        return fallback_value

