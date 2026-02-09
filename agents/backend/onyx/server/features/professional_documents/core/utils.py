"""
Utility functions and decorators for the professional documents module.
"""

import functools
import inspect
import logging
from typing import Callable, TypeVar, ParamSpec, Awaitable, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def handle_api_errors(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to handle common API error patterns.
    
    Automatically handles HTTPException re-raising and converts
    other exceptions to HTTPException with proper logging.
    Works with both sync and async functions.
    """
    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to {func.__name__.replace('_', ' ')}: {str(e)}"
            )
    
    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to {func.__name__.replace('_', ' ')}: {str(e)}"
            )
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

