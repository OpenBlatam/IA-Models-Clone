"""
Controller helper functions for consistent error handling and response building.

This module provides decorators and utilities to reduce code duplication
across API controllers.
"""

from fastapi import HTTPException
from typing import Callable, Any, TypeVar, Awaitable
import logging
from functools import wraps

from ...application.exceptions import (
    TrackNotFoundException,
    AnalysisException,
    UseCaseException
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_use_case_exceptions(
    func: Callable[..., Awaitable[T]]
) -> Callable[..., Awaitable[T]]:
    """
    Decorator to handle common use case exceptions consistently.
    
    Automatically handles:
    - TrackNotFoundException -> 404
    - AnalysisException -> 500
    - UseCaseException -> 400
    - Generic Exception -> 500 with logging
    
    Usage:
        @handle_use_case_exceptions
        async def my_endpoint(...):
            result = await use_case.execute(...)
            return result
    
    Args:
        func: Async function to wrap
    
    Returns:
        Wrapped function with exception handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TrackNotFoundException as e:
            raise HTTPException(status_code=404, detail=str(e))
        except AnalysisException as e:
            raise HTTPException(status_code=500, detail=str(e))
        except UseCaseException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {e}",
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
    return wrapper








