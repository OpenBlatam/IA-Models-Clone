"""
Route helper utilities

This module provides common utilities for API routes to reduce duplication
and ensure consistent patterns across endpoints.
"""

from typing import TypeVar, Optional, Callable, Awaitable, Any
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from utils.cache.helpers import get_cached_or_fetch, cache_entity
from utils.exceptions import NotFoundError, ValidationError, BusinessLogicError
from utils.logger import logger
from utils.constants import DEFAULT_CACHE_TTL

T = TypeVar('T')


async def handle_route_with_cache(
    cache_key: str,
    fetch_func: Callable[[], Awaitable[T]],
    entity_name: str,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    model_class: Optional[type] = None
) -> T:
    """
    Handle route with automatic caching
    
    Args:
        cache_key: Cache key
        fetch_func: Function to fetch data
        entity_name: Name of entity (for error messages)
        cache_ttl: Cache TTL
        model_class: Optional model class for deserialization
        
    Returns:
        Fetched entity
        
    Raises:
        NotFoundError: If entity not found
    """
    try:
        return await get_cached_or_fetch(
            cache_key,
            fetch_func,
            ttl=cache_ttl,
            model_class=model_class
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error fetching {entity_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch {entity_name}"
        )


async def handle_create_route(
    entity: T,
    cache_key: str,
    entity_name: str,
    cache_ttl: int = DEFAULT_CACHE_TTL
) -> T:
    """
    Handle create route with automatic caching
    
    Args:
        entity: Created entity
        cache_key: Cache key
        entity_name: Name of entity (for logging)
        cache_ttl: Cache TTL
        
    Returns:
        Created entity
    """
    try:
        await cache_entity(cache_key, entity, cache_ttl)
        logger.info(f"{entity_name} created: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache {entity_name} {cache_key}: {e}")
    
    return entity


def handle_route_errors(func: Callable) -> Callable:
    """
    Decorator to handle common route errors
    
    Wraps route handlers to provide consistent error handling.
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except NotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
        except BusinessLogicError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )
    return wrapper







