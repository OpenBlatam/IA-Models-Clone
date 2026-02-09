"""
Route definition helper functions.

This module provides utilities for creating routes with consistent
patterns, reducing boilerplate in route decorators.
"""

from typing import Any, Optional, Dict, Type, Callable
from fastapi import APIRouter, Query, Depends
from functools import wraps


def create_route_decorator(
    router: APIRouter,
    method: str,
    path: str,
    response_model: Optional[Type] = None,
    error_responses: Optional[Dict[int, Dict[str, Type]]] = None,
    tags: Optional[list] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """
    Create a route decorator with standardized configuration.
    
    Args:
        router: APIRouter instance
        method: HTTP method (get, post, put, delete, patch)
        path: Route path
        response_model: Optional response model class
        error_responses: Optional dict mapping status codes to error models
        tags: Optional list of tags
        summary: Optional route summary
        description: Optional route description
    
    Returns:
        Decorator function
    
    Example:
        analyze_route = create_route_decorator(
            router,
            "post",
            "",
            response_model=AnalysisResponse,
            error_responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
        )
        
        @analyze_route
        @handle_use_case_exceptions
        async def analyze_track(...):
            ...
    """
    # Build responses dict
    responses = error_responses or {}
    
    # Get router method
    route_method = getattr(router, method.lower())
    
    # Create decorator
    def decorator(func: Callable) -> Callable:
        kwargs = {}
        if response_model:
            kwargs["response_model"] = response_model
        if responses:
            kwargs["responses"] = responses
        if tags:
            kwargs["tags"] = tags
        if summary:
            kwargs["summary"] = summary
        if description:
            kwargs["description"] = description
        
        return route_method(path, **kwargs)(func)
    
    return decorator


def standard_error_responses(
    *status_codes: int,
    error_model: Type = None
) -> Dict[int, Dict[str, Type]]:
    """
    Create standard error responses dictionary.
    
    Args:
        *status_codes: Status codes to include (e.g., 400, 404, 500)
        error_model: Error model class (uses ErrorResponse if None)
    
    Returns:
        Dictionary for FastAPI responses parameter
    
    Example:
        responses = standard_error_responses(404, 500, error_model=ErrorResponse)
        # Returns: {404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
    """
    if error_model is None:
        # Try to import ErrorResponse
        try:
            from ..v1.schemas.responses import ErrorResponse
            error_model = ErrorResponse
        except ImportError:
            from typing import Dict as DictType
            error_model = DictType[str, Any]
    
    return {code: {"model": error_model} for code in status_codes}


def create_query_param(
    default: Any = ...,
    description: Optional[str] = None,
    ge: Optional[int] = None,
    le: Optional[int] = None,
    alias: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create a Query parameter with common validation patterns.
    
    Args:
        default: Default value (use ... for required)
        description: Parameter description
        ge: Minimum value (greater than or equal)
        le: Maximum value (less than or equal)
        alias: Parameter alias
        **kwargs: Additional Query() parameters
    
    Returns:
        Query parameter
    
    Example:
        limit = create_query_param(
            default=20,
            description="Maximum number of results",
            ge=1,
            le=50
        )
    """
    query_kwargs = {}
    
    if default is not ...:
        query_kwargs["default"] = default
    
    if description:
        query_kwargs["description"] = description
    
    if ge is not None:
        query_kwargs["ge"] = ge
    
    if le is not None:
        query_kwargs["le"] = le
    
    if alias:
        query_kwargs["alias"] = alias
    
    query_kwargs.update(kwargs)
    
    return Query(**query_kwargs)


def create_limit_param(
    default: int = 20,
    min_val: int = 1,
    max_val: int = 100,
    description: Optional[str] = None
) -> int:
    """
    Create a standardized limit query parameter.
    
    Args:
        default: Default limit value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        description: Optional description
    
    Returns:
        Query parameter for limit
    
    Example:
        limit: int = create_limit_param(default=20, min_val=1, max_val=50)
    """
    return create_query_param(
        default=default,
        description=description or f"Maximum number of results ({min_val}-{max_val})",
        ge=min_val,
        le=max_val
    )


def create_offset_param(
    default: int = 0,
    min_val: int = 0,
    description: Optional[str] = None
) -> int:
    """
    Create a standardized offset query parameter.
    
    Args:
        default: Default offset value
        min_val: Minimum allowed value (usually 0)
        description: Optional description
    
    Returns:
        Query parameter for offset
    
    Example:
        offset: int = create_offset_param(default=0)
    """
    return create_query_param(
        default=default,
        description=description or "Pagination offset",
        ge=min_val
    )


def create_page_param(
    default: int = 1,
    min_val: int = 1,
    description: Optional[str] = None
) -> int:
    """
    Create a standardized page query parameter.
    
    Args:
        default: Default page value
        min_val: Minimum allowed value (usually 1)
        description: Optional description
    
    Returns:
        Query parameter for page
    
    Example:
        page: int = create_page_param(default=1)
    """
    return create_query_param(
        default=default,
        description=description or "Page number (1-indexed)",
        ge=min_val
    )


def create_page_size_param(
    default: int = 20,
    min_val: int = 1,
    max_val: int = 100,
    description: Optional[str] = None
) -> int:
    """
    Create a standardized page_size query parameter.
    
    Args:
        default: Default page size value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        description: Optional description
    
    Returns:
        Query parameter for page_size
    
    Example:
        page_size: int = create_page_size_param(default=20, max_val=100)
    """
    return create_query_param(
        default=default,
        description=description or f"Items per page ({min_val}-{max_val})",
        ge=min_val,
        le=max_val
    )








