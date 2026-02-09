"""
Endpoint builder helper functions.

This module provides utilities for building complete endpoints
with all common patterns applied automatically.
"""

from typing import Any, Optional, Dict, Type, Callable, Awaitable
from fastapi import APIRouter, Depends
from functools import wraps


def build_endpoint(
    router: APIRouter,
    method: str,
    path: str,
    handler: Callable[..., Awaitable[Any]],
    response_model: Optional[Type] = None,
    error_responses: Optional[Dict[int, Dict[str, Type]]] = None,
    use_exception_handler: bool = True,
    tags: Optional[list] = None,
    summary: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """
    Build a complete endpoint with all common patterns applied.
    
    This is a convenience function that applies:
    - Route decorator with response_model and error_responses
    - Exception handling decorator (if enabled)
    - Logging (if enabled)
    
    Args:
        router: APIRouter instance
        method: HTTP method (get, post, put, delete, patch)
        path: Route path
        handler: Async handler function
        response_model: Optional response model class
        error_responses: Optional error responses dict
        use_exception_handler: Whether to apply exception handler
        tags: Optional list of tags
        summary: Optional route summary
        description: Optional route description
    
    Returns:
        Decorated handler function
    
    Example:
        analyze_endpoint = build_endpoint(
            router,
            "post",
            "",
            analyze_track_handler,
            response_model=AnalysisResponse,
            error_responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
            use_exception_handler=True
        )
    """
    from .controller_helpers import handle_use_case_exceptions
    from .route_helpers import create_route_decorator
    
    # Apply exception handler if requested
    if use_exception_handler:
        handler = handle_use_case_exceptions(handler)
    
    # Create and apply route decorator
    route_decorator = create_route_decorator(
        router,
        method,
        path,
        response_model=response_model,
        error_responses=error_responses,
        tags=tags,
        summary=summary,
        description=description
    )
    
    return route_decorator(handler)


def endpoint_factory(
    router: APIRouter,
    response_model: Optional[Type] = None,
    error_responses: Optional[Dict[int, Dict[str, Type]]] = None,
    use_exception_handler: bool = True,
    tags: Optional[list] = None
):
    """
    Create an endpoint factory with pre-configured settings.
    
    Returns a function that can be used to quickly create endpoints
    with the same configuration.
    
    Args:
        router: APIRouter instance
        response_model: Default response model
        error_responses: Default error responses
        use_exception_handler: Whether to use exception handler by default
        tags: Default tags
    
    Returns:
        Factory function for creating endpoints
    
    Example:
        create_endpoint = endpoint_factory(
            router,
            response_model=AnalysisResponse,
            error_responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
            tags=["Analysis"]
        )
        
        # Use factory
        create_endpoint("post", "", analyze_track_handler)
        create_endpoint("get", "/{track_id}", analyze_track_by_id_handler)
    """
    def factory(
        method: str,
        path: str,
        handler: Callable[..., Awaitable[Any]],
        **override_kwargs
    ):
        return build_endpoint(
            router,
            method,
            path,
            handler,
            response_model=override_kwargs.get("response_model", response_model),
            error_responses=override_kwargs.get("error_responses", error_responses),
            use_exception_handler=override_kwargs.get("use_exception_handler", use_exception_handler),
            tags=override_kwargs.get("tags", tags),
            summary=override_kwargs.get("summary"),
            description=override_kwargs.get("description")
        )
    
    return factory








