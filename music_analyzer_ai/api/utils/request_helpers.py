"""
Request processing helper functions.

This module provides utilities for processing and transforming request data.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel


def build_criteria_dict(
    **kwargs
) -> Dict[str, Any]:
    """
    Build a criteria dictionary from keyword arguments, removing None values.
    
    Useful for building filter/search criteria from request parameters.
    
    Args:
        **kwargs: Keyword arguments to include in criteria
    
    Returns:
        Dictionary with non-None values only
    
    Example:
        criteria = build_criteria_dict(
            genres=request.genres,
            moods=request.moods,
            energy_range=request.energy_range
        )
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def extract_request_fields(
    request: BaseModel,
    fields: Optional[List[str]] = None,
    exclude_none: bool = True
) -> Dict[str, Any]:
    """
    Extract fields from a Pydantic request model.
    
    Args:
        request: Pydantic model instance
        fields: Optional list of field names to extract (all if None)
        exclude_none: Whether to exclude None values
    
    Returns:
        Dictionary with extracted fields
    
    Example:
        criteria = extract_request_fields(
            request,
            fields=["genres", "moods", "energy_range"],
            exclude_none=True
        )
    """
    if fields:
        data = {field: getattr(request, field, None) for field in fields}
    else:
        data = request.dict() if hasattr(request, 'dict') else request.model_dump()
    
    if exclude_none:
        data = {k: v for k, v in data.items() if v is not None}
    
    return data


def sanitize_query_params(
    params: Dict[str, Any],
    allowed_keys: Optional[List[str]] = None,
    exclude_none: bool = True
) -> Dict[str, Any]:
    """
    Sanitize query parameters by filtering and removing None values.
    
    Args:
        params: Dictionary of query parameters
        allowed_keys: Optional list of allowed keys (all if None)
        exclude_none: Whether to exclude None values
    
    Returns:
        Sanitized dictionary
    
    Example:
        clean_params = sanitize_query_params(
            request.query_params,
            allowed_keys=["limit", "offset", "sort"],
            exclude_none=True
        )
    """
    if allowed_keys:
        params = {k: v for k, v in params.items() if k in allowed_keys}
    
    if exclude_none:
        params = {k: v for k, v in params.items() if v is not None}
    
    return params


def merge_request_data(
    *sources: Dict[str, Any],
    exclude_none: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later sources overriding earlier ones.
    
    Args:
        *sources: Variable number of dictionaries to merge
        exclude_none: Whether to exclude None values from final result
    
    Returns:
        Merged dictionary
    
    Example:
        data = merge_request_data(
            {"default": "value"},
            request.dict(),
            {"override": "value"}
        )
    """
    result = {}
    for source in sources:
        result.update(source)
    
    if exclude_none:
        result = {k: v for k, v in result.items() if v is not None}
    
    return result








