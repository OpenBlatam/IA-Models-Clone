"""
Validation helper functions.

This module provides utilities for validating request data and parameters.
"""

from typing import Any, List, Optional
from fastapi import HTTPException, Query
import re


def validate_track_id_format(track_id: str) -> None:
    """
    Validate Spotify track ID format.
    
    Spotify track IDs are typically 22-character base62 strings.
    
    Args:
        track_id: Track ID to validate
    
    Raises:
        HTTPException: If track ID format is invalid
    """
    if not track_id:
        raise HTTPException(status_code=400, detail="track_id cannot be empty")
    
    # Spotify IDs are typically 22 characters, alphanumeric
    if not re.match(r'^[a-zA-Z0-9]{22}$', track_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track_id format: {track_id}"
        )


def validate_limit(
    limit: int,
    min_val: int = 1,
    max_val: int = 100,
    default: Optional[int] = None
) -> int:
    """
    Validate and normalize limit parameter.
    
    Args:
        limit: Limit value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if limit is None
    
    Returns:
        Validated limit value
    
    Raises:
        HTTPException: If limit is invalid
    """
    if limit is None:
        if default is not None:
            return default
        raise HTTPException(status_code=400, detail="limit parameter is required")
    
    if limit < min_val:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be >= {min_val}, got {limit}"
        )
    
    if limit > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be <= {max_val}, got {limit}"
        )
    
    return limit


def validate_offset(offset: int, min_val: int = 0) -> int:
    """
    Validate and normalize offset parameter.
    
    Args:
        offset: Offset value to validate
        min_val: Minimum allowed value (default: 0)
    
    Returns:
        Validated offset value
    
    Raises:
        HTTPException: If offset is invalid
    """
    if offset < min_val:
        raise HTTPException(
            status_code=400,
            detail=f"offset must be >= {min_val}, got {offset}"
        )
    
    return offset


def validate_string_length(
    value: str,
    field_name: str,
    min_length: int = 1,
    max_length: Optional[int] = None
) -> None:
    """
    Validate string length.
    
    Args:
        value: String to validate
        field_name: Name of the field (for error messages)
        min_length: Minimum length
        max_length: Maximum length (None for no limit)
    
    Raises:
        HTTPException: If string length is invalid
    """
    if not value or len(value.strip()) < min_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at least {min_length} characters"
        )
    
    if max_length and len(value) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at most {max_length} characters"
        )


def validate_list_not_empty(
    items: List[Any],
    field_name: str,
    min_items: int = 1
) -> None:
    """
    Validate that a list is not empty.
    
    Args:
        items: List to validate
        field_name: Name of the field (for error messages)
        min_items: Minimum number of items
    
    Raises:
        HTTPException: If list is empty or too short
    """
    if not items:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty"
        )
    
    if len(items) < min_items:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must have at least {min_items} item(s)"
        )


def validate_enum_value(
    value: Any,
    allowed_values: List[Any],
    field_name: str
) -> None:
    """
    Validate that a value is in a list of allowed values.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        field_name: Name of the field (for error messages)
    
    Raises:
        HTTPException: If value is not in allowed values
    """
    if value not in allowed_values:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be one of {allowed_values}, got {value}"
        )


def validate_range(
    value: float,
    field_name: str,
    min_val: float,
    max_val: float
) -> None:
    """
    Validate that a numeric value is within a range.
    
    Args:
        value: Value to validate
        field_name: Name of the field (for error messages)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Raises:
        HTTPException: If value is outside range
    """
    if value < min_val or value > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {min_val} and {max_val}, got {value}"
        )








