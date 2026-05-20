"""
Pagination helper functions.

This module provides utilities for handling pagination in API endpoints.
"""

from typing import List, Dict, Any, Optional
from math import ceil


def calculate_pagination(
    total_items: int,
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    Calculate pagination metadata.
    
    Args:
        total_items: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        Dictionary with pagination metadata:
        {
            "page": int,
            "page_size": int,
            "total": int,
            "total_pages": int,
            "has_next": bool,
            "has_previous": bool
        }
    """
    total_pages = ceil(total_items / page_size) if page_size > 0 else 0
    
    return {
        "page": page,
        "page_size": page_size,
        "total": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }


def paginate_items(
    items: List[Any],
    page: int = 1,
    page_size: int = 20
) -> tuple[List[Any], Dict[str, Any]]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items to paginate
        page: Current page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        Tuple of (paginated_items, pagination_metadata)
    
    Example:
        items, pagination = paginate_items(all_items, page=1, page_size=20)
    """
    total = len(items)
    pagination = calculate_pagination(total, page, page_size)
    
    # Calculate slice indices
    start = (page - 1) * page_size
    end = start + page_size
    
    # Slice items
    paginated_items = items[start:end]
    
    return paginated_items, pagination


def validate_pagination_params(
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    max_page_size: int = 100,
    default_page: int = 1,
    default_page_size: int = 20
) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters.
    
    Args:
        page: Page number (None to use default)
        page_size: Page size (None to use default)
        max_page_size: Maximum allowed page size
        default_page: Default page number
        default_page_size: Default page size
    
    Returns:
        Tuple of (validated_page, validated_page_size)
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Use defaults if not provided
    page = page if page is not None else default_page
    page_size = page_size if page_size is not None else default_page_size
    
    # Validate page
    if page < 1:
        raise ValueError(f"Page must be >= 1, got {page}")
    
    # Validate page_size
    if page_size < 1:
        raise ValueError(f"Page size must be >= 1, got {page_size}")
    
    if page_size > max_page_size:
        raise ValueError(f"Page size must be <= {max_page_size}, got {page_size}")
    
    return page, page_size


def build_paginated_response(
    items: List[Any],
    page: int = 1,
    page_size: int = 20,
    total: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a paginated response with items and pagination metadata.
    
    Args:
        items: List of items (already paginated)
        page: Current page number
        page_size: Page size
        total: Total count (if None, uses len(items))
        **kwargs: Additional fields to include in response
    
    Returns:
        Dictionary with items and pagination metadata
    """
    total = total if total is not None else len(items)
    pagination = calculate_pagination(total, page, page_size)
    
    response = {
        "success": True,
        "items": items,
        "pagination": pagination
    }
    
    response.update(kwargs)
    
    return response








