"""
Pagination helper functions

Functions for calculating pagination metadata and validating pagination parameters.
"""

from typing import Dict, Any, Tuple

from ..constants import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE,
    MAX_PAGE_SIZE,
    MIN_PAGE,
    MIN_PAGE_SIZE,
)
from .math_helpers import clamp_value


def calculate_pagination_metadata(
    total: int,
    page: int,
    page_size: int
) -> Dict[str, Any]:
    """
    Calculates pagination metadata.
    
    Args:
        total: Total number of items
        page: Current page (1-indexed)
        page_size: Page size
        
    Returns:
        Dictionary with pagination metadata
    """
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    has_more = (page * page_size) < total
    has_previous = page > 1
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_more": has_more,
        "has_previous": has_previous,
        "offset": (page - 1) * page_size
    }


def validate_and_calculate_pagination(
    page: int,
    page_size: int,
    max_page: int = MAX_PAGE,
    max_page_size: int = MAX_PAGE_SIZE
) -> Tuple[int, int, int]:
    """
    Validate pagination parameters and calculate skip offset.
    
    Args:
        page: Page number (1-indexed)
        page_size: Page size
        max_page: Maximum allowed page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (validated_page, validated_page_size, skip)
    """
    validated_page = clamp_value(page, MIN_PAGE, max_page)
    validated_page_size = clamp_value(page_size, MIN_PAGE_SIZE, max_page_size)
    skip = (validated_page - 1) * validated_page_size
    
    return validated_page, validated_page_size, skip


def validate_page_params(page: int, page_size: int, max_page: int = MAX_PAGE, max_page_size: int = MAX_PAGE_SIZE) -> Tuple[int, int]:
    """
    Validates and normalizes pagination parameters.
    
    Args:
        page: Requested page
        page_size: Requested page size
        max_page: Maximum allowed page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (validated_page, validated_page_size)
    """
    if page < MIN_PAGE:
        page = DEFAULT_PAGE
    elif page > max_page:
        page = max_page
    
    if page_size < MIN_PAGE_SIZE:
        page_size = DEFAULT_PAGE_SIZE
    elif page_size > max_page_size:
        page_size = max_page_size
    
    return page, page_size



