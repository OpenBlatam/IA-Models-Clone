"""
Pagination utilities for API responses
"""

from typing import List, TypeVar, Generic, Optional
from math import ceil

T = TypeVar('T')


def calculate_pagination(
    total_items: int,
    page: int = 1,
    page_size: int = 10
) -> dict[str, int | bool]:
    """
    Calculate pagination metadata
    
    Args:
        total_items: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        Dictionary with pagination metadata
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10
    
    total_pages = ceil(total_items / page_size) if total_items > 0 else 0
    
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }


def paginate_items(
    items: List[T],
    page: int = 1,
    page_size: int = 10
) -> tuple[List[T], dict[str, int | bool]]:
    """
    Paginate a list of items
    
    Args:
        items: List of items to paginate
        page: Current page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        Tuple of (paginated_items, pagination_metadata)
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10
    
    total_items = len(items)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    
    paginated_items = items[start_index:end_index]
    pagination_meta = calculate_pagination(total_items, page, page_size)
    
    return paginated_items, pagination_meta


def validate_pagination_params(
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    max_page_size: int = 100
) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters
    
    Args:
        page: Page number
        page_size: Items per page
        max_page_size: Maximum allowed page size
    
    Returns:
        Tuple of (validated_page, validated_page_size)
    
    Raises:
        ValueError if parameters are invalid
    """
    validated_page = page if page and page > 0 else 1
    validated_page_size = page_size if page_size and page_size > 0 else 10
    
    if validated_page_size > max_page_size:
        raise ValueError(f"page_size cannot exceed {max_page_size}")
    
    return validated_page, validated_page_size

