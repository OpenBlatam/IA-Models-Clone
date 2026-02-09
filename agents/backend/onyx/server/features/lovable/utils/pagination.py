"""
Pagination utilities for consistent pagination across the API.
"""

from typing import Tuple, List, TypeVar, Generic
from math import ceil

T = TypeVar('T')


class PaginationResult(Generic[T]):
    """Pagination result container."""
    
    def __init__(
        self,
        items: List[T],
        page: int,
        page_size: int,
        total: int
    ):
        """Initialize pagination result."""
        self.items = items
        self.page = page
        self.page_size = page_size
        self.total = total
        self.total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "items": self.items,
            "page": self.page,
            "page_size": self.page_size,
            "total": self.total,
            "total_pages": self.total_pages,
            "has_next": self.page < self.total_pages,
            "has_previous": self.page > 1
        }


def paginate(
    items: List[T],
    page: int,
    page_size: int
) -> Tuple[List[T], int]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        Tuple of (paginated_items, total_count)
    """
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_items = items[start:end]
    
    return paginated_items, total


def calculate_pagination_metadata(
    page: int,
    page_size: int,
    total: int
) -> dict:
    """
    Calculate pagination metadata.
    
    Args:
        page: Current page number
        page_size: Items per page
        total: Total number of items
        
    Returns:
        Dictionary with pagination metadata
    """
    total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1
    }






