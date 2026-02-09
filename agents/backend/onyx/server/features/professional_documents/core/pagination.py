"""
Pagination utilities for professional documents module.

Helper functions for handling pagination in API responses.
"""

from typing import List, TypeVar, Generic, Optional
from math import ceil

T = TypeVar('T')


class PaginatedResult(Generic[T]):
    """Container for paginated results."""
    
    def __init__(
        self,
        items: List[T],
        total: int,
        page: int,
        page_size: int
    ):
        """
        Initialize paginated result.
        
        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number (1-indexed)
            page_size: Number of items per page
        """
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = ceil(total / page_size) if page_size > 0 else 0
        self.has_next = page < self.total_pages
        self.has_previous = page > 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "items": self.items,
            "pagination": {
                "page": self.page,
                "page_size": self.page_size,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous
            }
        }


def paginate(
    items: List[T],
    page: int = 1,
    page_size: int = 50
) -> PaginatedResult[T]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        PaginatedResult with paginated items and metadata
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 50
    
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    
    paginated_items = items[start:end]
    
    return PaginatedResult(
        items=paginated_items,
        total=total,
        page=page,
        page_size=page_size
    )


def calculate_offset(page: int, page_size: int) -> int:
    """
    Calculate offset from page number and page size.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        Offset value
    """
    return (page - 1) * page_size if page > 0 else 0


def validate_pagination_params(page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters.
    
    Args:
        page: Page number
        page_size: Items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (validated_page, validated_page_size)
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10
    if page_size > max_page_size:
        page_size = max_page_size
    
    return page, page_size






