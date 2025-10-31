"""
Pagination utilities
"""

from typing import List, TypeVar, Generic
from math import ceil

from ..models.schemas import PaginationParams, PaginatedResponse

T = TypeVar('T')


def create_paginated_response(
    items: List[T],
    total: int,
    pagination: PaginationParams
) -> PaginatedResponse:
    """Create a paginated response."""
    pages = ceil(total / pagination.size) if total > 0 else 1
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=pages
    )


def calculate_offset(page: int, size: int) -> int:
    """Calculate offset for database queries."""
    return (page - 1) * size


def validate_pagination_params(page: int, size: int, max_size: int = 100) -> tuple[int, int]:
    """Validate and normalize pagination parameters."""
    # Ensure page is at least 1
    page = max(1, page)
    
    # Ensure size is within bounds
    size = max(1, min(size, max_size))
    
    return page, size






























