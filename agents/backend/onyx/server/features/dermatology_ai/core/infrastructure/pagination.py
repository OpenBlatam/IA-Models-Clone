"""
Pagination Helpers
Utilities for API response pagination
"""

from typing import List, TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class PaginatedResponse(Generic[T]):
    """Paginated response wrapper"""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_next: bool
    has_previous: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "items": self.items,
            "pagination": {
                "total": self.total,
                "limit": self.limit,
                "offset": self.offset,
                "has_next": self.has_next,
                "has_previous": self.has_previous,
                "page": (self.offset // self.limit) + 1 if self.limit > 0 else 1,
                "total_pages": (self.total + self.limit - 1) // self.limit if self.limit > 0 else 1
            }
        }


class PaginationHelper:
    """Helper for pagination operations"""
    
    @staticmethod
    def paginate(
        items: List[T],
        limit: int,
        offset: int,
        total: Optional[int] = None
    ) -> PaginatedResponse[T]:
        """
        Create paginated response
        
        Args:
            items: List of items
            limit: Items per page
            offset: Number of items to skip
            total: Total number of items (if known)
            
        Returns:
            PaginatedResponse
        """
        if total is None:
            total = len(items)
        
        # Slice items
        paginated_items = items[offset:offset + limit]
        
        has_next = offset + limit < total
        has_previous = offset > 0
        
        return PaginatedResponse(
            items=paginated_items,
            total=total,
            limit=limit,
            offset=offset,
            has_next=has_next,
            has_previous=has_previous
        )
    
    @staticmethod
    def create_pagination_links(
        base_url: str,
        limit: int,
        offset: int,
        total: int
    ) -> Dict[str, Optional[str]]:
        """
        Create pagination links
        
        Args:
            base_url: Base URL for links
            limit: Items per page
            offset: Current offset
            total: Total number of items
            
        Returns:
            Dictionary with pagination links
        """
        links = {
            "first": None,
            "prev": None,
            "next": None,
            "last": None
        }
        
        # First page
        if offset > 0:
            links["first"] = f"{base_url}?limit={limit}&offset=0"
        
        # Previous page
        if offset >= limit:
            prev_offset = max(0, offset - limit)
            links["prev"] = f"{base_url}?limit={limit}&offset={prev_offset}"
        
        # Next page
        if offset + limit < total:
            next_offset = offset + limit
            links["next"] = f"{base_url}?limit={limit}&offset={next_offset}"
        
        # Last page
        if offset + limit < total:
            last_offset = ((total - 1) // limit) * limit
            links["last"] = f"{base_url}?limit={limit}&offset={last_offset}"
        
        return links
    
    @staticmethod
    def validate_pagination(limit: int, offset: int) -> tuple[int, int]:
        """
        Validate and normalize pagination parameters
        
        Args:
            limit: Requested limit
            offset: Requested offset
            
        Returns:
            Tuple of (validated_limit, validated_offset)
        """
        # Clamp limit
        limit = max(1, min(100, limit))
        
        # Ensure non-negative offset
        offset = max(0, offset)
        
        return limit, offset















