"""
Pagination Testing Helpers
Specialized helpers for pagination testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock


class PaginationTestHelpers:
    """Helpers for pagination testing"""
    
    @staticmethod
    def create_paginated_response(
        items: List[Any],
        page: int = 1,
        page_size: int = 10,
        total: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create paginated response structure"""
        total = total or len(items)
        total_pages = (total + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_items = items[start_idx:end_idx]
        
        return {
            "items": paginated_items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
    
    @staticmethod
    def assert_pagination_valid(
        response: Dict[str, Any],
        expected_page: int = 1,
        expected_page_size: int = 10
    ):
        """Assert pagination structure is valid"""
        assert "pagination" in response, "Response missing pagination"
        pagination = response["pagination"]
        
        assert "page" in pagination, "Pagination missing page"
        assert "page_size" in pagination, "Pagination missing page_size"
        assert "total" in pagination, "Pagination missing total"
        
        assert pagination["page"] == expected_page, \
            f"Page {pagination['page']} does not match expected {expected_page}"
        assert pagination["page_size"] == expected_page_size, \
            f"Page size {pagination['page_size']} does not match expected {expected_page_size}"
    
    @staticmethod
    def assert_pagination_items_count(
        response: Dict[str, Any],
        expected_count: int
    ):
        """Assert paginated items count"""
        assert "items" in response, "Response missing items"
        assert len(response["items"]) == expected_count, \
            f"Items count {len(response['items'])} does not match expected {expected_count}"


class CursorPaginationHelpers:
    """Helpers for cursor-based pagination testing"""
    
    @staticmethod
    def create_cursor_paginated_response(
        items: List[Any],
        cursor: Optional[str] = None,
        limit: int = 10,
        has_next: bool = False
    ) -> Dict[str, Any]:
        """Create cursor-based paginated response"""
        return {
            "items": items,
            "pagination": {
                "cursor": cursor,
                "limit": limit,
                "has_next": has_next
            }
        }
    
    @staticmethod
    def assert_cursor_pagination_valid(response: Dict[str, Any]):
        """Assert cursor pagination structure is valid"""
        assert "pagination" in response, "Response missing pagination"
        pagination = response["pagination"]
        
        assert "limit" in pagination, "Pagination missing limit"
        assert "has_next" in pagination, "Pagination missing has_next"


# Convenience exports
create_paginated_response = PaginationTestHelpers.create_paginated_response
assert_pagination_valid = PaginationTestHelpers.assert_pagination_valid
assert_pagination_items_count = PaginationTestHelpers.assert_pagination_items_count

create_cursor_paginated_response = CursorPaginationHelpers.create_cursor_paginated_response
assert_cursor_pagination_valid = CursorPaginationHelpers.assert_cursor_pagination_valid



