"""
Comprehensive Unit Tests for API Pagination

Tests cover pagination utilities with diverse test cases
"""

import pytest
from pydantic import ValidationError

from api.pagination import (
    PaginationParams,
    PaginatedResponse,
    create_paginated_response
)


class TestPaginationParams:
    """Test cases for PaginationParams model"""
    
    def test_pagination_params_defaults(self):
        """Test pagination params with default values"""
        params = PaginationParams()
        assert params.limit == 50
        assert params.offset == 0
    
    def test_pagination_params_custom_values(self):
        """Test pagination params with custom values"""
        params = PaginationParams(limit=25, offset=100)
        assert params.limit == 25
        assert params.offset == 100
    
    def test_pagination_params_minimum_limit(self):
        """Test pagination params with minimum limit"""
        params = PaginationParams(limit=1)
        assert params.limit == 1
    
    def test_pagination_params_maximum_limit(self):
        """Test pagination params with maximum limit"""
        params = PaginationParams(limit=100)
        assert params.limit == 100
    
    def test_pagination_params_limit_below_minimum(self):
        """Test pagination params with limit below minimum raises error"""
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)
    
    def test_pagination_params_limit_above_maximum(self):
        """Test pagination params with limit above maximum raises error"""
        with pytest.raises(ValidationError):
            PaginationParams(limit=101)
    
    def test_pagination_params_negative_offset(self):
        """Test pagination params with negative offset raises error"""
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)
    
    def test_pagination_params_large_offset(self):
        """Test pagination params with large offset"""
        params = PaginationParams(offset=1000)
        assert params.offset == 1000


class TestPaginatedResponse:
    """Test cases for PaginatedResponse model"""
    
    def test_paginated_response_basic(self):
        """Test creating basic paginated response"""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        response = PaginatedResponse(
            items=items,
            total=10,
            limit=5,
            offset=0
        )
        
        assert response.items == items
        assert response.total == 10
        assert response.limit == 5
        assert response.offset == 0
        assert response.has_more is True
    
    def test_paginated_response_no_more(self):
        """Test paginated response with no more items"""
        items = [{"id": 1}, {"id": 2}]
        response = PaginatedResponse(
            items=items,
            total=2,
            limit=5,
            offset=0
        )
        
        assert response.has_more is False
    
    def test_paginated_response_empty_items(self):
        """Test paginated response with empty items"""
        response = PaginatedResponse(
            items=[],
            total=0,
            limit=10,
            offset=0
        )
        
        assert response.items == []
        assert response.has_more is False
    
    def test_paginated_response_next_offset_with_more(self):
        """Test next_offset when there are more items"""
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=10,
            limit=5,
            offset=0
        )
        
        assert response.next_offset == 5
    
    def test_paginated_response_next_offset_no_more(self):
        """Test next_offset when there are no more items"""
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=1,
            limit=5,
            offset=0
        )
        
        assert response.next_offset is None
    
    def test_paginated_response_prev_offset_at_start(self):
        """Test prev_offset when at start"""
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=10,
            limit=5,
            offset=0
        )
        
        assert response.prev_offset is None
    
    def test_paginated_response_prev_offset_not_at_start(self):
        """Test prev_offset when not at start"""
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=10,
            limit=5,
            offset=10
        )
        
        assert response.prev_offset == 5
    
    def test_paginated_response_prev_offset_small_offset(self):
        """Test prev_offset with small offset"""
        response = PaginatedResponse(
            items=[{"id": 1}],
            total=10,
            limit=5,
            offset=3
        )
        
        assert response.prev_offset == 0
    
    def test_paginated_response_generic_type(self):
        """Test paginated response with generic type"""
        items = ["item1", "item2", "item3"]
        response = PaginatedResponse[str](
            items=items,
            total=10,
            limit=5,
            offset=0
        )
        
        assert isinstance(response.items[0], str)
        assert response.items == items


class TestCreatePaginatedResponse:
    """Test cases for create_paginated_response function"""
    
    def test_create_paginated_response_basic(self):
        """Test creating basic paginated response"""
        items = [1, 2, 3, 4, 5]
        response = create_paginated_response(
            items=items,
            total=20,
            limit=5,
            offset=0
        )
        
        assert response.items == items
        assert response.total == 20
        assert response.limit == 5
        assert response.offset == 0
        assert response.has_more is True
    
    def test_create_paginated_response_last_page(self):
        """Test creating paginated response for last page"""
        items = [16, 17, 18, 19, 20]
        response = create_paginated_response(
            items=items,
            total=20,
            limit=5,
            offset=15
        )
        
        assert response.has_more is False
        assert response.next_offset is None
    
    def test_create_paginated_response_exact_boundary(self):
        """Test paginated response at exact boundary"""
        items = [11, 12, 13, 14, 15]
        response = create_paginated_response(
            items=items,
            total=20,
            limit=5,
            offset=10
        )
        
        assert response.has_more is True
        assert response.next_offset == 15
    
    def test_create_paginated_response_empty_total(self):
        """Test paginated response with zero total"""
        response = create_paginated_response(
            items=[],
            total=0,
            limit=10,
            offset=0
        )
        
        assert response.items == []
        assert response.total == 0
        assert response.has_more is False
    
    def test_create_paginated_response_offset_greater_than_total(self):
        """Test paginated response with offset greater than total"""
        response = create_paginated_response(
            items=[],
            total=10,
            limit=5,
            offset=20
        )
        
        assert response.items == []
        assert response.has_more is False
    
    def test_create_paginated_response_complex_items(self):
        """Test paginated response with complex item types"""
        items = [
            {"id": 1, "name": "Song 1"},
            {"id": 2, "name": "Song 2"}
        ]
        response = create_paginated_response(
            items=items,
            total=2,
            limit=10,
            offset=0
        )
        
        assert len(response.items) == 2
        assert response.items[0]["name"] == "Song 1"
    
    def test_create_paginated_response_calculates_has_more_correctly(self):
        """Test has_more calculation in various scenarios"""
        # Case 1: offset + limit < total
        response1 = create_paginated_response(
            items=[1, 2, 3],
            total=10,
            limit=3,
            offset=0
        )
        assert response1.has_more is True
        
        # Case 2: offset + limit == total
        response2 = create_paginated_response(
            items=[8, 9, 10],
            total=10,
            limit=3,
            offset=7
        )
        assert response2.has_more is False
        
        # Case 3: offset + limit > total
        response3 = create_paginated_response(
            items=[9, 10],
            total=10,
            limit=3,
            offset=8
        )
        assert response3.has_more is False















