"""
Comprehensive Unit Tests for Response Optimizer

Tests cover response optimization functionality with diverse test cases
"""

import pytest
from unittest.mock import patch, Mock
from fastapi.responses import Response

from utils.response_optimizer import ResponseOptimizer


class TestResponseOptimizer:
    """Test cases for ResponseOptimizer class"""
    
    def test_optimize_json_response_basic(self):
        """Test optimizing basic JSON response"""
        data = {"key": "value", "number": 123}
        result = ResponseOptimizer.optimize_json_response(data)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_optimize_json_response_complex(self):
        """Test optimizing complex JSON response"""
        data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42.5
        }
        result = ResponseOptimizer.optimize_json_response(data)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_optimize_json_response_with_numpy(self):
        """Test optimizing JSON with numpy arrays"""
        try:
            import numpy as np
            data = {"array": np.array([1, 2, 3])}
            result = ResponseOptimizer.optimize_json_response(data)
            
            assert isinstance(result, bytes)
        except ImportError:
            pytest.skip("NumPy not available")
    
    def test_optimize_json_response_fallback_to_json(self):
        """Test fallback to json when orjson fails"""
        with patch('utils.response_optimizer.orjson.dumps', side_effect=Exception("orjson error")):
            data = {"key": "value"}
            result = ResponseOptimizer.optimize_json_response(data)
            
            assert isinstance(result, bytes)
            assert b"key" in result
    
    def test_create_optimized_response_basic(self):
        """Test creating basic optimized response"""
        data = {"status": "success"}
        response = ResponseOptimizer.create_optimized_response(data)
        
        assert isinstance(response, Response)
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_create_optimized_response_custom_status(self):
        """Test creating response with custom status code"""
        data = {"error": "not found"}
        response = ResponseOptimizer.create_optimized_response(data, status_code=404)
        
        assert response.status_code == 404
    
    def test_create_optimized_response_with_headers(self):
        """Test creating response with custom headers"""
        data = {"result": "ok"}
        headers = {"X-Custom-Header": "value"}
        response = ResponseOptimizer.create_optimized_response(data, headers=headers)
        
        assert response.headers.get("X-Custom-Header") == "value"
        assert "Content-Type" in response.headers
    
    def test_create_optimized_response_content_length(self):
        """Test response includes content length"""
        data = {"key": "value"}
        response = ResponseOptimizer.create_optimized_response(data)
        
        assert "Content-Length" in response.headers
        content_length = int(response.headers["Content-Length"])
        assert content_length > 0
    
    def test_paginate_response_basic(self):
        """Test creating basic paginated response"""
        items = [1, 2, 3]
        result = ResponseOptimizer.paginate_response(items, page=1, page_size=10)
        
        assert result["items"] == items
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert result["total"] == 3
    
    def test_paginate_response_with_total(self):
        """Test paginated response with total"""
        items = [1, 2, 3]
        result = ResponseOptimizer.paginate_response(
            items, page=1, page_size=10, total=100
        )
        
        assert result["total"] == 100
        assert result["has_next"] is True
        assert result["has_prev"] is False
    
    def test_paginate_response_has_next(self):
        """Test has_next calculation"""
        items = [1, 2, 3]
        result = ResponseOptimizer.paginate_response(
            items, page=1, page_size=10, total=100
        )
        
        assert result["has_next"] is True
        
        # Last page
        result2 = ResponseOptimizer.paginate_response(
            items, page=10, page_size=10, total=100
        )
        assert result2["has_next"] is False
    
    def test_paginate_response_has_prev(self):
        """Test has_prev calculation"""
        items = [1, 2, 3]
        
        # First page
        result1 = ResponseOptimizer.paginate_response(
            items, page=1, page_size=10, total=100
        )
        assert result1["has_prev"] is False
        
        # Later page
        result2 = ResponseOptimizer.paginate_response(
            items, page=2, page_size=10, total=100
        )
        assert result2["has_prev"] is True
    
    def test_paginate_response_empty_items(self):
        """Test paginated response with empty items"""
        result = ResponseOptimizer.paginate_response([], page=1, page_size=10)
        
        assert result["items"] == []
        assert result["total"] == 0
        assert result["has_next"] is False















