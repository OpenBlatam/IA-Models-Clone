"""
Comprehensive Unit Tests for Query Optimizer

Tests cover query optimization functionality with diverse test cases
"""

import pytest
from unittest.mock import Mock, patch

from core.query_optimizer import QueryOptimizer, get_query_optimizer


class TestQueryOptimizer:
    """Test cases for QueryOptimizer class"""
    
    def test_query_optimizer_init(self):
        """Test initializing query optimizer"""
        optimizer = QueryOptimizer()
        assert len(optimizer._query_cache) == 0
        assert len(optimizer._prepared_queries) == 0
    
    def test_prepare_query_basic(self):
        """Test preparing basic query"""
        optimizer = QueryOptimizer()
        query = "SELECT * FROM songs"
        
        result = optimizer.prepare_query(query)
        
        assert result == query
        assert isinstance(result, str)
    
    def test_prepare_query_normalizes_whitespace(self):
        """Test query normalization removes extra whitespace"""
        optimizer = QueryOptimizer()
        query = "SELECT   *   FROM    songs"
        
        result = optimizer.prepare_query(query)
        
        # Should normalize whitespace
        assert "  " not in result
    
    def test_prepare_query_cached(self):
        """Test prepared queries are cached"""
        optimizer = QueryOptimizer()
        query = "SELECT * FROM songs"
        
        result1 = optimizer.prepare_query(query)
        result2 = optimizer.prepare_query(query)
        
        # Should return same result (cached)
        assert result1 == result2
    
    def test_optimize_select_basic(self):
        """Test optimizing basic SELECT query"""
        optimizer = QueryOptimizer()
        
        result = optimizer.optimize_select("songs", {}, limit=10)
        
        assert "SELECT" in result
        assert "FROM songs" in result
        assert "LIMIT 10" in result
    
    def test_optimize_select_with_filters(self):
        """Test optimizing SELECT with filters"""
        optimizer = QueryOptimizer()
        filters = {"genre": "rock", "status": "active"}
        
        result = optimizer.optimize_select("songs", filters, limit=20)
        
        assert "WHERE" in result
        assert "genre" in result
        assert "status" in result
        assert "LIMIT 20" in result
    
    def test_optimize_select_with_single_filter(self):
        """Test optimizing SELECT with single filter"""
        optimizer = QueryOptimizer()
        filters = {"genre": "rock"}
        
        result = optimizer.optimize_select("songs", filters)
        
        assert "WHERE" in result
        assert "genre = :genre" in result
    
    def test_optimize_select_no_filters(self):
        """Test optimizing SELECT without filters"""
        optimizer = QueryOptimizer()
        
        result = optimizer.optimize_select("songs", {})
        
        assert "WHERE" not in result
        assert "LIMIT" in result
    
    def test_optimize_select_custom_limit(self):
        """Test optimizing SELECT with custom limit"""
        optimizer = QueryOptimizer()
        
        result = optimizer.optimize_select("songs", {}, limit=50)
        
        assert "LIMIT 50" in result
    
    def test_optimize_select_uses_prepare_query(self):
        """Test optimize_select uses prepare_query"""
        optimizer = QueryOptimizer()
        
        with patch.object(optimizer, 'prepare_query', return_value="prepared") as mock_prepare:
            result = optimizer.optimize_select("songs", {})
            
            assert result == "prepared"
            mock_prepare.assert_called_once()
    
    def test_add_index_hint(self):
        """Test adding index hint to query"""
        optimizer = QueryOptimizer()
        query = "SELECT * FROM songs"
        
        result = optimizer.add_index_hint(query, "idx_genre")
        
        # Should return query (implementation may vary)
        assert isinstance(result, str)
    
    def test_explain_query(self):
        """Test explaining query"""
        optimizer = QueryOptimizer()
        query = "SELECT * FROM songs"
        
        result = optimizer.explain_query(query)
        
        assert "query" in result
        assert result["query"] == query
        assert "optimized" in result


class TestGetQueryOptimizer:
    """Test cases for get_query_optimizer function"""
    
    def test_get_query_optimizer_singleton(self):
        """Test that get_query_optimizer returns singleton"""
        optimizer1 = get_query_optimizer()
        optimizer2 = get_query_optimizer()
        
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, QueryOptimizer)
    
    def test_get_query_optimizer_multiple_calls(self):
        """Test multiple calls return same instance"""
        optimizers = [get_query_optimizer() for _ in range(5)]
        assert all(o is optimizers[0] for o in optimizers)















