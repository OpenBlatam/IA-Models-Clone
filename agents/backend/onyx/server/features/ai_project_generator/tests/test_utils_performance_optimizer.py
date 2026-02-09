"""
Tests for PerformanceOptimizer utility
"""

import pytest
import time
from ..utils.performance_optimizer import ProjectCache


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer"""

    def test_cache_init(self):
        """Test ProjectCache initialization"""
        cache = ProjectCache(max_size=100, ttl=3600)
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert len(cache.cache) == 0

    def test_cache_set_get(self):
        """Test setting and getting from cache"""
        cache = ProjectCache()
        
        project_info = {"project_id": "test-123", "name": "Test Project"}
        cache.set("A test project", project_info)
        
        result = cache.get("A test project")
        
        assert result is not None
        assert result["project_id"] == "test-123"

    def test_cache_expiration(self):
        """Test cache expiration"""
        cache = ProjectCache(ttl=1)  # 1 second TTL
        
        project_info = {"project_id": "test-123"}
        cache.set("Test project", project_info)
        
        # Should be available immediately
        assert cache.get("Test project") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("Test project") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction"""
        cache = ProjectCache(max_size=3)
        
        # Add 4 items
        cache.set("Project 1", {"id": "1"})
        cache.set("Project 2", {"id": "2"})
        cache.set("Project 3", {"id": "3"})
        cache.set("Project 4", {"id": "4"})
        
        # First item should be evicted
        assert cache.get("Project 1") is None
        assert cache.get("Project 4") is not None

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = ProjectCache()
        
        cache.set("Project 1", {"id": "1"})
        cache.set("Project 2", {"id": "2"})
        cache.get("Project 1")  # Access
        
        stats = cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0

    def test_cache_clear(self):
        """Test clearing cache"""
        cache = ProjectCache()
        
        cache.set("Project 1", {"id": "1"})
        cache.set("Project 2", {"id": "2"})
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("Project 1") is None

