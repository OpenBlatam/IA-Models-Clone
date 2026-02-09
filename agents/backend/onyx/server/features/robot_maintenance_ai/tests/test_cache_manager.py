"""
Tests for cache manager.
"""

import time
import pytest
from ..utils.cache_manager import CacheManager


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def test_cache_set_get(self):
        """Test setting and getting from cache."""
        cache = CacheManager(enabled=True, ttl=3600, max_size=100)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = CacheManager(enabled=True, ttl=3600, max_size=100)
        assert cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = CacheManager(enabled=True, ttl=1, max_size=100)  # 1 second TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(1.1)  # Wait for expiration
        assert cache.get("key1") is None
    
    def test_cache_disabled(self):
        """Test cache when disabled."""
        cache = CacheManager(enabled=False)
        cache.set("key1", "value1")
        assert cache.get("key1") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = CacheManager(enabled=True, ttl=3600, max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = CacheManager(enabled=True, ttl=3600, max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.size() == 0
    
    def test_cache_size(self):
        """Test cache size tracking."""
        cache = CacheManager(enabled=True, ttl=3600, max_size=100)
        assert cache.size() == 0
        cache.set("key1", "value1")
        assert cache.size() == 1
        cache.set("key2", "value2")
        assert cache.size() == 2






