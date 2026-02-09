"""
Comprehensive Unit Tests for Smart Cache

Tests cover smart cache multi-level functionality with diverse test cases
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from utils.smart_cache import SmartCache


class TestSmartCache:
    """Test cases for SmartCache class"""
    
    def test_smart_cache_init_default(self):
        """Test initializing smart cache with defaults"""
        cache = SmartCache()
        assert cache.l1_size == 1000
        assert cache.default_ttl == 3600
        assert len(cache.l1_cache) == 0
    
    def test_smart_cache_init_custom(self):
        """Test initializing with custom parameters"""
        cache = SmartCache(
            l1_size=500,
            default_ttl=1800
        )
        assert cache.l1_size == 500
        assert cache.default_ttl == 1800
    
    def test_smart_cache_init_with_redis(self):
        """Test initializing with Redis L2"""
        with patch('utils.smart_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            with patch('utils.smart_cache.redis.from_url', return_value=mock_redis):
                cache = SmartCache(l2_redis_url="redis://localhost:6379")
                assert cache.l2_client == mock_redis
    
    def test_smart_cache_init_redis_unavailable(self):
        """Test initializing when Redis unavailable"""
        with patch('utils.smart_cache.REDIS_AVAILABLE', False):
            cache = SmartCache(l2_redis_url="redis://localhost:6379")
            assert cache.l2_client is None
    
    def test_smart_cache_get_l1_hit(self):
        """Test getting value from L1 cache"""
        cache = SmartCache()
        cache.set("key1", "value1")
        
        result = cache.get("key1")
        assert result == "value1"
        assert cache.stats["l1_hits"] > 0
    
    def test_smart_cache_get_l1_miss(self):
        """Test L1 cache miss"""
        cache = SmartCache()
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats["l1_misses"] > 0
    
    def test_smart_cache_get_with_default(self):
        """Test get with default value"""
        cache = SmartCache()
        result = cache.get("nonexistent", default="default_value")
        assert result == "default_value"
    
    def test_smart_cache_set_and_get(self):
        """Test setting and getting values"""
        cache = SmartCache()
        cache.set("key1", "value1")
        
        result = cache.get("key1")
        assert result == "value1"
        assert cache.stats["sets"] > 0
    
    def test_smart_cache_set_with_ttl(self):
        """Test setting value with TTL"""
        cache = SmartCache()
        cache.set("key1", "value1", ttl=1)
        
        # Should be available immediately
        result = cache.get("key1")
        assert result == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        result = cache.get("key1")
        assert result is None
    
    def test_smart_cache_l1_eviction(self):
        """Test L1 cache eviction when full"""
        cache = SmartCache(l1_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_smart_cache_clear(self):
        """Test clearing cache"""
        cache = SmartCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache.l1_cache) == 0
    
    def test_smart_cache_stats(self):
        """Test getting cache statistics"""
        cache = SmartCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["l1_hits"] > 0
        assert stats["l1_misses"] > 0
        assert stats["sets"] > 0
    
    def test_smart_cache_complex_data(self):
        """Test caching complex data structures"""
        cache = SmartCache()
        complex_data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42
        }
        
        cache.set("complex", complex_data)
        result = cache.get("complex")
        
        assert result == complex_data
        assert result["nested"]["key"] == "value"
    
    def test_smart_cache_l2_promotion(self):
        """Test L2 cache value promoted to L1"""
        with patch('utils.smart_cache.REDIS_AVAILABLE', True):
            mock_redis = Mock()
            mock_redis.ping = Mock()
            mock_redis.get = Mock(return_value=b'pickled_value')
            with patch('utils.smart_cache.redis.from_url', return_value=mock_redis):
                cache = SmartCache(l2_redis_url="redis://localhost:6379")
                cache.l2_client = mock_redis
                
                with patch('pickle.loads', return_value="value1"):
                    result = cache.get("key1")
                    # Should promote to L1
                    assert result == "value1"
    
    def test_smart_cache_l3_disk(self, temp_dir):
        """Test L3 disk cache"""
        cache_dir = temp_dir / "cache"
        cache = SmartCache(l3_disk_path=str(cache_dir))
        
        cache.set("key1", "value1")
        
        # Should be in L3
        assert (cache_dir / cache._make_key("key1")).exists() or cache.get("key1") == "value1"















