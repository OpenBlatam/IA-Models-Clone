"""
Comprehensive Unit Tests for Fast Cache System

Tests cover LRUCache and FastCache with diverse test cases
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from core.fast_cache import LRUCache, FastCache, get_fast_cache


class TestLRUCache:
    """Test cases for LRUCache class"""
    
    def test_lru_cache_get_existing(self):
        """Test getting existing key from cache"""
        cache = LRUCache(maxsize=3)
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
    
    def test_lru_cache_get_missing(self):
        """Test getting missing key returns None"""
        cache = LRUCache()
        result = cache.get("missing")
        assert result is None
    
    def test_lru_cache_set_and_get(self):
        """Test setting and getting values"""
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
    
    def test_lru_cache_maxsize_eviction(self):
        """Test that cache evicts oldest when full"""
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_cache_lru_ordering(self):
        """Test LRU ordering - least recently used is evicted"""
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Make key1 recently used
        cache.set("key3", "value3")  # Should evict key2 (least recently used)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_lru_cache_delete(self):
        """Test deleting key from cache"""
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_lru_cache_delete_missing(self):
        """Test deleting missing key doesn't raise error"""
        cache = LRUCache()
        cache.delete("missing")  # Should not raise
    
    def test_lru_cache_clear(self):
        """Test clearing cache"""
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache.cache) == 0
    
    def test_lru_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert "hit_rate" in stats
    
    def test_lru_cache_hit_rate_calculation(self):
        """Test hit rate calculation"""
        cache = LRUCache()
        cache.set("key1", "value1")
        
        # 3 hits, 2 misses
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")
        cache.get("missing1")
        cache.get("missing2")
        
        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert float(stats["hit_rate"].rstrip("%")) == pytest.approx(60.0, rel=0.1)
    
    def test_lru_cache_zero_hit_rate(self):
        """Test hit rate with no hits"""
        cache = LRUCache()
        cache.get("missing1")
        cache.get("missing2")
        
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert float(stats["hit_rate"].rstrip("%")) == 0.0


class TestFastCache:
    """Test cases for FastCache class"""
    
    @pytest.mark.asyncio
    async def test_fast_cache_get_l1_hit(self):
        """Test getting value from L1 cache"""
        cache = FastCache()
        cache._l1_cache.set("key1", "value1")
        
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_fast_cache_get_miss(self):
        """Test getting missing key returns None"""
        cache = FastCache()
        result = await cache.get("missing")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fast_cache_set_and_get(self):
        """Test setting and getting values"""
        cache = FastCache()
        await cache.set("key1", "value1")
        
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_fast_cache_ttl_expiration(self):
        """Test TTL expiration"""
        cache = FastCache()
        await cache.set("key1", "value1", ttl=0.1)  # Very short TTL
        
        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fast_cache_ttl_no_expiration(self):
        """Test value doesn't expire with long TTL"""
        cache = FastCache()
        await cache.set("key1", "value1", ttl=3600)
        
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_fast_cache_l2_promotion(self):
        """Test L2 cache value is promoted to L1"""
        cache = FastCache()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps("value1"))
        cache.set_l2_cache(mock_redis)
        
        result = await cache.get("key1")
        
        assert result == "value1"
        # Should be in L1 now
        assert cache._l1_cache.get("key1") == "value1"
    
    @pytest.mark.asyncio
    async def test_fast_cache_l2_set(self):
        """Test setting value in L2 cache"""
        cache = FastCache()
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        cache.set_l2_cache(mock_redis)
        
        await cache.set("key1", "value1", ttl=3600)
        
        # Verify L2 was called
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "key1"
        assert call_args[1]["ttl"] == 3600
    
    @pytest.mark.asyncio
    async def test_fast_cache_l2_error_handling(self):
        """Test L2 cache errors don't break functionality"""
        cache = FastCache()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis error"))
        cache.set_l2_cache(mock_redis)
        
        # Should not raise, should return None
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fast_cache_delete(self):
        """Test deleting from cache"""
        cache = FastCache()
        await cache.set("key1", "value1")
        cache.delete("key1")
        
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fast_cache_delete_l2(self):
        """Test deleting from L2 cache"""
        cache = FastCache()
        mock_redis = AsyncMock()
        mock_redis.delete = MagicMock()
        cache.set_l2_cache(mock_redis)
        
        cache.delete("key1")
        mock_redis.delete.assert_called_once_with("key1")
    
    def test_fast_cache_stats(self):
        """Test cache statistics"""
        cache = FastCache()
        cache._l1_cache.set("key1", "value1")
        
        stats = cache.stats()
        assert "l1" in stats
        assert isinstance(stats["l1"], dict)
    
    def test_fast_cache_stats_with_l2(self):
        """Test statistics with L2 cache enabled"""
        cache = FastCache()
        mock_redis = AsyncMock()
        cache.set_l2_cache(mock_redis)
        
        stats = cache.stats()
        assert "l2" in stats
        assert stats["l2"] == "enabled"
    
    @pytest.mark.asyncio
    async def test_fast_cache_complex_object(self):
        """Test caching complex objects"""
        cache = FastCache()
        complex_obj = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42
        }
        
        await cache.set("complex", complex_obj)
        result = await cache.get("complex")
        
        assert result == complex_obj
        assert result["nested"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_fast_cache_string_value(self):
        """Test caching string values in L2"""
        cache = FastCache()
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        cache.set_l2_cache(mock_redis)
        
        await cache.set("key1", "string_value", ttl=3600)
        
        # Verify string was passed directly
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][1] == "string_value"


class TestGetFastCache:
    """Test cases for get_fast_cache function"""
    
    def test_get_fast_cache_singleton(self):
        """Test that get_fast_cache returns singleton"""
        cache1 = get_fast_cache()
        cache2 = get_fast_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, FastCache)
    
    def test_get_fast_cache_multiple_calls(self):
        """Test multiple calls return same instance"""
        caches = [get_fast_cache() for _ in range(10)]
        assert all(cache is caches[0] for cache in caches)















