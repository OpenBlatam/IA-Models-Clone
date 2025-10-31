"""
Unit tests for Cache Service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from services.cache_service import AdvancedCacheService, cache_service

class TestAdvancedCacheService:
    """Test cases for AdvancedCacheService"""
    
    @pytest.fixture
    def cache_service(self):
        """Create cache service instance for testing"""
        config = {
            'redis_url': 'redis://localhost:6379',
            'cache': {
                'default_ttl': 3600,
                'max_memory': '100mb',
                'compression': True,
                'serialization': 'json'
            }
        }
        return AdvancedCacheService(config)
    
    @pytest.mark.asyncio
    async def test_get_miss(self, cache_service):
        """Test cache get on miss"""
        result = await cache_service.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_service):
        """Test cache set and get"""
        test_data = {"test": "data", "number": 123}
        
        # Set data
        success = await cache_service.set("test_key", test_data)
        assert success is True
        
        # Get data
        result = await cache_service.get("test_key")
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_service):
        """Test cache delete"""
        # Set data
        await cache_service.set("test_key", "test_value")
        
        # Verify it exists
        result = await cache_service.get("test_key")
        assert result == "test_value"
        
        # Delete it
        success = await cache_service.delete("test_key")
        assert success is True
        
        # Verify it's gone
        result = await cache_service.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_or_set(self, cache_service):
        """Test get_or_set functionality"""
        def factory():
            return "generated_value"
        
        # First call should generate value
        result = await cache_service.get_or_set("test_key", factory)
        assert result == "generated_value"
        
        # Second call should return cached value
        result = await cache_service.get_or_set("test_key", factory)
        assert result == "generated_value"
    
    @pytest.mark.asyncio
    async def test_get_or_set_async_factory(self, cache_service):
        """Test get_or_set with async factory"""
        async def async_factory():
            await asyncio.sleep(0.01)  # Simulate async operation
            return "async_generated_value"
        
        # First call should generate value
        result = await cache_service.get_or_set("test_key", async_factory)
        assert result == "async_generated_value"
        
        # Second call should return cached value
        result = await cache_service.get_or_set("test_key", async_factory)
        assert result == "async_generated_value"
    
    @pytest.mark.asyncio
    async def test_clear_namespace(self, cache_service):
        """Test clearing namespace"""
        # Set data in namespace
        await cache_service.set("key1", "value1", namespace="test")
        await cache_service.set("key2", "value2", namespace="test")
        await cache_service.set("key3", "value3", namespace="other")
        
        # Clear test namespace
        success = await cache_service.clear_namespace("test")
        assert success is True
        
        # Verify test namespace is cleared
        assert await cache_service.get("key1", namespace="test") is None
        assert await cache_service.get("key2", namespace="test") is None
        
        # Verify other namespace is intact
        assert await cache_service.get("key3", namespace="other") == "value3"
    
    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, cache_service):
        """Test invalidating by pattern"""
        # Set multiple keys
        await cache_service.set("user:1:profile", "profile1")
        await cache_service.set("user:1:settings", "settings1")
        await cache_service.set("user:2:profile", "profile2")
        await cache_service.set("post:1:content", "content1")
        
        # Invalidate user:1:* pattern
        count = await cache_service.invalidate_pattern("user:1:*")
        assert count >= 2
        
        # Verify user:1 keys are gone
        assert await cache_service.get("user:1:profile") is None
        assert await cache_service.get("user:1:settings") is None
        
        # Verify other keys remain
        assert await cache_service.get("user:2:profile") == "profile2"
        assert await cache_service.get("post:1:content") == "content1"
    
    @pytest.mark.asyncio
    async def test_warm_cache(self, cache_service):
        """Test cache warming"""
        def factory1():
            return "warmed_value1"
        
        def factory2():
            return "warmed_value2"
        
        warming_functions = {
            "key1": factory1,
            "key2": factory2
        }
        
        results = await cache_service.warm_cache(warming_functions)
        
        assert results["key1"] is True
        assert results["key2"] is True
        
        # Verify values are cached
        assert await cache_service.get("key1") == "warmed_value1"
        assert await cache_service.get("key2") == "warmed_value2"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, cache_service):
        """Test getting cache statistics"""
        # Perform some operations
        await cache_service.set("key1", "value1")
        await cache_service.get("key1")  # hit
        await cache_service.get("nonexistent")  # miss
        
        stats = await cache_service.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "local_cache_size" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache_service):
        """Test cleanup of expired entries"""
        # This is a simple test since we don't track TTL in local cache
        count = await cache_service.cleanup_expired()
        assert isinstance(count, int)
    
    def test_cache_decorator(self, cache_service):
        """Test cache decorator"""
        call_count = 0
        
        @cache_service.cache_decorator(ttl=3600)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
    
    def test_cache_decorator_with_key_func(self, cache_service):
        """Test cache decorator with custom key function"""
        call_count = 0
        
        def key_func(x, y):
            return f"custom_key_{x}_{y}"
        
        @cache_service.cache_decorator(ttl=3600, key_func=key_func)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(3, 4)
        assert result1 == 7
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(3, 4)
        assert result2 == 7
        assert call_count == 1  # Should not increment
    
    @pytest.mark.asyncio
    async def test_serialization_json(self, cache_service):
        """Test JSON serialization"""
        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        await cache_service.set("test_key", test_data)
        result = await cache_service.get("test_key")
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_serialization_pickle(self, cache_service):
        """Test Pickle serialization"""
        cache_service.cache_config.serialization = "pickle"
        
        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        await cache_service.set("test_key", test_data)
        result = await cache_service.get("test_key")
        
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache_service):
        """Test error handling"""
        # Test with invalid data
        result = await cache_service.get("invalid_key")
        assert result is None
        
        # Test delete non-existent key
        success = await cache_service.delete("nonexistent_key")
        assert success is True  # Should not fail
    
    @pytest.mark.asyncio
    async def test_close(self, cache_service):
        """Test closing cache service"""
        await cache_service.close()
        # Should not raise any exceptions
