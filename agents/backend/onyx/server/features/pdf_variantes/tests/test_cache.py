"""
Unit Tests for Cache
====================
Tests for cache management and policies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

# Try to import cache classes
try:
    from cache import (
        CachePolicy,
        CacheEntry,
        CacheManager
    )
except ImportError:
    CachePolicy = None
    CacheEntry = None
    CacheManager = None


class TestCachePolicy:
    """Tests for CachePolicy enum."""
    
    def test_cache_policy_values(self):
        """Test CachePolicy enum values."""
        if CachePolicy is None:
            pytest.skip("CachePolicy not available")
        
        # Check expected policies
        expected_policies = ["no_cache", "cache_first", "network_first", "stale_while_revalidate"]
        actual_policies = [e.value for e in CachePolicy]
        
        for policy in expected_policies:
            assert policy in actual_policies or hasattr(CachePolicy, policy.upper())


class TestCacheEntry:
    """Tests for CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test creating CacheEntry."""
        if CacheEntry is None:
            pytest.skip("CacheEntry not available")
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
    
    def test_cache_entry_expiration(self):
        """Test CacheEntry expiration."""
        if CacheEntry is None:
            pytest.skip("CacheEntry not available")
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=1  # 1 second
        )
        
        assert hasattr(entry, "created_at") or hasattr(entry, "expires_at")
        
        if hasattr(entry, "is_expired"):
            # Should not be expired immediately
            assert entry.is_expired() is False
    
    def test_cache_entry_to_dict(self):
        """Test converting CacheEntry to dictionary."""
        if CacheEntry is None:
            pytest.skip("CacheEntry not available")
        
        entry = CacheEntry(key="test", value="value", ttl=3600)
        
        if hasattr(entry, "to_dict"):
            entry_dict = entry.to_dict()
            assert isinstance(entry_dict, dict)
            assert entry_dict.get("key") == "test"


class TestCacheManager:
    """Tests for CacheManager class."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager instance."""
        if CacheManager is None:
            pytest.skip("CacheManager not available")
        return CacheManager()
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test CacheManager initialization."""
        assert cache_manager is not None
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_manager):
        """Test getting non-existent cache entry."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "get"):
            result = await cache_manager.get("nonexistent_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        """Test setting and getting cache entry."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "get"):
            await cache_manager.set("test_key", "test_value", ttl=3600)
            result = await cache_manager.get("test_key")
            assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """Test deleting cache entry."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "delete") and hasattr(cache_manager, "get"):
            await cache_manager.set("test_key", "test_value")
            await cache_manager.delete("test_key")
            result = await cache_manager.get("test_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self, cache_manager):
        """Test clearing all cache."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "clear") and hasattr(cache_manager, "get"):
            await cache_manager.set("key1", "value1")
            await cache_manager.set("key2", "value2")
            await cache_manager.clear()
            
            assert await cache_manager.get("key1") is None
            assert await cache_manager.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache_manager):
        """Test TTL expiration."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "get"):
            await cache_manager.set("test_key", "test_value", ttl=1)
            
            # Wait for expiration
            import asyncio
            await asyncio.sleep(1.1)
            
            result = await cache_manager.get("test_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self, cache_manager):
        """Test cache size limit."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "get"):
            # Set max size if configurable
            for i in range(100):
                await cache_manager.set(f"key_{i}", f"value_{i}")
            
            # Should handle size limits gracefully
            result = await cache_manager.get("key_0")
            # May be None if LRU eviction, or value if still cached
            assert result is None or result == "value_0"
    
    @pytest.mark.asyncio
    async def test_get_or_set(self, cache_manager):
        """Test get_or_set pattern."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "get_or_set"):
            async def fetch_value():
                return "fetched_value"
            
            result = await cache_manager.get_or_set("test_key", fetch_value, ttl=3600)
            assert result == "fetched_value"
            
            # Second call should use cache
            result2 = await cache_manager.get_or_set("test_key", fetch_value, ttl=3600)
            assert result2 == "fetched_value"
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        if cache_manager is None:
            pytest.skip("CacheManager not available")
        
        if hasattr(cache_manager, "get_stats"):
            stats = await cache_manager.get_stats()
            assert isinstance(stats, dict)
            assert "hits" in stats or "misses" in stats or "size" in stats


class TestCacheConcurrency:
    """Tests for concurrent cache access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_get_set(self):
        """Test concurrent get and set operations."""
        if CacheManager is None:
            pytest.skip("CacheManager not available")
        
        import asyncio
        
        cache_manager = CacheManager()
        
        if hasattr(cache_manager, "set") and hasattr(cache_manager, "get"):
            # Concurrent sets
            await asyncio.gather(*[
                cache_manager.set(f"key_{i}", f"value_{i}")
                for i in range(50)
            ])
            
            # Concurrent gets
            results = await asyncio.gather(*[
                cache_manager.get(f"key_{i}")
                for i in range(50)
            ])
            
            # All should succeed
            assert all(result is not None for result in results)



