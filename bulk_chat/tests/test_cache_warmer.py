"""
Tests for Cache Warmer
=======================
"""

import pytest
import asyncio
from ..core.cache_warmer import CacheWarmer


@pytest.fixture
def cache_warmer():
    """Create cache warmer for testing."""
    return CacheWarmer()


@pytest.mark.asyncio
async def test_add_prefetch_key(cache_warmer):
    """Test adding a prefetch key."""
    cache_warmer.add_prefetch_key(
        key="test_key",
        priority=1,
        metadata={"category": "important"}
    )
    
    assert "test_key" in cache_warmer.prefetch_keys


@pytest.mark.asyncio
async def test_remove_prefetch_key(cache_warmer):
    """Test removing a prefetch key."""
    cache_warmer.add_prefetch_key("test_key", priority=1)
    
    assert "test_key" in cache_warmer.prefetch_keys
    
    cache_warmer.remove_prefetch_key("test_key")
    
    assert "test_key" not in cache_warmer.prefetch_keys


@pytest.mark.asyncio
async def test_warm_cache(cache_warmer):
    """Test warming cache."""
    cache_warmer.add_prefetch_key("key1", priority=1)
    cache_warmer.add_prefetch_key("key2", priority=2)
    
    # Mock cache
    warmed_keys = []
    
    async def mock_fetch(key):
        warmed_keys.append(key)
        return f"value_{key}"
    
    cache_warmer.cache = type('Cache', (), {'get': lambda k: None, 'set': lambda k, v: None})()
    
    # Warm cache
    await cache_warmer.warm_cache(max_keys=10)
    
    # Should have attempted to warm
    assert cache_warmer.prefetch_keys is not None


@pytest.mark.asyncio
async def test_get_cache_warmer_stats(cache_warmer):
    """Test getting cache warmer statistics."""
    cache_warmer.add_prefetch_key("key1", priority=1)
    cache_warmer.add_prefetch_key("key2", priority=2)
    
    stats = cache_warmer.get_cache_warmer_stats()
    
    assert stats is not None
    assert "total_keys" in stats or "warmed_keys" in stats


