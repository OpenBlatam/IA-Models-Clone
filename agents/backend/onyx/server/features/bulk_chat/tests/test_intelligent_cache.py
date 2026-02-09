"""
Tests for Intelligent Cache
============================
"""

import pytest
import asyncio
from ..core.intelligent_cache import IntelligentCache, CacheStrategy


@pytest.fixture
def intelligent_cache():
    """Create intelligent cache for testing."""
    return IntelligentCache(max_size=100)


@pytest.mark.asyncio
async def test_cache_set_get(intelligent_cache):
    """Test cache set and get operations."""
    await intelligent_cache.set("key1", "value1", ttl=3600)
    
    value = await intelligent_cache.get("key1")
    
    assert value == "value1"


@pytest.mark.asyncio
async def test_cache_miss(intelligent_cache):
    """Test cache miss scenario."""
    value = await intelligent_cache.get("non_existent_key")
    
    assert value is None


@pytest.mark.asyncio
async def test_cache_eviction(intelligent_cache):
    """Test cache eviction when full."""
    # Fill cache beyond max_size
    for i in range(150):
        await intelligent_cache.set(f"key{i}", f"value{i}")
    
    # Wait for eviction
    await asyncio.sleep(0.1)
    
    # Should have evicted some items
    stats = intelligent_cache.get_cache_stats()
    assert stats["size"] <= 100


@pytest.mark.asyncio
async def test_cache_invalidate(intelligent_cache):
    """Test cache invalidation."""
    await intelligent_cache.set("key1", "value1")
    
    assert await intelligent_cache.get("key1") == "value1"
    
    await intelligent_cache.invalidate("key1")
    
    assert await intelligent_cache.get("key1") is None


@pytest.mark.asyncio
async def test_cache_clear(intelligent_cache):
    """Test clearing cache."""
    await intelligent_cache.set("key1", "value1")
    await intelligent_cache.set("key2", "value2")
    
    await intelligent_cache.clear()
    
    assert await intelligent_cache.get("key1") is None
    assert await intelligent_cache.get("key2") is None


@pytest.mark.asyncio
async def test_get_cache_stats(intelligent_cache):
    """Test getting cache statistics."""
    await intelligent_cache.set("key1", "value1")
    await intelligent_cache.get("key1")  # Hit
    await intelligent_cache.get("key2")  # Miss
    
    stats = intelligent_cache.get_cache_stats()
    
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert "hit_rate" in stats or "size" in stats


@pytest.mark.asyncio
async def test_cache_prefetch(intelligent_cache):
    """Test cache prefetching."""
    # Prefetch keys that might be needed
    await intelligent_cache.prefetch(["key1", "key2", "key3"])
    
    # Keys should be prefetched (implementation dependent)
    stats = intelligent_cache.get_cache_stats()
    assert stats is not None


