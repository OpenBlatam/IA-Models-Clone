"""
Tests for Response Cache
========================
"""

import pytest
from bulk_chat.core.response_cache import ResponseCache


@pytest.mark.asyncio
async def test_cache_set_get():
    """Test basic cache operations."""
    cache = ResponseCache(max_size=10, ttl_seconds=3600)
    
    messages = [{"role": "user", "content": "Hello"}]
    response = "This is a test response"
    
    # Set cache
    await cache.set(messages, response)
    
    # Get from cache
    cached = await cache.get(messages)
    
    assert cached == response


@pytest.mark.asyncio
async def test_cache_miss():
    """Test cache miss scenario."""
    cache = ResponseCache()
    
    messages = [{"role": "user", "content": "Hello"}]
    
    # Get from empty cache
    cached = await cache.get(messages)
    
    assert cached is None


@pytest.mark.asyncio
async def test_cache_stats():
    """Test cache statistics."""
    cache = ResponseCache(max_size=10)
    
    messages = [{"role": "user", "content": "Test"}]
    
    # Miss
    await cache.get(messages)
    
    # Set and get (hit)
    await cache.set(messages, "Response")
    await cache.get(messages)
    
    stats = cache.get_stats()
    
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clearing."""
    cache = ResponseCache()
    
    await cache.set([{"role": "user", "content": "Test"}], "Response")
    assert cache.get_stats()["size"] == 1
    
    await cache.clear()
    assert cache.get_stats()["size"] == 0
    assert cache.get_stats()["hits"] == 0
    assert cache.get_stats()["misses"] == 0
































